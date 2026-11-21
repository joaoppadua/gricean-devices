"""Turn completion (spec 13.1): a post hoc analysis, added after the results and not part of the original design.

Whether a greedy response ended by itself before the limit of new tokens, and whether a response coded
`intended` did so (a complete turn). English coded rows only. The count of tokens is obtained by tokenizing
the stored text again; the tolerance that this needs is computed from committed calibration rows."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import pandas as pd

from .coding import INTENT, _all_responses, load_sheet, select_coded, select_main
from .io import read_jsonl, write_jsonl
from .report import INDEX, _condition

LANGUAGE = "en"                      # author's decision: no Portuguese response is counted
MAX_TOLERANCE = 10                   # beyond this the route by re-tokenizing is abandoned
MIN_AT_LIMIT = 20                    # calibration responses that must have reached the limit
COMPARISON = ("olmo_n500", "olmo_full")     # the one comparison across data scale, fixed in advance
CODES = ("correct", "incorrect")

# Twenty English prompts, none of them a stimulus (`check_prompts`): six requests that invite a long answer,
# six short questions, eight openings of running text.
CALIBRATION_PROMPTS = [
    "Write a detailed essay on the causes of the French Revolution.",
    "Explain step by step how a bill becomes law in the United States.",
    "Describe the life cycle of a star, from nebula to remnant.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List twenty common kitchen utensils and say what each one is used for.",
    "Compare and contrast classical and operant conditioning in detail.",
    "What is the capital of Australia?",
    "How many legs does a spider have?",
    "Who wrote Pride and Prejudice?",
    "What is the chemical symbol for gold?",
    "Name three primary colors.",
    "What is 17 plus 26?",
    "The train left the station ten minutes late, and by the time it reached the river",
    "Minutes of the meeting held on 3 March. Present: the chair, the treasurer and four members. Item 1:",
    "Dear Ms. Alvarez, thank you for your letter of last week. I am writing to",
    "Chapter 2\n\nThe house at the end of the lane had been empty for years.",
    "Q: What is the tallest animal?\nA: The giraffe.\nQ: What is the fastest land animal?\nA:",
    "Ingredients: 200 g flour, 2 eggs, 300 ml milk, a pinch of salt. Method:",
    "1. Wash the rice. 2. Bring the water to a boil. 3.",
    "The following table lists the planets of the solar system in order of distance from the sun:",
]


def count_tokens(tokenizer, text: str) -> int:
    """Tokens of a stored response: the text alone, no special tokens."""
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def load_tokenizer(spec):
    from transformers import AutoTokenizer

    from .lm import resolve_revision
    return AutoTokenizer.from_pretrained(spec.repo, revision=resolve_revision(spec.repo, spec.revision))


def write_lengths(resp_root: Path, out: Path, expected: dict[str, bool], tokenizer_for: Callable,
                  language: str = LANGUAGE) -> int:
    """Count the tokens of every coded response in `language`, each with the tokenizer of its model."""
    df = select_coded(_all_responses(resp_root), expected)
    df = df[df["language"] == language]
    toks = {model: tokenizer_for(model) for model in sorted(df["model"].unique())}
    table = pd.DataFrame({"row_id": df["row_id"].to_numpy(),
                          "n_tokens": [count_tokens(toks[m], r) for m, r in zip(df["model"], df["response"])]})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.sort_values("row_id").to_csv(out, index=False)
    return len(table)


def load_lengths(path: Path) -> pd.Series:
    return pd.read_csv(path, dtype={"row_id": str, "n_tokens": int}).set_index("row_id")["n_tokens"]


def check_prompts(prompts: list[str], stimulus_prompts: list[str]) -> None:
    """Calibration runs on text outside the experiment."""
    clash = sorted(set(prompts) & set(stimulus_prompts))
    if clash:
        raise ValueError(f"calibration prompt(s) that are stimuli: {clash[:3]}")


def calibrate(lm, prompts: list[str], *, template: bool, max_new_tokens: int) -> list[dict]:
    """Greedy responses with their true number of new tokens, whether the model produced end-of-sequence,
    and the count obtained from the decoded text (`n_recount`), which is all a stored response allows."""
    rows = []
    for i, prompt in enumerate(prompts):
        tokens = list(lm.generate_tokens(lm.format(prompt, template), max_new_tokens=max_new_tokens,
                                         temperature=None, seed=None))
        eos = len(tokens) < max_new_tokens or tokens[-1] in lm.eos_ids
        rows.append({"model": lm.name, "template": template, "prompt": i,
                     "n_generated": sum(t not in lm.eos_ids for t in tokens), "eos": bool(eos),
                     "n_recount": count_tokens(lm.tokenizer, lm.decode(tokens))})
    return rows


def save_calibration(path: Path, model: str, rows: list[dict]) -> None:
    """Put one model's calibration rows in the file, in place of any it already has there."""
    kept = [r for r in read_jsonl(path) if r["model"] != model]
    write_jsonl(path, sorted(kept + rows, key=lambda r: (r["model"], r["template"], r["prompt"])))


def tolerance(cal: pd.DataFrame, cap: int, *, min_at_limit: int = MIN_AT_LIMIT,
              max_tolerance: int = MAX_TOLERANCE) -> int:
    """The largest shortfall of the recount among calibration responses that reached the limit (spec 13.1)."""
    at_limit = cal[~cal["eos"].astype(bool)]
    if len(at_limit) < min_at_limit:
        raise ValueError(f"{len(at_limit)} calibration response(s) reached the limit; {min_at_limit} are needed")
    tol = max(0, int((cap - at_limit["n_recount"]).max()))
    if tol > max_tolerance:
        raise ValueError(f"tolerance {tol} larger than {max_tolerance}: counting by re-tokenizing is abandoned")
    return tol


def flag(coded: pd.DataFrame, lengths: pd.Series, models: list[str], expected: dict[str, bool], *,
         cap: int, tolerance: int, language: str = LANGUAGE) -> pd.DataFrame:
    """Coded rows of `models` in their main condition and in `language`, with `ended`, `at_limit`, `in_band`.

    A response reached the limit when its count is at least `cap - tolerance`. `in_band` marks those below
    `cap`: the responses a tolerance of 0 would read as ended."""
    df = select_main(coded[coded["model"].isin(models)], expected)
    df = df[df["language"] == language].copy()
    n = df["row_id"].map(lengths)
    if n.isna().any():
        raise ValueError(f"no token count for {int(n.isna().sum())} coded row(s): "
                         f"{df.loc[n.isna(), 'row_id'].tolist()[:5]} (run `gricean turns --count`)")
    df["n_tokens"] = n.astype(int)
    df["at_limit"] = df["n_tokens"] >= cap - tolerance
    df["ended"] = ~df["at_limit"]
    df["in_band"] = df["at_limit"] & (df["n_tokens"] < cap)
    df["condition"] = _condition(df)
    return df


def _cells(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[INDEX].drop_duplicates()).sort_values()


def termination_table(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """For each intent: how many responses carry it (`n`) and the share of them that ended.

    Rows and conditions are those of the intent table. Where no response has the intent the share is empty;
    a model without rows is missing in both."""
    cells, out = _cells(df), {}
    for model in models:
        rows = df[df["model"] == model]
        for intent in INTENT:
            ended = rows[rows["intent"] == intent].groupby(INDEX)["ended"]
            n = ended.size().reindex(cells, fill_value=0).astype("Int64")
            out[(model, intent, "n")] = n if len(rows) else n.mask(n == 0)
            out[(model, intent, "ended")] = ended.mean().reindex(cells).astype(float)
    return pd.DataFrame(out, index=cells)


def termination_correctness_table(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Responses coded `intended`, by their correctness code: how many (`n`) and how many reached the limit.

    Among the `incorrect` this sets the responses that did not end apart from the other errors; a `correct`
    response at the limit is a long answer cut off."""
    intended = df[df["intent"] == "intended"]
    unjudged = intended[~intended["correctness"].isin(CODES)]
    if len(unjudged):
        raise ValueError(f"{len(unjudged)} response(s) coded `na` under intended: {unjudged['row_id'].tolist()[:5]}")
    cells, out = _cells(df), {}
    for model in models:
        has_rows = bool((df["model"] == model).any())
        rows = intended[intended["model"] == model]
        for code in CODES:
            at_limit = rows[rows["correctness"] == code].groupby(INDEX)["at_limit"]
            n = at_limit.size().reindex(cells, fill_value=0).astype("Int64")
            k = at_limit.sum().reindex(cells, fill_value=0).astype("Int64")
            out[(model, code, "n")] = n if has_rows else n.mask(n == 0)
            out[(model, code, "at_limit")] = k if has_rows else k.mask(n == 0)
    return pd.DataFrame(out, index=cells)


def mcnemar_exact(only_first: int, only_second: int) -> float:
    """Exact two-sided McNemar test: a binomial test, at one half, on the items where the two differ."""
    n = only_first + only_second
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_first, only_second) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def _complete(df: pd.DataFrame, model: str, version: str) -> pd.Series:
    """item -> complete turn (coded `intended` and ended), on the dissociation items of one version."""
    rows = df[(df["model"] == model) & ~df["control"].astype(bool) & (df["version"] == version)]
    if rows["item_id"].duplicated().any():
        raise ValueError(f"{model}: an item more than once in version {version}")
    return ((rows["intent"] == "intended") & rows["ended"]).set_axis(rows["item_id"]).sort_index()


def paired(df: pd.DataFrame, first: str, second: str, version: str = "full") -> dict:
    """Complete turns of two models on the same dissociation items: the four counts and the exact test."""
    a, b = _complete(df, first, version), _complete(df, second, version)
    if not a.index.equals(b.index):
        raise ValueError(f"{first} and {second} are not coded on the same items ({len(a)} and {len(b)})")
    only_first, only_second = int((a & ~b).sum()), int((~a & b).sum())
    return {"n": len(a), "both": int((a & b).sum()), "only_first": only_first, "only_second": only_second,
            "neither": int((~a & ~b).sum()), "p": mcnemar_exact(only_first, only_second)}


def build(coded_path: Path, lengths_path: Path, calibration_path: Path, models: list[str], out_dir: Path,
          expected: dict[str, bool], *, cap: int, comparison: tuple[str, str] | None = None,
          min_at_limit: int = MIN_AT_LIMIT) -> dict:
    """Write the two tables and the record of the tolerance, of the counts it bears on and of the comparison."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = pd.DataFrame(read_jsonl(calibration_path))
    tol = tolerance(cal, cap, min_at_limit=min_at_limit)
    df = flag(load_sheet(coded_path), load_lengths(lengths_path), models, expected, cap=cap, tolerance=tol)
    for name, table in (("termination", termination_table(df, models)),
                        ("termination_correctness", termination_correctness_table(df, models))):
        table.to_csv(out_dir / f"{name}.csv")
        (out_dir / f"{name}.tex").write_text(table.to_latex(float_format="%.2f", na_rep="--"))
    eos = cal["eos"].astype(bool)
    turns = {m: _complete(df, m, "full") for m in models if (df["model"] == m).any()}
    lines = [
        "# turn completion (spec 13.1): post hoc, English coded rows only",
        f"limit={cap} tolerance={tol}",
        f"calibration responses={len(cal)} reached_limit={int((~eos).sum())} ended={int(eos.sum())} "
        f"ended_inside_band={int((eos & (cal['n_recount'] >= cap - tol)).sum())}",
        f"coded responses={len(df)} reached_limit={int(df['at_limit'].sum())} inside_band={int(df['in_band'].sum())}",
        "# complete turns (coded `intended` and ended) on the dissociation items, `full` version",
        "complete_turns " + " ".join(f"{m}={int(t.sum())}/{len(t)}" for m, t in turns.items()),
    ]
    summary = {"tolerance": tol, "n": int(len(df)), "at_limit": int(df["at_limit"].sum()),
               "in_band": int(df["in_band"].sum()), "complete_turns": {m: int(t.sum()) for m, t in turns.items()},
               "comparison": None}
    if comparison is not None:
        got = paired(df, *comparison)
        lines += ["# exploratory: read as fixed beforehand in spec 13.1",
                  f"{comparison[0]} vs {comparison[1]}: n={got['n']} both={got['both']} "
                  f"only_first={got['only_first']} only_second={got['only_second']} neither={got['neither']} "
                  f"mcnemar_exact_p={got['p']:.4f}"]
        summary["comparison"] = got
    (out_dir / "termination.txt").write_text("\n".join(lines) + "\n")
    return summary
