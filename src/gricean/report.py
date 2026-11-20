from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .coding import INTENT, agreement, load_sheet, select_main
from .io import read_jsonl

INDEX = ["category", "condition"]
MAIN_LANGUAGE = "en"            # its tables carry no suffix and feed the figure; the others get `_<language>`


def _one_language(df: pd.DataFrame, language: str | None) -> pd.DataFrame:
    """Rows of one language. Portuguese is reported separately and never pooled with English (spec 4.4):
    without `language`, the rows must already be in a single language."""
    if language is not None:
        return df[df["language"] == language]
    if df["language"].nunique() > 1:
        raise ValueError(f"rows in more than one language {sorted(df['language'].unique())}: pass language=")
    return df


def _languages(df: pd.DataFrame | None) -> list[str]:
    """Languages present, the main one first."""
    if df is None or df.empty or "language" not in df.columns:
        return []
    return sorted(df["language"].unique(), key=lambda lang: (lang != MAIN_LANGUAGE, lang))


def _suffix(language: str) -> str:
    return "" if language == MAIN_LANGUAGE else f"_{language}"


def select_template_control(df: pd.DataFrame, expected: dict[str, bool]) -> pd.DataFrame:
    """Raw rows of tuned models (full/stripped): the 'template alone does not explain it' control."""
    exp = df["model"].map(expected)
    return df[(exp == True) & ~df["template"] & (df["version"] != "embedded")]  # noqa: E712


def _condition(df: pd.DataFrame) -> pd.Series:
    if "control" in df.columns:
        return np.where(df["control"], "control", df["version"])
    return df["version"]


def _rates(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    df = df.assign(condition=_condition(df))
    counts = df.groupby(INDEX + ["model", "intent"]).size()
    totals = df.groupby(INDEX + ["model"]).size()
    rates = (counts / totals).rename("rate").reset_index()
    table = rates.pivot_table(index=INDEX, columns=["model", "intent"], values="rate", fill_value=0.0)
    cols = pd.MultiIndex.from_product([models, list(INTENT)])
    table = table.reindex(columns=cols)
    seen = table.columns.get_level_values(0).isin(set(df["model"]))   # a model with rows that never shows an
    table.loc[:, seen] = table.loc[:, seen].fillna(0.0)              # intent: rate 0; a model without rows: missing
    return table


def intent_table(coded: pd.DataFrame, models: list[str], expected: dict[str, bool],
                 language: str | None = None) -> pd.DataFrame:
    """Rate of literal / intended / neither responses by (category, condition) and model, in one language.

    Conditions are the stimulus versions plus 'control' for control items."""
    df = select_main(coded[coded["model"].isin(models)], expected)
    return _rates(_one_language(df, language), models)


def template_control_table(coded: pd.DataFrame, models: list[str], expected: dict[str, bool],
                           language: str | None = None) -> pd.DataFrame:
    df = select_template_control(coded[coded["model"].isin(models)], expected)
    return _rates(_one_language(df, language), models)


JUDGED = ("intended", "literal")     # the intents under which `correct` means right (spec 5.1, step 17)


def correctness_table(coded: pd.DataFrame, models: list[str], expected: dict[str, bool],
                      language: str | None = None) -> pd.DataFrame:
    """For the responses that follow a reading: how many there are (`n`) and the share of them coded `correct`.

    Rows and conditions are those of `intent_table`. `neither` has no column: under it the code says whether
    the text is well formed, not whether it is right. Where no response follows the reading, `n` is 0 and the
    share is empty; a model without rows is missing in both."""
    df = _one_language(select_main(coded[coded["model"].isin(models)], expected), language)
    df = df.assign(condition=_condition(df), right=df["correctness"] == "correct")
    unjudged = df[df["intent"].isin(JUDGED) & (df["correctness"] == "na")]
    if len(unjudged):
        raise ValueError(f"{len(unjudged)} response(s) coded `na` under intended or literal: "
                         f"{unjudged['row_id'].tolist()[:5]}")
    cells = pd.MultiIndex.from_frame(df[INDEX].drop_duplicates()).sort_values()
    out = {}
    for model in models:
        rows = df[df["model"] == model]
        for intent in JUDGED:
            right = rows[rows["intent"] == intent].groupby(INDEX)["right"]
            n = right.size().reindex(cells, fill_value=0).astype("Int64")
            out[(model, intent, "n")] = n if len(rows) else n.mask(n == 0)
            out[(model, intent, "correct")] = right.mean().reindex(cells).astype(float)
    return pd.DataFrame(out, index=cells)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_rows(root: Path) -> pd.DataFrame:
    """Every row of the JSONL files of one stage (`data/logprobs`, `data/responses`)."""
    rows = []
    for path in sorted(Path(root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)


def logodds_table(lp_root: Path, models: list[str], expected: dict[str, bool],
                  language: str | None = None) -> pd.DataFrame:
    df = _load_rows(lp_root)
    if df.empty:
        raise ValueError(f"no log-prob rows under {lp_root}")
    return _logodds(_one_language(select_main(df[df["model"].isin(models)], expected), language), models)


def logodds_template_control_table(lp_root: Path, models: list[str], expected: dict[str, bool],
                                   language: str | None = None) -> pd.DataFrame:
    """Line 2 on the raw passes of tuned models: the template control, scored without a judge.

    Columns are the models that have a raw control, in ladder order."""
    df = _load_rows(lp_root)
    df = _one_language(select_template_control(df[df["model"].isin(models)], expected), language) if not df.empty else df
    if df.empty:
        raise ValueError(f"no template-control log-prob rows under {lp_root}")
    return _logodds(df, [m for m in models if m in set(df["model"])])


def _logodds(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    df = df.assign(condition=lambda d: d["version"])
    out = {}
    for (cat, cond, model), g in df.groupby(INDEX + ["model"]):
        x = g["log_odds"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(x)
        out.setdefault((cat, cond), {}).update({
            (model, "mean"): float(x.mean()), (model, "ci_low"): lo, (model, "ci_high"): hi,
            (model, "share_neither"): float(g["share_neither"].mean())})
    table = pd.DataFrame.from_dict(out, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=INDEX)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    cols = pd.MultiIndex.from_product([models, ["mean", "ci_low", "ci_high", "share_neither"]])
    return table.reindex(columns=cols)


# Printed when the greedy response is empty (EOS was the top first token). A finding, not a gap: the
# model treated the prompt as complete. Spec §5.1.
EMPTY_RESPONSE = "⟨empty: model emitted end-of-sequence⟩"


EXAMPLES = {"per_rung": 3, "per_category": 1, "seed": 0}     # the draw of the printed examples (spec 6)


def _draw_rank(seed: int, *key: str) -> str:
    """A fixed pseudo-random rank: sorting by it shuffles, and the first n of a longer draw are the shorter one."""
    return hashlib.sha256("|".join([str(seed), *key]).encode()).hexdigest()


def _quoted(text: str) -> str:
    """Text as a fenced block, so that nothing in it (`### Instruction:`, a fence) is read as markup."""
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}\n"


def examples(resp_root: Path, models: list[str], expected: dict[str, bool], language: str | None = None,
             coded: pd.DataFrame | None = None, per_rung: int = EXAMPLES["per_rung"],
             per_category: int = EXAMPLES["per_category"], seed: int = EXAMPLES["seed"]) -> str:
    """Printed examples in one language, drawn at random and never chosen.

    The pool is the greedy response of each model's main condition to the `full` version of the dissociation
    items. Two parts: `per_category` items of each category with the response of every rung to the same prompt,
    and `per_rung` responses of each rung. The first draw does not depend on the models, so two ladders show
    the same prompts. With `coded`, each response carries the codes the sheet gives it."""
    df = _load_rows(resp_root)
    if df.empty:
        return ""
    df = df[(df["mode"] == "greedy") & (df["version"] == "full") & ~df["control"].astype(bool) & df["model"].isin(models)]
    df = _one_language(select_main(df, expected), language)
    if df.empty:
        return ""
    codes = {} if coded is None else {(r.model, r.item_id, r.version): (r.intent, r.correctness)
                                      for r in coded.itertuples()}

    def prompt(row: pd.Series) -> str:
        shown = row["stimulus_text"] if pd.notna(row.get("stimulus_text")) else row["prompt_text"]
        return f"**Prompt**\n\n{_quoted(shown)}"

    def response(row: pd.Series, head: str) -> str:
        code = codes.get((row["model"], row["item_id"], row["version"]))
        text = row["response"].strip()
        return (f"**{head}**" + (f" (coded `{code[0]}` · `{code[1]}`)" if code else "") + "\n\n"
                + (_quoted(text) if text else EMPTY_RESPONSE + "\n"))

    blocks = [f"Greedy responses to the `full` version of the dissociation items, each model in its main condition "
              f"(spec §6). Items are drawn at random with seed {seed}; none was chosen by hand.\n",
              "## The same prompt at every rung\n"]
    for category, items in df.drop_duplicates("item_id").groupby("category")["item_id"]:
        for item_id in sorted(items, key=lambda i: _draw_rank(seed, "prompt", i))[:per_category]:
            rows = df[df["item_id"] == item_id]
            blocks += [f"### {item_id} · {category}\n", prompt(rows.iloc[0])]
            blocks += [response(r, r["model"]) for m in models for _, r in rows[rows["model"] == m].iterrows()]
    blocks.append("## Responses drawn for each rung\n")
    for model in models:
        rows = df[df["model"] == model]
        if rows.empty:
            continue
        blocks.append(f"### {model}\n")
        drawn = sorted(rows.index, key=lambda i: _draw_rank(seed, model, rows.at[i, "item_id"]))[:per_rung]
        for _, r in rows.loc[drawn].iterrows():
            blocks += [f"#### {r['item_id']} · {r['category']}\n", prompt(r), response(r, "Response")]
    return "\n".join(blocks)


DODGE = 0.07       # distance on the x axis between two categories at one model, in units of one model


def draw_ladder(intent: pd.DataFrame | None, logodds: pd.DataFrame | None, models: list[str]):
    """The two measures along the ladder, `full` version: rate of intended responses (Line 1) and mean log-odds (Line 2).

    Each Line 2 mean carries the interval of its table (`ci_low`, `ci_high`); the categories are set slightly
    apart at each model so that the intervals do not hide one another. Line 1 rates carry no interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.arange(len(models))
    if intent is not None:
        full_i = intent.xs("full", level="condition")
        for cat in full_i.index:
            axes[0].plot(x, [full_i.loc[cat, (m, "intended")] for m in models], marker="o", label=cat)
        axes[0].legend(fontsize=7)
    if logodds is not None:
        full_l = logodds.xs("full", level="condition")
        axes[1].axhline(0.0, color="0.6", linewidth=0.8, zorder=0)      # no preference between the two readings
        for k, cat in enumerate(full_l.index):
            mean, low, high = (np.array([full_l.loc[cat, (m, col)] for m in models], dtype=float)
                               for col in ("mean", "ci_low", "ci_high"))
            axes[1].errorbar(x + (k - (len(full_l.index) - 1) / 2) * DODGE, mean, yerr=[mean - low, high - mean],
                             marker="o", markersize=4, linewidth=1.2, elinewidth=1, capsize=2, label=cat)
    axes[0].set_ylabel("rate of intended responses"); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("log-odds intended vs literal\n(per token)")     # two lines: one is cut on a short panel
    axes[1].set_title("mean over items, with its 95% bootstrap interval", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, models)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def ladder_figure(intent: pd.DataFrame | None, logodds: pd.DataFrame | None, models: list[str], out: Path) -> None:
    import matplotlib.pyplot as plt
    fig = draw_ladder(intent, logodds, models)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build(coded_path: Path, key_path: Path | None, coder2_path: Path | None, resp_root: Path,
          lp_root: Path, models: list[str], out_dir: Path, expected: dict[str, bool]) -> dict:
    """Write every table the data supports, one set per language; skip (with a warning) what is not there yet.

    The main language's tables carry no suffix and feed the figure; the others are written as `<name>_<language>`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []

    def write(table: pd.DataFrame, name: str, language: str, tex: bool = True, na_rep: str = "NaN") -> None:
        table.to_csv(out_dir / f"{name}{_suffix(language)}.csv")
        if tex:
            (out_dir / f"{name}{_suffix(language)}.tex").write_text(
                table.to_latex(float_format="%.2f", na_rep=na_rep))

    coded = intent = None
    try:
        coded = load_sheet(coded_path)
        mine = coded[coded["model"].isin(models)]
        for language in _languages(mine):
            table = intent_table(coded, models, expected, language)
            write(table, "intent", language)
            intent = table if intent is None else intent        # the first language is the main one when present
            try:
                write(correctness_table(coded, models, expected, language), "correctness", language, na_rep="--")
            except ValueError as e:
                warns.append(f"correctness skipped ({language}): {e}")
            if len(_one_language(select_template_control(mine, expected), language)):   # coded only if asked for
                write(template_control_table(coded, models, expected, language), "template_control", language, tex=False)
    except (FileNotFoundError, ValueError) as e:
        warns.append(f"Line 1 skipped (coding sheet not usable): {e}")
    lo = None
    try:
        lp = _load_rows(lp_root)
        if lp.empty:
            raise ValueError(f"no log-prob rows under {lp_root}")
        for language in _languages(lp[lp["model"].isin(models)]):
            table = logodds_table(lp_root, models, expected, language)
            write(table, "logodds", language)
            lo = table if lo is None else lo
            try:
                write(logodds_template_control_table(lp_root, models, expected, language), "logodds_template_control", language)
            except ValueError:
                pass                        # a ladder without raw passes (self-run) has no template control
    except (FileNotFoundError, ValueError, KeyError) as e:
        warns.append(f"Line 2 skipped (no log-prob rows): {e}")
    if intent is not None or lo is not None:
        ladder_figure(intent, lo, models, out_dir / "ladder.png")
    resp = _load_rows(resp_root)
    shown = _languages(resp[resp["model"].isin(models)]) if not resp.empty else []
    for language in shown or [MAIN_LANGUAGE]:               # without responses the main file is written empty
        (out_dir / f"examples{_suffix(language)}.md").write_text(
            examples(resp_root, models, expected, language, coded) if shown else "")
    kappa, agree = None, None
    if coded is not None and coder2_path is not None:
        try:
            table = agreement(coded, load_sheet(coder2_path, key=key_path))
            if not table.loc["intent", "n"]:
                raise ValueError("no row of coder 2 is in coder 1's sheet")
            agree = {name: {"n": int(r["n"]), "same": int(r["same"]), "kappa": float(r["kappa"])}
                     for name, r in table.iterrows()}
            kappa = agree["intent"]["kappa"]
            (out_dir / "kappa.txt").write_text("".join(
                f"cohen_kappa_{name}={v['kappa']:.3f} n={v['n']} same={v['same']}\n" for name, v in agree.items()))
        except (FileNotFoundError, ValueError) as e:
            warns.append(f"kappa skipped (coder 2 sheet not usable): {e}")
    for w in warns:
        warnings.warn(w)
    return {"n_coded": int(len(coded)) if coded is not None else 0, "kappa": kappa, "agreement": agree,
            "models": models, "warnings": warns}
