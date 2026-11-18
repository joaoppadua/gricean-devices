from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd

from .io import read_jsonl

INTENT = ("literal", "intended", "neither")
CORRECT = ("correct", "incorrect", "na")
PUBLIC = ["row_id", "item_id", "category", "language", "control", "version", "mode", "seed",
          "stimulus_text", "response"]
HIDDEN = ["model", "template", "prompt_text"]   # prompt_text reveals the chat template, hence the stage
EMPTY_CODES = {"intent": "neither", "correctness": "na"}   # an empty response is coded by rule, not by hand
READINGS = ["literal_reading", "intended_reading", "literal_continuation", "intended_continuation"]


def select_main(df: pd.DataFrame, expected: dict[str, bool]) -> pd.DataFrame:
    """Rows of each model's main condition: templated for tuned models, raw for base models.

    The embedded version is raw for every model (spec 4.3) and therefore always belongs here."""
    exp = df["model"].map(expected)
    keep = ((df["version"] == "embedded") & ~df["template"]) | (df["template"] == exp)
    return df[keep & exp.notna()]


def select_coded(df: pd.DataFrame, expected: dict[str, bool]) -> pd.DataFrame:
    """Rows Line 1 codes (spec 5.1): the greedy response of each model's main condition.

    Sampled responses and the raw template control of tuned models stay in `data/responses`
    uncoded; the template control is reported from Line 2."""
    return select_main(df[df["mode"] == "greedy"], expected)


def _all_responses(response_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(response_root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    df["row_id"] = [f"{r.model}|{int(r.template)}|{r.item_id}|{r.version}|{r.mode}|{r.seed}"
                    for r in df.itertuples()]
    return df


def _blind_id(row_id: str, seed: int) -> str:
    """An id that says nothing about the row: the full sheet's `row_id` begins with the model."""
    return hashlib.sha256(f"{seed}|{row_id}".encode()).hexdigest()[:12]


def export_sheet(response_root: Path, out: Path, *, coder: str, sample_frac: float | None,
                 seed: int, expected: dict[str, bool], readings: dict[str, dict] | None = None) -> int:
    """Write the coding sheet for `select_coded` rows; `expected` maps model -> templated main condition.

    Empty responses arrive coded (`EMPTY_CODES`). The blind sample (`sample_frac`) is drawn from the
    rows that are coded by hand, so agreement is not inflated by rows no one judged. Its rows carry an
    opaque `row_id` and are ordered by item and version, then by that id, so neither the id nor the
    position of a row gives its model away; the key file leads back to the full sheet's `row_id`
    (`source_row_id`). `readings` (item id -> the four `READINGS` fields) puts the two readings of
    each item beside its rows."""
    df = select_coded(_all_responses(response_root), expected)
    empty = df["response"].str.strip() == ""
    if sample_frac is not None:
        df = df[~empty].sample(frac=sample_frac, random_state=seed).rename(columns={"row_id": "source_row_id"})
        df["row_id"] = [_blind_id(r, seed) for r in df["source_row_id"]]
        if not df["row_id"].is_unique:
            raise ValueError("blind ids collide; choose another seed")
        df = df.sort_values(["item_id", "version", "row_id"])
        df[["row_id", "source_row_id", *HIDDEN]].to_csv(Path(out).with_suffix(".key.csv"), index=False)
        cols = PUBLIC
        if readings is not None:
            missing = sorted(set(df["item_id"]) - set(readings))
            if missing:
                raise ValueError(f"no readings for item(s): {missing[:5]}")
            for field in READINGS:
                df[field] = [readings[i][field] or "" for i in df["item_id"]]
            cols = PUBLIC + READINGS
    else:
        cols = PUBLIC + HIDDEN
    sheet = df[cols].copy()
    sheet["coder"] = coder
    by_rule = sheet["response"].str.strip() == ""
    for col, code in EMPTY_CODES.items():
        sheet[col] = pd.Series(pd.NA, index=sheet.index, dtype=object).mask(by_rule, code)
    out = Path(out)
    if out.exists():                     # re-export keeps codes already entered; new rows arrive blank
        old = pd.read_csv(out, dtype=str, keep_default_na=False).set_index("row_id")
        for col in ("intent", "correctness"):
            if col in old.columns:
                carried = sheet["row_id"].map(old[col]).replace("", pd.NA)
                sheet[col] = carried.where(carried.notna(), sheet[col])
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)
    return len(sheet)


def _check_ids(ids: pd.Series, expected: pd.Series, *, held: str = "the sheet", by: str = "the key") -> None:
    """`held` has every row of `by` once, under the id `by` gives it.

    A merge on `row_id` would drop any other row without a word. A spreadsheet makes such rows: it
    rewrites an id that looks like a number (`0374…` loses its zero, twelve digits turn into
    `4.10363E+11`). A coder who returns the codes in a file of their own can skip or repeat a row."""
    repeated = sorted(set(ids[ids.duplicated()]))
    if repeated:
        raise ValueError(f"row_id(s) more than once in {held}: {repeated[:5]}")
    unknown, missing = sorted(set(ids) - set(expected)), sorted(set(expected) - set(ids))
    if unknown:
        raise ValueError(f"{len(unknown)} row_id(s) not in {by}: {unknown[:5]}; "
                         f"ids of {by} without a row: {missing[:5]}")
    if missing:
        raise ValueError(f"{len(missing)} row_id(s) of {by} missing from {held}: {missing[:5]}")


def load_sheet(path: Path, key: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col, allowed in (("intent", INTENT), ("correctness", CORRECT)):
        blank = df[df[col].str.strip() == ""]
        if len(blank):
            raise ValueError(f"blank {col} in row_id(s): {blank['row_id'].tolist()[:5]}")
        bad = df[~df[col].str.strip().str.lower().isin(allowed)]
        if len(bad):
            raise ValueError(f"invalid {col} {bad[col].tolist()[:5]} at row_id(s) {bad['row_id'].tolist()[:5]}")
        df[col] = df[col].str.strip().str.lower()
    if key is not None:
        k = pd.read_csv(key, dtype=str)
        _check_ids(df["row_id"], k["row_id"])
        df = df.merge(k, on="row_id", how="left")
        if "source_row_id" in df.columns:        # a blind sheet: back to the ids of the full sheet
            df["row_id"] = df.pop("source_row_id")
    df["control"] = df["control"].str.lower() == "true"
    if "template" in df.columns:
        df["template"] = df["template"].astype(str).str.lower() == "true"
    return df


BATCH_FIELDS = ["row_id", "item_id", "category", "language", "control", "version", *READINGS]
FORM = ["row_id", "intent", "correctness"]


def _row_text(row: pd.Series, k: int, total: int) -> str:
    fields = [f"{f}: {row[f]}" for f in BATCH_FIELDS if f in row.index]
    return "\n".join([f'<row n="{k}" of="{total}">', *fields,
                      "<stimulus_text>", row["stimulus_text"], "</stimulus_text>",
                      "<response>", row["response"], "</response>", "</row>", ""])


def write_batches(sheet: Path, out_dir: Path, n: int) -> list[Path]:
    """Write a blind sheet as at most `n` text files, each with a blank form for its codes.

    For a coder who reads text and not a spreadsheet. The batches follow the order of the sheet and
    never split an item, because the codebook asks for the same codes on the same text within an
    item. A batch (`batch_NN.md`) shows the columns of the sheet without the bookkeeping ones
    (`mode`, `seed`, `coder`); its form (`codes_NN.csv`) lists the same rows with both codes blank.
    A folder that already holds forms is refused: they may carry codes."""
    out_dir = Path(out_dir)
    if any(out_dir.glob("codes_*.csv")):
        raise ValueError(f"{out_dir} already holds forms; choose an empty folder")
    df = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    sizes = df.groupby("item_id", sort=False).size()
    batch_of = ((sizes.cumsum() - sizes) * n // len(df)).astype(int)        # by the rows that come before the item
    number = pd.factorize(df["item_id"].map(batch_of))[0] + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for b in range(1, number.max() + 1):
        rows = df[number == b]
        head = f"# Batch {b:02d} of {number.max():02d}: {len(rows)} rows\n\n"
        body = "\n".join(_row_text(r, k, len(rows)) for k, (_, r) in enumerate(rows.iterrows(), 1))
        paths.append(out_dir / f"batch_{b:02d}.md")
        paths[-1].write_text(head + body)
        rows[["row_id"]].assign(intent="", correctness="").to_csv(out_dir / f"codes_{b:02d}.csv", index=False)
    return paths


def enter_codes(sheet: Path, forms: Path) -> int:
    """Copy the codes of the filled forms (`codes_*.csv` under `forms`) into the sheet, as entered.

    Only `intent` and `correctness` change. Nothing is written unless the forms hold every row of the
    sheet once and every code is one `load_sheet` accepts (case and outer spaces aside)."""
    df = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    paths = sorted(Path(forms).glob("codes_*.csv"))
    if not paths:
        raise ValueError(f"no codes_*.csv under {forms}")
    codes = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths], ignore_index=True)
    lacking = [c for c in FORM if c not in codes.columns]
    if lacking:
        raise ValueError(f"forms lack column(s) {lacking}")
    codes["row_id"] = codes["row_id"].str.strip()
    _check_ids(codes["row_id"], df["row_id"], held="the forms", by="the sheet")
    for col, allowed in (("intent", INTENT), ("correctness", CORRECT)):
        codes[col] = codes[col].str.strip().str.lower()
        bad = codes[~codes[col].isin(allowed)]
        if len(bad):
            raise ValueError(f"invalid {col} {bad[col].tolist()[:5]} at row_id(s) {bad['row_id'].tolist()[:5]}")
    entered = codes.set_index("row_id")
    for col in ("intent", "correctness"):
        df[col] = df["row_id"].map(entered[col])
    df.to_csv(sheet, index=False)
    return len(df)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Agreement of two coders on the rows both coded (matched by `row_id`): `n`, `same`, `kappa` per measure.

    `intent` is compared on all those rows. `correctness` is compared twice: on the rows where the
    two coders gave the same intent, which is the main measure, because `correct` means one thing
    under `neither` and another under the other two codes; and on all rows, so that the first number
    can be seen not to depend on what it leaves out. A measure without rows has no kappa."""
    m = a.merge(b[["row_id", "intent", "correctness"]], on="row_id", suffixes=("", "_b"))
    same_intent = m[m["intent"] == m["intent_b"]]
    out = {}
    for name, rows, col in (("intent", m, "intent"), ("correctness_same_intent", same_intent, "correctness"),
                            ("correctness_all", m, "correctness")):
        x, y = rows[col].tolist(), rows[f"{col}_b"].tolist()
        out[name] = {"n": len(x), "same": sum(p == q for p, q in zip(x, y)),
                     "kappa": cohen_kappa(x, y) if x else float("nan")}
    return pd.DataFrame.from_dict(out, orient="index")
