from __future__ import annotations

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


def export_sheet(response_root: Path, out: Path, *, coder: str, sample_frac: float | None,
                 seed: int, expected: dict[str, bool]) -> int:
    """Write the coding sheet for `select_coded` rows; `expected` maps model -> templated main condition.

    Empty responses arrive coded (`EMPTY_CODES`). The blind sample (`sample_frac`) is drawn from the
    rows that are coded by hand, so agreement is not inflated by rows no one judged."""
    df = select_coded(_all_responses(response_root), expected)
    empty = df["response"].str.strip() == ""
    if sample_frac is not None:
        df = df[~empty].sample(frac=sample_frac, random_state=seed).sort_values("row_id")
        df[["row_id", *HIDDEN]].to_csv(Path(out).with_suffix(".key.csv"), index=False)
        cols = PUBLIC
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
        df = df.merge(k, on="row_id", how="left")
    df["control"] = df["control"].str.lower() == "true"
    if "template" in df.columns:
        df["template"] = df["template"].astype(str).str.lower() == "true"
    return df


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
