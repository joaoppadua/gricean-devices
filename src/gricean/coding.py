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
        if "source_row_id" in df.columns:        # a blind sheet: back to the ids of the full sheet
            df["row_id"] = df.pop("source_row_id")
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
