"""The generated blocks of `paper/RESULTS.md` (design of 2026-11-21).

The page is written by hand. Between a pair of marker lines its text is generated here, from the
tables `gricean report` committed under `paper/<ladder>/`; nothing is recomputed from `data/`, so
the page cannot disagree with the tables the paper cites. The opening marker names the table and
the cells it shows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .report import MAIN_LANGUAGE, _suffix
from .stimuli import QUOTAS

TABLES = ("intent", "logodds", "kappa", "termination", "summary")
CELL_TABLES = ("intent", "logodds")             # read from a CSV of category x condition by model
LISTS = ("condition", "category", "models")     # keys whose value is a comma-separated list
KEYS = ("table", "ladder", "language", *LISTS)

MODEL_LABELS = {
    "olmo_stage1": "Stage 1", "olmo_base": "Base", "olmo_sft": "SFT", "olmo_dpo": "DPO", "olmo_instruct": "Instruct",
    "olmo_stage1_alpaca": "Stage 1 Alpaca", "olmo_n500": "500", "olmo_n2000": "2,000", "olmo_full": "All pairs",
    "olmo_format_only": "Format only",
}
LANGUAGE_NAMES = {"en": "English", "pt": "Portuguese"}
CAPTIONS = {
    "intent": "Line 1. {language}, {conditions}: responses coded `intended`, of the items in each cell (`{file}`).",
    "logodds": ("Line 2. {language}, {conditions}: mean log-odds of the intended over the literal continuation, "
                "with its bootstrap interval (`{file}`)."),
}
NO_VALUE = "—"
MINUS = "−"


def parse_spec(text: str) -> dict:
    """What an opening marker asks for: `table=intent ladder=olmo condition=full,stripped` as a dict."""
    spec: dict = {}
    for word in text.split():
        key, eq, value = word.partition("=")
        if not eq or not value:
            raise ValueError(f"{word!r} is not of the form key=value")
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r} (allowed: {', '.join(KEYS)})")
        spec[key] = value.split(",") if key in LISTS else value
    table = spec.get("table")
    if table not in TABLES:
        raise ValueError(f"table must be one of {', '.join(TABLES)}; found {table!r}")
    if table != "summary" and "ladder" not in spec:
        raise ValueError(f"table={table} needs ladder=")
    if table in CELL_TABLES and "condition" not in spec:
        raise ValueError(f"table={table} needs condition=")
    spec.setdefault("language", MAIN_LANGUAGE)
    if spec["language"] not in QUOTAS:
        raise ValueError(f"language must be one of {', '.join(QUOTAS)}; found {spec['language']!r}")
    return spec


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """A table padded to equal column widths, so that the source of the page reads as a table too."""
    widths = [max(3, *(len(r[i]) for r in [header, *rows])) for i in range(len(header))]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"   # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def _label(category: str) -> str:
    return category.replace("_", " ").capitalize()


def _number(x: float) -> str:
    return f"{round(float(x), 2) + 0.0:.2f}".replace("-", MINUS)      # `+ 0.0`: no "-0.00"


def _count(rate: float, n: int, where: str) -> int:
    k = rate * n
    if abs(k - round(k)) > 1e-6:                # a rate that is not k/n: the quota is not this table's; do not round
        raise ValueError(f"{where}: rate {rate} is not a count out of {n}")
    return round(k)


def _cell_table(spec: dict, paper: Path) -> str:
    name = f"{spec['table']}{_suffix(spec['language'])}.csv"
    shown = f"paper/{spec['ladder']}/{name}"
    path = Path(paper) / spec["ladder"] / name
    if not path.exists():
        raise ValueError(f"{shown} is not there")
    df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    in_file = list(dict.fromkeys(df.columns.get_level_values(0)))
    models = spec.get("models", in_file)
    if missing := [m for m in models if m not in in_file]:
        raise ValueError(f"{shown} has no model {', '.join(missing)}")
    categories = list(dict.fromkeys(df.index.get_level_values(0)))
    if "category" in spec:
        if missing := [c for c in spec["category"] if c not in categories]:
            raise ValueError(f"{shown} has no category {', '.join(missing)}")
        categories = [c for c in categories if c in spec["category"]]
    conditions = spec["condition"]
    several = len(conditions) > 1
    quota, control_quota = QUOTAS[spec["language"]]
    rows, totals = [], {c: [] for c in conditions}          # totals: per condition, one list of counts per row
    for category in categories:
        for condition in conditions:
            if (category, condition) not in df.index:
                if "category" in spec:
                    raise ValueError(f"{shown} has no row {category}, {condition}")
                continue                                    # not every category exists in every version
            label = f"{_label(category)}, `{condition}`" if several else _label(category)
            cells, counts = [], []
            for model in models:
                if spec["table"] == "intent":
                    n = control_quota if condition == "control" else quota
                    rate = df.loc[(category, condition), (model, "intended")]
                    k = None if pd.isna(rate) else _count(rate, n, f"{shown}, {category}, {condition}, {model}")
                    counts.append((k, n))
                    cells.append(NO_VALUE if k is None else f"{k}/{n}")
                else:
                    mean, low, high = (df.loc[(category, condition), (model, c)] for c in ("mean", "ci_low", "ci_high"))
                    cells.append(NO_VALUE if pd.isna(mean) else f"{_number(mean)} [{_number(low)}, {_number(high)}]")
            rows.append([label, *cells])
            totals[condition].append(counts)
    if empty := [c for c in conditions if not totals[c]]:
        raise ValueError(f"{shown} has no condition {', '.join(empty)}")
    if spec["table"] == "intent":                           # the one sum on the page; Line 2 means are never pooled
        for condition in conditions:
            per_row = totals[condition]
            if len(per_row) < 2:
                continue
            label = f"All {len(per_row)} categories" + (f", `{condition}`" if several else "")
            cells = []
            for column in zip(*per_row):
                known = all(k is not None for k, _ in column)
                cells.append(f"{sum(k for k, _ in column)}/{sum(n for _, n in column)}" if known else NO_VALUE)
            rows.append([label, *cells])
    caption = CAPTIONS[spec["table"]].format(language=LANGUAGE_NAMES[spec["language"]], file=shown,
                                             conditions=", ".join(f"`{c}`" for c in conditions))
    return caption + "\n\n" + _markdown_table(["Category", *(MODEL_LABELS.get(m, m) for m in models)], rows)


def render_block(spec: dict, paper: Path) -> str:
    """The Markdown of one generated block, without its markers and without a final line break."""
    if spec["table"] in CELL_TABLES:
        return _cell_table(spec, paper)
    raise ValueError(f"table={spec['table']} cannot be rendered here")
