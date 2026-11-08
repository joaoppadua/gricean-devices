from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .coding import INTENT, cohen_kappa, load_sheet
from .io import read_jsonl

INDEX = ["category", "condition"]


def select_main(df: pd.DataFrame, expected: dict[str, bool]) -> pd.DataFrame:
    """Rows of each model's main condition: templated for tuned models, raw for base models.

    The embedded version is raw for every model (spec 4.3) and therefore always belongs here."""
    exp = df["model"].map(expected)
    keep = ((df["version"] == "embedded") & ~df["template"]) | (df["template"] == exp)
    return df[keep & exp.notna()]


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
    return table.reindex(columns=cols)


def intent_table(coded: pd.DataFrame, models: list[str], expected: dict[str, bool]) -> pd.DataFrame:
    """Rate of literal / intended / neither responses by (category, condition) and model.

    Conditions are the stimulus versions plus 'control' for control items."""
    df = select_main(coded[coded["model"].isin(models)], expected)
    return _rates(df, models)


def template_control_table(coded: pd.DataFrame, models: list[str], expected: dict[str, bool]) -> pd.DataFrame:
    df = select_template_control(coded[coded["model"].isin(models)], expected)
    return _rates(df, models)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_logprobs(lp_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(lp_root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)


def logodds_table(lp_root: Path, models: list[str], expected: dict[str, bool]) -> pd.DataFrame:
    df = _load_logprobs(lp_root)
    if df.empty:
        raise ValueError(f"no log-prob rows under {lp_root}")
    df = select_main(df[df["model"].isin(models)], expected).assign(condition=lambda d: d["version"])
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


def examples(resp_root: Path, models: list[str], item_ids: list[str], expected: dict[str, bool]) -> str:
    rows = []
    for path in sorted(Path(resp_root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    if df.empty:
        return ""
    df = select_main(df[(df["mode"] == "greedy") & (df["version"] == "full") & df["model"].isin(models)], expected)
    blocks = []
    for item_id in item_ids:
        sub = df[df["item_id"] == item_id]
        if sub.empty:
            continue
        first = sub.iloc[0]
        prompt = first["stimulus_text"] if "stimulus_text" in sub.columns and pd.notna(first["stimulus_text"]) else first["prompt_text"]
        blocks.append(f"### {item_id}\n\n**Prompt:** {prompt}\n")
        for model in models:
            r = sub[sub["model"] == model]
            if not r.empty:
                blocks.append(f"**{model}:** {r.iloc[0]['response'].strip()}\n")
    return "\n".join(blocks)


def ladder_figure(intent: pd.DataFrame | None, logodds: pd.DataFrame | None, models: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    if intent is not None:
        full_i = intent.xs("full", level="condition")
        for cat in full_i.index:
            axes[0].plot(models, [full_i.loc[cat, (m, "intended")] for m in models], marker="o", label=cat)
        axes[0].legend(fontsize=7)
    if logodds is not None:
        full_l = logodds.xs("full", level="condition")
        for cat in full_l.index:
            axes[1].plot(models, [full_l.loc[cat, (m, "mean")] for m in models], marker="o", label=cat)
    axes[0].set_ylabel("rate of intended responses"); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("log-odds intended vs literal (per token)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build(coded_path: Path, key_path: Path | None, coder2_path: Path | None, resp_root: Path,
          lp_root: Path, models: list[str], out_dir: Path, expected: dict[str, bool],
          example_ids: list[str] | None = None) -> dict:
    """Write every table the data supports; skip (with a warning) what is not there yet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []
    coded = intent = None
    try:
        coded = load_sheet(coded_path)
        intent = intent_table(coded, models, expected)
        intent.to_csv(out_dir / "intent.csv"); (out_dir / "intent.tex").write_text(intent.to_latex(float_format="%.2f"))
        template_control_table(coded, models, expected).to_csv(out_dir / "template_control.csv")
    except (FileNotFoundError, ValueError) as e:
        warns.append(f"Line 1 skipped (coding sheet not usable): {e}")
    lo = None
    try:
        lo = logodds_table(lp_root, models, expected)
        lo.to_csv(out_dir / "logodds.csv"); (out_dir / "logodds.tex").write_text(lo.to_latex(float_format="%.2f"))
    except (FileNotFoundError, ValueError, KeyError) as e:
        warns.append(f"Line 2 skipped (no log-prob rows): {e}")
    if intent is not None or lo is not None:
        ladder_figure(intent, lo, models, out_dir / "ladder.png")
    if example_ids is None:
        example_ids = sorted(coded["item_id"].unique())[:6] if coded is not None else []
        if not example_ids:
            rows = [r for p in sorted(Path(resp_root).glob("*.jsonl")) for r in read_jsonl(p)]
            example_ids = sorted({r["item_id"] for r in rows})[:6]
    (out_dir / "examples.md").write_text(examples(resp_root, models, example_ids, expected))
    kappa = None
    if coded is not None and coder2_path is not None:
        try:
            c2 = load_sheet(coder2_path, key=key_path)
            merged = coded.merge(c2[["row_id", "intent"]], on="row_id", suffixes=("", "_c2"))
            kappa = cohen_kappa(merged["intent"].tolist(), merged["intent_c2"].tolist())
            (out_dir / "kappa.txt").write_text(f"cohen_kappa_intent={kappa:.3f} n={len(merged)}\n")
        except (FileNotFoundError, ValueError) as e:
            warns.append(f"kappa skipped (coder 2 sheet not usable): {e}")
    for w in warns:
        warnings.warn(w)
    return {"n_coded": int(len(coded)) if coded is not None else 0, "kappa": kappa, "models": models,
            "warnings": warns}
