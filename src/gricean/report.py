from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .coding import INTENT, cohen_kappa, load_sheet
from .io import read_jsonl

INDEX = ["category", "version"]


def intent_table(coded: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    df = coded[coded["model"].isin(models) & ~coded["control"]]
    counts = df.groupby(INDEX + ["model", "intent"]).size()
    totals = df.groupby(INDEX + ["model"]).size()
    rates = (counts / totals).rename("rate").reset_index()
    table = rates.pivot_table(index=INDEX, columns=["model", "intent"], values="rate", fill_value=0.0)
    cols = pd.MultiIndex.from_product([models, list(INTENT)])
    return table.reindex(columns=cols, fill_value=0.0)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def logodds_table(lp_root: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(lp_root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    df = df[df["model"].isin(models)]
    out = {}
    for (cat, ver, model), g in df.groupby(INDEX + ["model"]):
        x = g["log_odds"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(x)
        out[(cat, ver)] = out.get((cat, ver), {})
        out[(cat, ver)].update({(model, "mean"): float(x.mean()), (model, "ci_low"): lo,
                                (model, "ci_high"): hi, (model, "share_neither"): float(g["share_neither"].mean())})
    table = pd.DataFrame.from_dict(out, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=INDEX)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    cols = pd.MultiIndex.from_product([models, ["mean", "ci_low", "ci_high", "share_neither"]])
    return table.reindex(columns=cols)


def examples(resp_root: Path, models: list[str], item_ids: list[str]) -> str:
    rows = []
    for path in sorted(Path(resp_root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    df = df[(df["mode"] == "greedy") & (df["version"] == "full") & df["model"].isin(models)]
    blocks = []
    for item_id in item_ids:
        sub = df[df["item_id"] == item_id]
        if sub.empty:
            continue
        prompt = sub.iloc[0]["prompt_text"]
        blocks.append(f"### {item_id}\n\n**Prompt:** {prompt}\n")
        for model in models:
            r = sub[sub["model"] == model]
            if not r.empty:
                blocks.append(f"**{model}:** {r.iloc[0]['response'].strip()}\n")
    return "\n".join(blocks)


def ladder_figure(intent: pd.DataFrame, logodds: pd.DataFrame, models: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    full_i = intent.xs("full", level="version")
    full_l = logodds.xs("full", level="version")
    for cat in full_i.index:
        axes[0].plot(models, [full_i.loc[cat, (m, "intended")] for m in models], marker="o", label=cat)
    for cat in full_l.index:
        axes[1].plot(models, [full_l.loc[cat, (m, "mean")] for m in models], marker="o", label=cat)
    axes[0].set_ylabel("rate of intended responses"); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("log-odds intended vs literal (per token)")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build(coded_path: Path, key_path: Path | None, coder2_path: Path | None, resp_root: Path,
          lp_root: Path, models: list[str], out_dir: Path, example_ids: list[str] | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coded = load_sheet(coded_path)
    intent = intent_table(coded, models)
    lo = logodds_table(lp_root, models)
    intent.to_csv(out_dir / "intent.csv"); (out_dir / "intent.tex").write_text(intent.to_latex(float_format="%.2f"))
    lo.to_csv(out_dir / "logodds.csv"); (out_dir / "logodds.tex").write_text(lo.to_latex(float_format="%.2f"))
    ladder_figure(intent, lo, models, out_dir / "ladder.png")
    ids = example_ids or sorted(coded["item_id"].unique())[:6]
    (out_dir / "examples.md").write_text(examples(resp_root, models, ids))
    kappa = None
    if coder2_path is not None:
        c2 = load_sheet(coder2_path, key=key_path)
        merged = coded.merge(c2[["row_id", "intent"]], on="row_id", suffixes=("", "_c2"))
        kappa = cohen_kappa(merged["intent"].tolist(), merged["intent_c2"].tolist())
        (out_dir / "kappa.txt").write_text(f"cohen_kappa_intent={kappa:.3f} n={len(merged)}\n")
    return {"n_coded": int(len(coded)), "kappa": kappa, "models": models}
