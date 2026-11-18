from __future__ import annotations

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


def logodds_table(lp_root: Path, models: list[str], expected: dict[str, bool],
                  language: str | None = None) -> pd.DataFrame:
    df = _load_logprobs(lp_root)
    if df.empty:
        raise ValueError(f"no log-prob rows under {lp_root}")
    return _logodds(_one_language(select_main(df[df["model"].isin(models)], expected), language), models)


def logodds_template_control_table(lp_root: Path, models: list[str], expected: dict[str, bool],
                                   language: str | None = None) -> pd.DataFrame:
    """Line 2 on the raw passes of tuned models: the template control, scored without a judge.

    Columns are the models that have a raw control, in ladder order."""
    df = _load_logprobs(lp_root)
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
                blocks.append(f"**{model}:** {r.iloc[0]['response'].strip() or EMPTY_RESPONSE}\n")
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
    """Write every table the data supports, one set per language; skip (with a warning) what is not there yet.

    The main language's tables carry no suffix and feed the figure; the others are written as `<name>_<language>`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []

    def write(table: pd.DataFrame, name: str, language: str, tex: bool = True) -> None:
        table.to_csv(out_dir / f"{name}{_suffix(language)}.csv")
        if tex:
            (out_dir / f"{name}{_suffix(language)}.tex").write_text(table.to_latex(float_format="%.2f"))

    coded = intent = None
    try:
        coded = load_sheet(coded_path)
        mine = coded[coded["model"].isin(models)]
        for language in _languages(mine):
            table = intent_table(coded, models, expected, language)
            write(table, "intent", language)
            intent = table if intent is None else intent        # the first language is the main one when present
            if len(_one_language(select_template_control(mine, expected), language)):   # coded only if asked for
                write(template_control_table(coded, models, expected, language), "template_control", language, tex=False)
    except (FileNotFoundError, ValueError) as e:
        warns.append(f"Line 1 skipped (coding sheet not usable): {e}")
    lo = None
    try:
        lp = _load_logprobs(lp_root)
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
    if example_ids is None:
        example_ids = sorted(coded["item_id"].unique())[:6] if coded is not None else []
        if not example_ids:
            rows = [r for p in sorted(Path(resp_root).glob("*.jsonl")) for r in read_jsonl(p)]
            example_ids = sorted({r["item_id"] for r in rows})[:6]
    (out_dir / "examples.md").write_text(examples(resp_root, models, example_ids, expected))
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
