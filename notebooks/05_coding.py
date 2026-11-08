import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from gricean.lm import ladders, expected_templates
    from gricean.coding import load_sheet, cohen_kappa
    from gricean.report import build
    return Path, build, cohen_kappa, expected_templates, ladders, load_sheet, mo


@app.cell
def _(Path, cohen_kappa, load_sheet, mo):
    c1, _c2, key = Path("data/coding/coder1.csv"), Path("data/coding/coder2.csv"), Path("data/coding/coder2.key.csv")
    if c1.exists() and _c2.exists():
        a = load_sheet(c1); b = load_sheet(_c2, key=key)
        m = a.merge(b[["row_id", "intent"]], on="row_id", suffixes=("", "_c2"))
        out = mo.md(f"Cohen's kappa (intent), n={len(m)}: **{cohen_kappa(m['intent'].tolist(), m['intent_c2'].tolist()):.3f}**")
    else:
        out = mo.md("Both coder sheets are needed for kappa.")
    out
    return


@app.cell
def _(ladders, mo):
    ladder = mo.ui.dropdown(list(ladders()), value="olmo", label="ladder")
    go = mo.ui.run_button(label="build paper/ tables")
    mo.hstack([ladder, go])
    return go, ladder


@app.cell
def _(Path, build, expected_templates, go, ladder, ladders, mo):
    mo.stop(not go.value)
    _c2 = Path("data/coding/coder2.csv")
    summary = build(Path("data/coding/coder1.csv"), Path("data/coding/coder2.key.csv") if _c2.exists() else None,
                    _c2 if _c2.exists() else None, Path("data/responses"), Path("data/logprobs"),
                    ladders()[ladder.value], Path("paper") / ladder.value, expected_templates())
    mo.md(f"`{summary}`")
    return


if __name__ == "__main__":
    app.run()
