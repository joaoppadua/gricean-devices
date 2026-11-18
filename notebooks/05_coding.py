import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from gricean.paths import enter_repo_root
    enter_repo_root(mo.notebook_dir())   # package paths are relative to the repo root
    from pathlib import Path
    from gricean.lm import ladders, expected_templates
    from gricean.coding import load_sheet, agreement
    from gricean.report import build
    return Path, agreement, build, expected_templates, ladders, load_sheet, mo


@app.cell
def _(Path, agreement, load_sheet, mo):
    c1, _c2, key = Path("data/coding/coder1.csv"), Path("data/coding/coder2.csv"), Path("data/coding/coder2.key.csv")
    if c1.exists() and _c2.exists():
        out = mo.vstack([mo.md("Agreement of the two coders (Cohen's kappa; `same` = rows with the same code)"),
                         agreement(load_sheet(c1), load_sheet(_c2, key=key)).round(3)])
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
