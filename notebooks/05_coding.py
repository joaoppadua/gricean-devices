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
    from gricean.report import build, correctness_table
    return Path, agreement, build, correctness_table, expected_templates, ladders, load_sheet, mo


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
    language = mo.ui.dropdown(["en", "pt"], value="en", label="language (never pooled, spec 4.4)")
    go = mo.ui.run_button(label="build paper/ tables")
    mo.hstack([ladder, language, go], justify="start")
    return go, ladder, language


@app.cell
def _(Path, correctness_table, expected_templates, ladder, ladders, language, load_sheet, mo):
    _c1 = Path("data/coding/coder1.csv")
    if _c1.exists():
        _out = mo.vstack([
            mo.md("Correctness of the responses that follow a reading (spec §5.1, step 17): `n` = responses "
                  "coded with that intent, `correct` = share of them coded `correct`; empty where `n` is 0"),
            mo.plain(correctness_table(load_sheet(_c1), ladders()[ladder.value], expected_templates(),
                                       language.value).round(2))])
    else:
        _out = mo.md("The coded sheet is needed for the correctness table.")
    _out
    return


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
