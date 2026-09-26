import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from gricean.lm import ladders
    from gricean.report import logodds_table, ladder_figure, intent_table
    from gricean.coding import load_sheet
    return Path, intent_table, ladder_figure, ladders, load_sheet, logodds_table, mo


@app.cell
def _(ladders, mo):
    ladder = mo.ui.dropdown(list(ladders()), value="olmo", label="ladder")
    ladder
    return (ladder,)


@app.cell
def _(Path, ladder, ladders, logodds_table, mo):
    models = ladders()[ladder.value]
    try:
        lo = logodds_table(Path("data/logprobs"), models)
        _out = mo.ui.table(lo.round(3).reset_index())
    except Exception as e:  # no data yet
        lo = None
        _out = mo.md(f"No log-prob data yet: {e}")
    _out
    return lo, models


@app.cell
def _(Path, intent_table, ladder_figure, lo, load_sheet, mo, models):
    coded_path = Path("data/coding/coder1.csv")
    if lo is not None and coded_path.exists():
        intent = intent_table(load_sheet(coded_path), models)
        fig_path = Path(".marimo") / "ladder_preview.png"
        ladder_figure(intent, lo, models, fig_path)
        _out = mo.image(str(fig_path))
    else:
        _out = mo.md("Ladder figure appears once both log-probs and the coded sheet exist.")
    _out
    return


if __name__ == "__main__":
    app.run()
