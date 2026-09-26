import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    from gricean.lm import ladders
    from gricean.report import logodds_table
    return Path, json, ladders, logodds_table, mo, pd


@app.cell
def _(Path, json, mo, pd):
    logs = {}
    for _d in sorted(Path("adapters").glob("smollm_*")):
        _f = _d / "train_log.json"
        if _f.exists():
            logs[_d.name] = pd.DataFrame([_r for _r in json.loads(_f.read_text()) if "loss" in _r])
    _out = mo.md("No adapters trained yet. Run `make finetune`.") if not logs else mo.md(
        "\n".join(f"- **{_k}**: {len(_v)} logged steps, final loss {_v['loss'].iloc[-1]:.3f}" for _k, _v in logs.items()))
    _out
    return (logs,)


@app.cell
def _(logs, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    for _k, _v in logs.items():
        ax.plot(_v["step"], _v["loss"], label=_k)
    ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.legend(fontsize=7)
    mo.mpl.interactive(fig) if logs else mo.md("")
    return


@app.cell
def _(Path, ladders, logodds_table, mo):
    try:
        _out = mo.ui.table(logodds_table(Path("data/logprobs"), ladders()["smollm"]).round(3).reset_index())
    except Exception as e:
        _out = mo.md(f"No smollm log-prob data yet: {e}")
    _out
    return


if __name__ == "__main__":
    app.run()
