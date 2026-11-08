import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    from gricean.io import read_jsonl
    from gricean.lm import ladders, load_registry
    from gricean.generate import response_path
    from gricean import cli
    return Path, cli, ladders, load_registry, mo, pd, read_jsonl, response_path


@app.cell
def _(ladders, mo):
    ladder = mo.ui.dropdown(list(ladders()), value="olmo", label="ladder")
    ladder
    return (ladder,)


@app.cell
def _(ladder, ladders, mo, pd, read_jsonl, response_path):
    models = ladders()[ladder.value]
    frames = []
    for _m in models:
        for _tmpl in (True, False):
            frames.extend(read_jsonl(response_path(_m, _tmpl)))
    resp = pd.DataFrame(frames)
    mo.md(f"{len(resp)} response rows on disk for ladder **{ladder.value}**" if len(resp)
          else "No responses yet. Use the run button below.")
    return models, resp


@app.cell
def _(mo, resp):
    item = mo.ui.dropdown(sorted(resp["item_id"].unique()) if len(resp) else [], label="item")
    version = mo.ui.dropdown(["full", "stripped", "embedded"], value="full", label="version")
    mo.hstack([item, version])
    return item, version


@app.cell
def _(item, mo, models, resp, version):
    if len(resp) and item.value:
        sub = resp[(resp["item_id"] == item.value) & (resp["version"] == version.value) & (resp["mode"] == "greedy")]
        cols = []
        for _m in models:
            for _tmpl in (True, False):
                r = sub[(sub["model"] == _m) & (sub["template"] == _tmpl)]
                if len(r):
                    cols.append(mo.md(f"**{_m}** ({'template' if _tmpl else 'raw'})\n\n{r.iloc[0]['response']}"))
        out = mo.hstack(cols, wrap=True)
    else:
        out = mo.md("")
    out
    return


@app.cell
def _(mo, models):
    run_model = mo.ui.dropdown(models, label="model to generate")
    run = mo.ui.run_button(label="run generation (slow)")
    mo.hstack([run_model, run])
    return run, run_model


@app.cell
def _(cli, mo, run, run_model):
    mo.stop(not run.value)
    rc = cli.main(["generate", "--model", run_model.value, "--raw"]) if run_model.value else 1
    mo.md(f"generate exit code {rc}; re-open the item dropdown to refresh.")
    return


if __name__ == "__main__":
    app.run()
