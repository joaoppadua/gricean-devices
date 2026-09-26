import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    from gricean.stimuli import load_all, current_checksum, CATEGORIES
    return CATEGORIES, Path, current_checksum, load_all, mo, pd


@app.cell
def _(current_checksum, load_all, mo, pd):
    items = load_all()
    checksum = current_checksum()
    df = pd.DataFrame([{"id": i.id, "category": i.category, "language": i.language, "control": i.control,
                        "n_versions": len(i.versions), "prompt": i.prompt} for i in items])
    mo.md(f"**Frozen stimuli** · checksum `{checksum[:12]}…` · {len(items)} items")
    return checksum, df, items


@app.cell
def _(CATEGORIES, mo):
    category = mo.ui.dropdown(list(CATEGORIES), value=CATEGORIES[0], label="category")
    category
    return (category,)


@app.cell
def _(category, df, mo):
    mo.ui.table(df[df["category"] == category.value], selection=None)
    return


@app.cell
def _(category, items, mo):
    rows = []
    for it in items:
        if it.category == category.value:
            for v, text in it.versions.items():
                rows.append(f"| {it.id} | {v} | {text} |")
    mo.md("| id | version | prompt |\n|---|---|---|\n" + "\n".join(rows))
    return


if __name__ == "__main__":
    app.run()
