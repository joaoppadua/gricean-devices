import importlib.util
from pathlib import Path
import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("*.py"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_imports_and_is_marimo_app(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "app")
    src = path.read_text()
    assert "def " not in src.replace("def _(", "").replace("def __(", ""), \
        "notebooks must not define helper functions; put logic in the package"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_enters_the_repo_root_in_its_first_cell(path: Path):
    """Every path in the package is relative to the repo root, and a notebook may be started from
    elsewhere (the VS Code extension starts it inside `notebooks/`). The first cell is the one every
    other cell depends on, so the call must be there."""
    first_cell = path.read_text().split("@app.cell")[1]
    assert "enter_repo_root(mo.notebook_dir())" in first_cell
