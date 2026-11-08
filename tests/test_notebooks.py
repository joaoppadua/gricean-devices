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
