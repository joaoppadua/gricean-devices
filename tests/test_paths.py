import importlib.util
import shutil
from pathlib import Path

import pytest

from gricean import paths

REPO_NOTEBOOKS = Path("notebooks").resolve()


def _checkout(root: Path) -> Path:
    """A minimal stand-in for a checkout of this repo, built in tmp_path."""
    for d in ("configs", "stimuli", "notebooks"):
        (root / d).mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    (root / "configs" / "models.yaml").write_text(
        "models:\n  m: {repo: org/m, revision: main, family: f, template: none}\n"
        "ladders: {olmo: [m]}\nneither_source: {f: m}\n")
    return root


def test_repo_root_is_found_from_the_root_and_from_a_subfolder(tmp_path: Path):
    root = _checkout(tmp_path / "repo")
    assert paths.repo_root(root) == root
    assert paths.repo_root(root / "notebooks") == root


def test_repo_root_defaults_to_the_working_directory(tmp_path: Path, monkeypatch):
    root = _checkout(tmp_path / "repo")
    monkeypatch.chdir(root / "notebooks")
    assert paths.repo_root() == root
    assert paths.repo_root(None) == root      # `mo.notebook_dir()` may return None


def test_repo_root_outside_a_checkout_says_where_it_started(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not inside a gricean-devices checkout"):
        paths.repo_root(tmp_path)


def test_enter_repo_root_makes_relative_paths_resolve(tmp_path: Path, monkeypatch):
    from gricean.lm import ladders
    root = _checkout(tmp_path / "repo")
    monkeypatch.chdir(root / "notebooks")     # where the VS Code marimo extension starts the kernel
    with pytest.raises(FileNotFoundError):
        ladders()
    assert paths.enter_repo_root() == root
    assert Path.cwd() == root
    assert ladders() == {"olmo": ["m"]}


def test_responses_notebook_runs_when_started_inside_notebooks(tmp_path: Path, monkeypatch):
    """Regression: launched by the VS Code extension, the kernel's working directory is `notebooks/`
    and `configs/models.yaml` was not found."""
    root = _checkout(tmp_path / "repo")
    nb = root / "notebooks" / "02_responses.py"
    shutil.copy(REPO_NOTEBOOKS / "02_responses.py", nb)
    monkeypatch.chdir(root / "notebooks")
    spec = importlib.util.spec_from_file_location("nb02_fixture", nb)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.run()                              # raised FileNotFoundError before the fix
    assert Path.cwd() == root


def test_responses_notebook_opens_on_the_first_item(tmp_path: Path, monkeypatch):
    """With nothing selected the comparison cell is blank, and a notebook that has already run
    looks as if no cell could run. It opens on the first item instead."""
    import json
    root = _checkout(tmp_path / "repo")
    rows = [{"item_id": i, "version": "full", "mode": "greedy", "seed": None, "model": "m",
             "template": False, "response": f"answer to {i}"} for i in ("en_b_01", "en_a_01")]
    (root / "data" / "responses").mkdir(parents=True)
    (root / "data" / "responses" / "m__raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    nb = root / "notebooks" / "02_responses.py"
    shutil.copy(REPO_NOTEBOOKS / "02_responses.py", nb)
    monkeypatch.chdir(root / "notebooks")
    spec = importlib.util.spec_from_file_location("nb02_fixture_items", nb)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, defs = mod.app.run()
    assert defs["item"].value == "en_a_01"
    assert "answer to en_a_01" in defs["out"].text
