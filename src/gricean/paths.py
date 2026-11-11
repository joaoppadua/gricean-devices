from __future__ import annotations

import os
from pathlib import Path

# What marks the root of a checkout: every default path in the package is relative to this folder.
MARKERS = ("pyproject.toml", "configs", "stimuli")


def repo_root(start: Path | str | None = None) -> Path:
    """The checkout that contains `start` (default: the working directory)."""
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for folder in (here, *here.parents):
        if all((folder / m).exists() for m in MARKERS):
            return folder
    raise FileNotFoundError(
        f"{here} is not inside a gricean-devices checkout (no folder above it holds {', '.join(MARKERS)})")


def enter_repo_root(start: Path | str | None = None) -> Path:
    """Make the repo root the working directory and return it.

    The `make` stages always start at the root; a notebook may not (the VS Code marimo extension
    starts its kernel inside `notebooks/`), and then `configs/models.yaml`, `stimuli/` and `data/`
    are not found. Notebooks call this in their first cell with `mo.notebook_dir()`."""
    root = repo_root(start)
    os.chdir(root)
    return root
