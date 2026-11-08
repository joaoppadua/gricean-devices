from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SORT_FIELDS = ("item_id", "version", "mode", "seed")


def _sort_key(row: dict):
    return tuple("" if row.get(f) is None else str(row.get(f)) for f in SORT_FIELDS)


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in sorted(rows, key=_sort_key):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def item_stamp(*parts: str) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def merge_fresh(path: Path, key_fields: tuple[str, ...], stamp_field: str,
                wanted: dict[tuple, str]) -> tuple[list[dict], list[tuple]]:
    existing = {tuple(r.get(f) for f in key_fields): r for r in read_jsonl(path)}
    keep, todo = [], []
    for key, stamp in wanted.items():
        row = existing.get(key)
        if row is not None and row.get(stamp_field) == stamp:
            keep.append(row)
        else:
            todo.append(key)
    return keep, todo
