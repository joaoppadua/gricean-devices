from pathlib import Path
from gricean import io as IO


def test_roundtrip_sorted(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    IO.write_jsonl(p, [{"item_id": "b", "version": "full", "mode": "greedy", "seed": None, "v": 1},
                       {"item_id": "a", "version": "full", "mode": "greedy", "seed": None, "v": 2}])
    rows = IO.read_jsonl(p)
    assert [r["item_id"] for r in rows] == ["a", "b"]


def test_read_missing_is_empty(tmp_path: Path):
    assert IO.read_jsonl(tmp_path / "nope.jsonl") == []


def test_item_stamp_changes_with_any_part():
    assert IO.item_stamp("a", "b") != IO.item_stamp("a", "c")
    assert len(IO.item_stamp("a")) == 16


def test_merge_fresh_recomputes_only_changed(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    IO.write_jsonl(p, [
        {"item_id": "a", "version": "full", "mode": "greedy", "seed": None, "stamp": "s1", "v": 1},
        {"item_id": "b", "version": "full", "mode": "greedy", "seed": None, "stamp": "s2", "v": 2},
    ])
    wanted = {("a", "full", "greedy", None): "s1",      # unchanged
              ("b", "full", "greedy", None): "s2x",     # stimulus changed
              ("c", "full", "greedy", None): "s3"}      # new
    keep, todo = IO.merge_fresh(p, ("item_id", "version", "mode", "seed"), "stamp", wanted)
    assert [r["item_id"] for r in keep] == ["a"]
    assert sorted(todo) == [("b", "full", "greedy", None), ("c", "full", "greedy", None)]
