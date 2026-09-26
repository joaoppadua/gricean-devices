from __future__ import annotations

import re
from pathlib import Path

from .io import item_stamp, merge_fresh, read_jsonl, write_jsonl
from .lm import LM, ModelSpec, spec_stamp
from .stimuli import Item

KEY = ("item_id", "version", "mode", "seed")   # version/mode/seed are constant here
FLAG_THRESHOLD = 0.5


def neither_path(family: str, root: Path = Path("stimuli/neither")) -> Path:
    return Path(root) / f"{family}.jsonl"


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def token_overlap(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def truncate(text: str, max_tokens: int = 20) -> str:
    first = text.split("\n", 1)[0].strip()
    return " ".join(first.split()[:max_tokens])


def sample_neither(source: LM, source_spec: ModelSpec, family: str, items: list[Item], *,
                   cfg: dict, stimuli_checksum: str, root: Path = Path("stimuli/neither")) -> int:
    path = neither_path(family, root)
    targets = [it for it in items if not it.control]
    wanted, by_key = {}, {}
    for it in targets:
        stamp = item_stamp(it.prompt, it.literal_continuation, it.intended_continuation,
                           spec_stamp(source_spec), cfg["temperature"], cfg["max_new_tokens"],
                           ",".join(map(str, cfg["seeds"][: cfg["n"]])))
        key = (it.id, "full", "neither", None)
        wanted[key], by_key[key] = stamp, (it, stamp)
    keep, todo = merge_fresh(path, KEY, "stamp", wanted)
    new_rows = []
    for key in todo:
        it, stamp = by_key[key]
        seeds = cfg["seeds"][: cfg["n"]]
        samples = [truncate(source.generate(it.prompt, max_new_tokens=cfg["max_new_tokens"],
                                            temperature=cfg["temperature"], seed=s), cfg["max_new_tokens"])
                   for s in seeds]
        scored = [(max(token_overlap(s, it.literal_continuation), token_overlap(s, it.intended_continuation)), s)
                  for s in samples]
        overlap, chosen = min(scored, key=lambda t: t[0])
        new_rows.append({
            "item_id": it.id, "version": "full", "mode": "neither", "seed": None,
            "family": family, "source_model": source_spec.name, "source_revision": source_spec.revision,
            "seeds": seeds, "samples": samples, "chosen": chosen, "overlap": overlap,
            "flag": overlap >= FLAG_THRESHOLD, "stamp": stamp, "stimuli_checksum": stimuli_checksum,
        })
    if new_rows:
        write_jsonl(path, keep + new_rows)
    return len(new_rows)


def load_neither(family: str, root: Path = Path("stimuli/neither")) -> dict[str, str]:
    return {r["item_id"]: r["chosen"] for r in read_jsonl(neither_path(family, root))}
