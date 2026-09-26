from __future__ import annotations

import re
from pathlib import Path

from .io import item_stamp, merge_fresh, read_jsonl, write_jsonl
from .lm import LM, ModelSpec, spec_stamp
from .stimuli import Item

KEY = ("item_id", "version", "mode", "seed")   # version/mode/seed are constant here
FLAG_THRESHOLD = 0.5
RULE = "containment-v2"
MIN_WORDS = 2             # samples shorter than this (e.g. an immediate newline) are never chosen   # bump when the selection rule changes so stale rows recompute


def neither_path(family: str, root: Path = Path("stimuli/neither")) -> Path:
    return Path(root) / f"{family}.jsonl"


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def token_overlap(sample: str, reference: str) -> float:
    """Fraction of the reference continuation's words that appear in the sample.

    Containment rather than Jaccard: a long sample that embeds the whole intended
    answer must score 1.0, not be diluted by its extra words."""
    ws, wr = _words(sample), _words(reference)
    if not wr:
        return 1.0
    return len(ws & wr) / len(wr)


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
                           ",".join(map(str, cfg["seeds"][: cfg["n"]])), RULE)
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
        valid = [s for s in samples if len(s.split()) >= MIN_WORDS]
        scored = [(max(token_overlap(s, it.literal_continuation), token_overlap(s, it.intended_continuation)), s)
                  for s in valid]
        if scored:
            overlap, chosen = min(scored, key=lambda t: t[0])
            flag = overlap >= FLAG_THRESHOLD
        else:                      # nothing usable: keep the longest sample, force author review
            chosen = max(samples, key=len) or "(no usable sample)"
            overlap, flag = 1.0, True
        new_rows.append({
            "item_id": it.id, "version": "full", "mode": "neither", "seed": None,
            "family": family, "source_model": source_spec.name, "source_revision": source_spec.revision,
            "seeds": seeds, "samples": samples, "chosen": chosen, "overlap": overlap,
            "flag": flag, "approved": False, "stamp": stamp, "stimuli_checksum": stimuli_checksum,
        })
    if new_rows:
        write_jsonl(path, keep + new_rows)
    return len(new_rows)


def load_neither(family: str, root: Path = Path("stimuli/neither")) -> dict[str, str]:
    """item_id -> chosen continuation. Flagged rows must carry `approved: true` (author's pass)."""
    rows = read_jsonl(neither_path(family, root))
    blocked = [r["item_id"] for r in rows if r.get("flag") and not r.get("approved")]
    if blocked:
        raise ValueError(
            f"flagged neither continuations for {blocked} in {neither_path(family, root)}: review each row, "
            "then either edit `chosen` or set `approved: true`")
    return {r["item_id"]: r["chosen"] for r in rows}
