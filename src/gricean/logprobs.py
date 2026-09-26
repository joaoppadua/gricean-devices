from __future__ import annotations

import math
from pathlib import Path

from .io import item_stamp, merge_fresh, write_jsonl
from .lm import LM, ModelSpec, spec_stamp
from .stimuli import Item

KEY = ("item_id", "version", "mode", "seed")
OPTIONS = ("literal", "intended", "neither")


def logprob_path(model_name: str, template: bool, root: Path = Path("data/logprobs")) -> Path:
    return Path(root) / f"{model_name}__{'tmpl' if template else 'raw'}.jsonl"


def three_way(lp: dict[str, tuple[float, int]]) -> dict:
    norm = {k: lp[k][0] / max(1, lp[k][1]) for k in OPTIONS}
    m = max(norm.values())
    exp = {k: math.exp(v - m) for k, v in norm.items()}
    z = sum(exp.values())
    return {"norm": norm, "share": {k: v / z for k, v in exp.items()},
            "log_odds": norm["intended"] - norm["literal"]}


def run_logprobs(model: LM, spec: ModelSpec, items: list[Item], neither: dict[str, str], *,
                 template: bool, stimuli_checksum: str, root: Path = Path("data/logprobs")) -> int:
    path = logprob_path(spec.name, template, root)
    targets = [it for it in items if not it.control]
    for it in targets:
        if it.id not in neither:
            raise KeyError(f"no neither continuation for item {it.id!r}")
    wanted, by_key = {}, {}
    for it in targets:
        for version in it.versions:
            prompt_text = model.format(it.versions[version], template)
            conts = {"literal": it.literal_continuation, "intended": it.intended_continuation,
                     "neither": neither[it.id]}
            stamp = item_stamp(prompt_text, *conts.values(), spec_stamp(spec))
            key = (it.id, version, "logprob", None)
            wanted[key], by_key[key] = stamp, (it, prompt_text, conts, stamp)
    keep, todo = merge_fresh(path, KEY, "stamp", wanted)
    new_rows = []
    for key in todo:
        it, prompt_text, conts, stamp = by_key[key]
        lp = {k: model.continuation_logprob(prompt_text, v) for k, v in conts.items()}
        tw = three_way(lp)
        row = {"item_id": it.id, "category": it.category, "language": it.language,
               "version": key[1], "model": spec.name, "template": template,
               "mode": "logprob", "seed": None}
        for k in OPTIONS:
            row[f"lp_{k}"], row[f"n_{k}"] = lp[k]
            row[f"norm_{k}"], row[f"share_{k}"] = tw["norm"][k], tw["share"][k]
        row.update(log_odds=tw["log_odds"], stamp=stamp, stimuli_checksum=stimuli_checksum)
        new_rows.append(row)
    if new_rows:
        write_jsonl(path, keep + new_rows)
    return len(new_rows)
