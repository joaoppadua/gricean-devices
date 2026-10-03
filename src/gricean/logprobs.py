from __future__ import annotations

import math
from pathlib import Path

from .generate import plan_versions
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
                 template: bool, stimuli_checksum: str, root: Path = Path("data/logprobs"),
                 raw_separator: str = " ") -> int:
    """Score literal / intended / neither continuations for every dissociation item.

    Raw (untemplated) prompts end mid-line, so continuations get `raw_separator` prefixed to
    them, giving the first token its ordinary space-prefixed form; templated prompts end the
    assistant turn already and get no separator."""
    path = logprob_path(spec.name, template, root)
    targets = [it for it in items if not it.control]
    for it in targets:
        if it.id not in neither:
            raise KeyError(f"no neither continuation for item {it.id!r}")
        if not neither[it.id].strip():
            raise ValueError(f"empty neither continuation for item {it.id!r}")
    wanted, by_key = {}, {}
    for it in targets:
        for version, use_template in plan_versions(it, spec, template):
            prompt_text = model.format(it.versions[version], use_template)
            sep = "" if use_template else raw_separator
            conts = {"literal": sep + it.literal_continuation, "intended": sep + it.intended_continuation,
                     "neither": sep + neither[it.id]}
            stamp = item_stamp(prompt_text, *conts.values(), spec_stamp(spec))
            key = (it.id, version, "logprob", None)
            wanted[key], by_key[key] = stamp, (it, use_template, prompt_text, conts, sep, stamp)
    keep, todo = merge_fresh(path, KEY, "stamp", wanted)
    new_rows = []
    for key in todo:
        it, use_template, prompt_text, conts, sep, stamp = by_key[key]
        lp = {k: model.continuation_logprob(prompt_text, v) for k, v in conts.items()}
        tw = three_way(lp)
        row = {"item_id": it.id, "category": it.category, "language": it.language,
               "version": key[1], "model": spec.name, "template": use_template,
               "mode": "logprob", "seed": None, "prompt_text": prompt_text, "raw_separator": sep}
        for k in OPTIONS:
            row[f"lp_{k}"], row[f"n_{k}"] = lp[k]
            row[f"norm_{k}"], row[f"share_{k}"] = tw["norm"][k], tw["share"][k]
        row.update(log_odds=tw["log_odds"], stamp=stamp, stimuli_checksum=stimuli_checksum)
        new_rows.append(row)
    if new_rows:
        write_jsonl(path, keep + new_rows)
    return len(new_rows)
