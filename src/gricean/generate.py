from __future__ import annotations

from pathlib import Path

from .io import item_stamp, merge_fresh, write_jsonl
from .lm import DECODING, LM, ModelSpec, spec_stamp
from .stimuli import Item

KEY = ("item_id", "version", "mode", "seed")


def response_path(model_name: str, template: bool, root: Path = Path("data/responses")) -> Path:
    return Path(root) / f"{model_name}__{'tmpl' if template else 'raw'}.jsonl"


def plan_versions(item: Item, spec: ModelSpec, template: bool) -> list[tuple[str, bool]]:
    """(version, use_template) pairs for one item in one pass.

    The embedded version is never templated (spec 4.3). For tuned models it is produced
    in the templated pass only, so the raw pass (the template control) skips it."""
    out = []
    for version in item.versions:
        if version == "embedded":
            if spec.template != "none" and not template:
                continue
            out.append((version, False))
        else:
            out.append((version, template))
    return out


def _plan(items: list[Item], spec: ModelSpec, template: bool, gen_cfg: dict):
    plan = []
    for it in items:
        for version, use_template in plan_versions(it, spec, template):
            if gen_cfg.get("greedy", True):
                plan.append((it, version, use_template, "greedy", None))
            for seed in gen_cfg["samples"]["seeds"][: gen_cfg["samples"]["n"]]:
                plan.append((it, version, use_template, "sample", seed))
    return plan


def run_generate(model: LM, spec: ModelSpec, items: list[Item], *, template: bool,
                 gen_cfg: dict, stimuli_checksum: str, root: Path = Path("data/responses")) -> int:
    path = response_path(spec.name, template, root)
    by_key, wanted = {}, {}
    for it, version, use_template, mode, seed in _plan(items, spec, template, gen_cfg):
        prompt_text = model.format(it.versions[version], use_template)
        stamp = item_stamp(prompt_text, mode, seed, spec_stamp(spec), gen_cfg["max_new_tokens"],
                           gen_cfg["samples"]["temperature"], sorted(DECODING.items()))
        key = (it.id, version, mode, seed)
        by_key[key] = (it, use_template, prompt_text, stamp)
        wanted[key] = stamp
    keep, todo = merge_fresh(path, KEY, "stamp", wanted)
    new_rows = []
    for key in todo:
        it, use_template, prompt_text, stamp = by_key[key]
        _, version, mode, seed = key
        temperature = None if mode == "greedy" else gen_cfg["samples"]["temperature"]
        response = model.generate(prompt_text, max_new_tokens=gen_cfg["max_new_tokens"],
                                  temperature=temperature, seed=seed)
        new_rows.append({
            "item_id": it.id, "category": it.category, "language": it.language,
            "control": it.control, "version": version, "model": spec.name,
            "template": use_template, "mode": mode, "seed": seed,
            "stimulus_text": it.versions[version], "prompt_text": prompt_text, "response": response,
            "stamp": stamp, "stimuli_checksum": stimuli_checksum,
        })
    if new_rows:
        write_jsonl(path, keep + new_rows)
    return len(new_rows)
