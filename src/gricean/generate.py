from __future__ import annotations

from pathlib import Path

from .io import item_stamp, merge_fresh, write_jsonl
from .lm import LM, ModelSpec, spec_stamp
from .stimuli import Item

KEY = ("item_id", "version", "mode", "seed")


def response_path(model_name: str, template: bool, root: Path = Path("data/responses")) -> Path:
    return Path(root) / f"{model_name}__{'tmpl' if template else 'raw'}.jsonl"


def _plan(items: list[Item], gen_cfg: dict) -> list[tuple[Item, str, str, int | None]]:
    plan = []
    for it in items:
        for version in it.versions:
            if gen_cfg.get("greedy", True):
                plan.append((it, version, "greedy", None))
            for seed in gen_cfg["samples"]["seeds"][: gen_cfg["samples"]["n"]]:
                plan.append((it, version, "sample", seed))
    return plan


def run_generate(model: LM, spec: ModelSpec, items: list[Item], *, template: bool,
                 gen_cfg: dict, stimuli_checksum: str, root: Path = Path("data/responses")) -> int:
    path = response_path(spec.name, template, root)
    plan = _plan(items, gen_cfg)
    by_key = {}
    wanted = {}
    for it, version, mode, seed in plan:
        prompt_text = model.format(it.versions[version], template)
        stamp = item_stamp(prompt_text, mode, seed, spec_stamp(spec),
                           gen_cfg["max_new_tokens"], gen_cfg["samples"]["temperature"])
        key = (it.id, version, mode, seed)
        by_key[key] = (it, prompt_text, stamp)
        wanted[key] = stamp
    keep, todo = merge_fresh(path, KEY, "stamp", wanted)
    new_rows = []
    for key in todo:
        it, prompt_text, stamp = by_key[key]
        _, version, mode, seed = key
        temperature = None if mode == "greedy" else gen_cfg["samples"]["temperature"]
        response = model.generate(prompt_text, max_new_tokens=gen_cfg["max_new_tokens"],
                                  temperature=temperature, seed=seed)
        new_rows.append({
            "item_id": it.id, "category": it.category, "language": it.language,
            "control": it.control, "version": version, "model": spec.name,
            "template": template, "mode": mode, "seed": seed,
            "prompt_text": prompt_text, "response": response,
            "stamp": stamp, "stimuli_checksum": stimuli_checksum,
        })
    if new_rows:
        write_jsonl(path, keep + new_rows)
    return len(new_rows)
