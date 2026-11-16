from __future__ import annotations

import json
import random
from pathlib import Path

from .lm import ALPACA, ModelSpec

DOLLY_KEEP = ("open_qa", "general_qa", "brainstorming", "classification", "creative_writing")


def filter_dolly(records: list[dict]) -> list[dict]:
    return [r for r in records if r["category"] in DOLLY_KEEP and not r.get("context")]


def load_dolly(revision: str) -> list[dict]:
    """The context-free Dolly pairs at a pinned dataset revision (`dataset_revision` in configs/finetune.yaml)."""
    import datasets
    ds = datasets.load_dataset("databricks/databricks-dolly-15k", split="train", revision=revision)
    return filter_dolly([dict(r) for r in ds])


def _rows(pairs: list[tuple[str, str]]) -> list[dict]:
    return [{"instruction": q, "response": a, "text": ALPACA.format(prompt=q) + a} for q, a in pairs]


def prepare(records: list[dict], *, sizes: list[int], seed: int, shuffle_responses: bool) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    recs = list(records)
    rng.shuffle(recs)
    pairs = [(r["instruction"], r["response"]) for r in recs]
    out = {"full": _rows(pairs)}
    for n in sizes:
        out[f"n{n}"] = _rows(pairs[:n])
    if shuffle_responses:
        responses = [a for _, a in pairs]
        rng2 = random.Random(seed + 1)
        rng2.shuffle(responses)
        out["format_only"] = _rows(list(zip([q for q, _ in pairs], responses)))
    return out


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokenize_rows(rows: list[dict], tok, max_len: int) -> list[dict]:
    """Full-sequence LM examples whose labels keep the end-of-text token, so the adapter learns to stop.

    (A collator that derives labels from pad_token_id would mask EOS out whenever pad == eos.)"""
    enc = tok([r["text"] + tok.eos_token for r in rows], truncation=True, max_length=max_len)
    return [{"input_ids": ids, "attention_mask": am, "labels": list(ids)}
            for ids, am in zip(enc["input_ids"], enc["attention_mask"])]


def collate(features: list[dict], pad_id: int, multiple: int) -> dict:
    """Right-pad a batch to the next multiple of `multiple` tokens, so batches come in a few fixed lengths.

    On MPS, memory grows with every new batch shape (measured 2026-11-14: 70 shapes in 80 batches took the run
    from 14 to 18 GiB and then out of memory; with multiples of 64 there are six shapes). The padding carries
    attention 0 and label -100 and sits to the right of a causal model, so no loss term changes."""
    import torch
    n = max(len(f["input_ids"]) for f in features)
    n = -(-n // multiple) * multiple
    pad = lambda key, fill: torch.tensor([f[key] + [fill] * (n - len(f[key])) for f in features])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100)}


def train_lora(base_spec: ModelSpec, rows: list[dict], out_dir: Path, cfg: dict,
               device: str | None = None) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)
    from .lm import pick_device
    device = device or pick_device()
    set_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(base_spec.repo, revision=base_spec.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg.get("dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(base_spec.repo, revision=base_spec.revision, dtype=dtype)
    model = get_peft_model(model, LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                                             lora_dropout=cfg["lora"]["dropout"],
                                             target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM"))
    ds = Dataset.from_list(tokenize_rows(rows, tok, cfg["max_seq_len"]))
    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "trainer"), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], logging_steps=20, save_strategy="no", report_to=[],
        seed=cfg["seed"], use_cpu=(device == "cpu"), dataloader_pin_memory=False,
    )
    class ReleaseMpsCache(TrainerCallback):
        """Hand cached GPU blocks back after each optimizer step and print the driver's total now and then.
        Without it the cache alone climbs past 13 GiB in 200 batches on the 16 GB M3."""
        def on_step_end(self, args, state, control, **kwargs):
            if device != "mps":
                return
            torch.mps.empty_cache()
            if state.global_step % 20 == 0:
                print(f"[mps] step {state.global_step}: {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB", flush=True)

    trainer = Trainer(model=model, args=args, train_dataset=ds, callbacks=[ReleaseMpsCache()],
                      data_collator=lambda features: collate(features, tok.pad_token_id, cfg["pad_to_multiple_of"]))
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    (Path(out_dir) / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    return Path(out_dir)
