from __future__ import annotations

import json
import random
from pathlib import Path

from .lm import ALPACA, ModelSpec

DOLLY_KEEP = ("open_qa", "general_qa", "brainstorming", "classification", "creative_writing")


def filter_dolly(records: list[dict]) -> list[dict]:
    return [r for r in records if r["category"] in DOLLY_KEEP and not r.get("context")]


def load_dolly() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
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


def train_lora(base_spec: ModelSpec, rows: list[dict], out_dir: Path, cfg: dict,
               device: str | None = None) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments, set_seed)
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
    ds = Dataset.from_list([{"text": r["text"] + tok.eos_token} for r in rows])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg["max_seq_len"]),
                batched=True, remove_columns=["text"])
    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "trainer"), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], logging_steps=20, save_strategy="no", report_to=[],
        seed=cfg["seed"], use_cpu=(device == "cpu"), dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    (Path(out_dir) / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    return Path(out_dir)
