from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import yaml

ALPACA = "### Instruction:\n{prompt}\n\n### Response:\n"
# Every decoding knob pinned explicitly, so a checkpoint's generation_config.json
# (e.g. Qwen2.5-Instruct ships top_k=20, top_p=0.8, repetition_penalty=1.05) cannot leak in.
DECODING = {"top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "num_beams": 1}


class LM(Protocol):
    name: str

    def format(self, prompt: str, template: bool) -> str: ...

    def generate(self, prompt: str, *, max_new_tokens: int,
                 temperature: float | None, seed: int | None) -> str: ...

    def continuation_logprob(self, prompt: str, continuation: str) -> tuple[float, int]: ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    revision: str
    family: str
    template: str            # "none" | "chat" | "alpaca"
    adapter: str | None = None


def _cfg(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_registry(path: Path = Path("configs/models.yaml")) -> dict[str, ModelSpec]:
    return {name: ModelSpec(name=name, **m) for name, m in _cfg(path)["models"].items()}


def ladders(path: Path = Path("configs/models.yaml")) -> dict[str, list[str]]:
    return _cfg(path)["ladders"]


def neither_source(path: Path = Path("configs/models.yaml")) -> dict[str, str]:
    return _cfg(path)["neither_source"]


def expected_templates(path: Path = Path("configs/models.yaml")) -> dict[str, bool]:
    """model name -> whether its main condition is the templated one (False for base models)."""
    return {name: spec.template != "none" for name, spec in load_registry(path).items()}


def resolve_revision(repo: str, revision: str) -> str:
    """Pin a symbolic revision (e.g. 'main') to its commit sha, from the local HF cache or the Hub."""
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import constants
    ref = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / revision
    if ref.exists():
        return ref.read_text().strip()
    from huggingface_hub import HfApi
    return HfApi().model_info(repo, revision=revision).sha


def render_template(spec: ModelSpec, prompt: str, tokenizer) -> str:
    if spec.template == "none":
        return prompt
    if spec.template == "alpaca":
        return ALPACA.format(prompt=prompt)
    if spec.template == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    raise ValueError(f"unknown template {spec.template!r}")


class FakeLM:
    """Deterministic stand-in for tests. logprob_rules maps continuation -> total logprob."""

    def __init__(self, name: str, logprob_rules: dict[str, float] | None = None,
                 canned: dict[str, str] | None = None):
        self.name = name
        self.rules = logprob_rules or {}
        self.canned = canned or {}

    def format(self, prompt: str, template: bool) -> str:
        return f"<user>{prompt}</user><assistant>" if template else prompt

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float | None,
                 seed: int | None) -> str:
        if prompt in self.canned:
            return self.canned[prompt]
        tag = "" if seed is None else f" s{seed}"
        return f"[{self.name}{tag}] {prompt[:20]}"

    def continuation_logprob(self, prompt: str, continuation: str) -> tuple[float, int]:
        n = max(1, len(continuation.split()))
        return self.rules.get(continuation, -10.0 * n), n


class HFLM:
    def __init__(self, spec: ModelSpec, model, tokenizer, device: str):
        import torch
        self.spec, self.name = spec, spec.name
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self._torch = torch
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format(self, prompt: str, template: bool) -> str:
        return render_template(self.spec, prompt, self.tokenizer) if template else prompt

    def _ids(self, text: str, special: bool):
        return self.tokenizer(text, add_special_tokens=special, return_tensors="pt").input_ids

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float | None,
                 seed: int | None) -> str:
        torch = self._torch
        ids = self._ids(prompt, special=True).to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, **DECODING)
        if temperature is None:
            kwargs.update(do_sample=False)
        else:
            kwargs.update(do_sample=True, temperature=temperature)
        with torch.no_grad():
            out = self.model.generate(ids, **kwargs)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def continuation_logprob(self, prompt: str, continuation: str) -> tuple[float, int]:
        """Tokenize prompt and continuation separately so the boundary matches training."""
        torch = self._torch
        p_ids = self._ids(prompt, special=True)
        c_ids = self._ids(continuation, special=False)
        if c_ids.shape[1] == 0:
            raise ValueError(f"empty continuation for prompt {prompt[:60]!r}")
        ids = torch.cat([p_ids, c_ids], dim=1).to(self.device)
        with torch.no_grad():
            logits = self.model(ids).logits.float()
        logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
        targets = ids[0, 1:]
        start = p_ids.shape[1] - 1
        tok_lp = logprobs[start:, :].gather(1, targets[start:].unsqueeze(1)).squeeze(1)
        return float(tok_lp.sum()), int(c_ids.shape[1])


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(spec: ModelSpec, device: str | None = None) -> HFLM:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or pick_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    spec = replace(spec, revision=resolve_revision(spec.repo, spec.revision))
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(spec.repo, revision=spec.revision, dtype=dtype)
    if spec.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec.adapter)
    model.to(device).eval()
    return HFLM(spec, model, tok, device)


def spec_stamp(spec: ModelSpec) -> str:
    """Stable identifier of a model configuration (repo, revision, template, adapter weights)."""
    weights = ""
    if spec.adapter:
        f = Path(spec.adapter) / "adapter_model.safetensors"
        weights = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "missing"
    return hashlib.sha256(
        f"{spec.repo}@{spec.revision}|{spec.template}|{spec.adapter}|{weights}".encode()).hexdigest()[:16]
