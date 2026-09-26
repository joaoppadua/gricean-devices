# Gricean Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the replicable pipeline that runs frozen dissociation stimuli through open base/instruction-tuned checkpoints (and a self-trained LoRA), scores responses two ways, and emits the paper's tables, figures and printed examples.

**Architecture:** A single Python package `gricean` with one module per pipeline stage (stimuli → neither → generate → logprobs → coding → report), each idempotent and writing plain JSONL/CSV to disk. All model access goes through one `LM` protocol with a Hugging Face backend and a deterministic `FakeLM` for tests. Marimo notebooks are thin viewers over the package; a CLI (`gricean <stage>`) and a Makefile drive the run.

**Tech Stack:** Python ≥3.12, uv (lockfile), `transformers`, `torch` (MPS/CPU), `peft`, `datasets`, `pyyaml`, `pandas`, `matplotlib`, `marimo`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md`

## Global Constraints

- Everything open-source and runnable on a 16 GB Apple M3; no paid APIs, no subscription-only models.
- Model checkpoints pinned to a revision hash in `configs/models.yaml`; Python deps pinned in `uv.lock`.
- Stimuli frozen (`stimuli/CHECKSUMS`) before any model runs; every stage records the checksum it consumed.
- Every stage is idempotent: skips work whose inputs (stimuli checksum + model revision + config) are unchanged.
- Base models get raw text; tuned models get their chat template and are additionally run raw (`template: false`).
- Generation: greedy (printed) + 3 samples at temperature 0.7 with seeds `[1, 2, 3]`, `max_new_tokens: 150`.
- `neither` continuations are sampled cross-family (OLMo ← Qwen base; Qwen ← OLMo base; SmolLM2 ← OLMo base), temperature 1.0, ≤20 tokens or first line break, 3 samples, keep lowest overlap.
- Rubric values: intent ∈ {literal, intended, neither}; correctness ∈ {correct, incorrect, na}.
- Line 2 excludes control items.
- Notebooks display, the package computes. No logic in notebook cells.
- Data outputs are committed. `make all` regenerates `paper/` from committed data without running any model.

## Review Focus

Inputs the spec implies but that no task's tests would otherwise exercise; each line's test is added to the owning task.

1. **A chat-templated prompt whose continuation begins with a space or newline** must be tokenized at the same boundary the model saw in training; a naive `tok(prompt + cont)` shifts tokens. Owner: Task 3 (`continuation_logprob` tokenizes prompt and continuation separately).
2. **A stimulus file with duplicate `id`s across categories** must be rejected at freeze time, otherwise later JSONL merges silently overwrite. Owner: Task 2.
3. **A `neither` sample that equals the intended continuation** (model got it right by chance) must not be selected; the overlap filter must reject it and the author must see a flag. Owner: Task 6.
4. **A coded sheet with an empty or misspelled rubric cell** must fail loudly at load, not become NaN in the table. Owner: Task 8.
5. **Re-running a stage after one stimulus changes** must recompute only that stimulus's rows and leave others byte-identical. Owner: Task 4 (per-item stamps).

---

## File Structure

```
pyproject.toml, uv.lock, Makefile, README.md, .gitignore
configs/models.yaml          registry: name → {repo, revision, template, family}
configs/generation.yaml      greedy/sample settings, seeds, max_new_tokens
configs/finetune.yaml        LoRA hyperparameters, data subsets
stimuli/en/<category>.yaml   hand-written items (author)
stimuli/pt/<category>.yaml   mirrored items (author)
stimuli/ablation.yaml        filler paragraphs and politeness list for versions
stimuli/neither/<family>.jsonl   sampled degenerate continuations
stimuli/CHECKSUMS            sha256 of every stimulus file
src/gricean/__init__.py
src/gricean/stimuli.py       Item dataclass, load_all, versions, validate, freeze
src/gricean/lm.py            LM protocol, FakeLM, HFLM backend, registry loader
src/gricean/io.py            JSONL read/write, stamps, is_fresh
src/gricean/generate.py      run_generate(model, version, ...) → data/responses/
src/gricean/neither.py       sample_neither(family, ...) → stimuli/neither/
src/gricean/logprobs.py      score(model, version) → data/logprobs/
src/gricean/coding.py        export_sheet, load_sheet, cohen_kappa
src/gricean/report.py        tables, figure, examples → paper/
src/gricean/finetune.py      prepare_dolly, train_lora
src/gricean/cli.py           argparse entry point
notebooks/01_stimuli.py … 05_coding.py
tests/conftest.py            FakeLM fixture, tmp stimuli
tests/test_*.py
data/responses/, data/logprobs/, adapters/, paper/
```

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `Makefile`, `README.md`, `.gitignore`, `src/gricean/__init__.py`, `tests/test_scaffold.py`, `configs/generation.yaml`

**Interfaces:**
- Produces: importable package `gricean` with `__version__`; `make test`.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "gricean"
version = "0.1.0"
description = "Computational simulation for 'LLMs are Gricean machines'"
requires-python = ">=3.12"
dependencies = [
  "torch>=2.4",
  "transformers>=4.45",
  "peft>=0.13",
  "datasets>=3.0",
  "accelerate>=1.0",
  "pyyaml>=6",
  "pandas>=2.2",
  "matplotlib>=3.9",
  "marimo>=0.9",
  "huggingface_hub>=0.25",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-timeout>=2"]

[project.scripts]
gricean = "gricean.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gricean"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs a real model download"]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Write package init, gitignore, Makefile, README stub**

`src/gricean/__init__.py`:
```python
__version__ = "0.1.0"
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.marimo/
hf_cache/
```

`Makefile`:
```make
.PHONY: install test freeze neither generate logprobs sheet report all
install:
	uv sync --extra dev
test:
	uv run pytest
freeze:
	uv run gricean freeze
neither:
	uv run gricean neither --family olmo --family qwen --family smollm
generate:
	uv run gricean generate --all
logprobs:
	uv run gricean logprobs --all
sheet:
	uv run gricean sheet
report:
	uv run gricean report
all: report
```

`README.md`:
```markdown
# gricean-devices

Simulation code for Pádua, "LLMs are Gricean machines" (Revista Texto Livre, 2027).

See `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md` for the design.

## Quick start

    make install
    make test
    make all     # regenerates paper/ from committed data; no model needed
```

`configs/generation.yaml`:
```yaml
max_new_tokens: 150
greedy: true
samples:
  n: 3
  temperature: 0.7
  seeds: [1, 2, 3]
neither:
  n: 3
  temperature: 1.0
  max_new_tokens: 20
  seeds: [11, 12, 13]
```

- [ ] **Step 3: Write the scaffold test**

`tests/test_scaffold.py`:
```python
import gricean

def test_package_importable():
    assert gricean.__version__ == "0.1.0"
```

- [ ] **Step 4: Install and run**

Run: `uv sync --extra dev && uv run pytest -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock Makefile README.md .gitignore src tests configs
git commit -m "chore: project scaffold"
```

---

### Task 2: Stimuli loading, versions, freeze

**Files:**
- Create: `src/gricean/stimuli.py`, `stimuli/ablation.yaml`, `stimuli/en/indirect_request.yaml` (seed example), `tests/test_stimuli.py`

**Interfaces:**
- Produces:
  ```python
  CATEGORIES = ("indirect_request", "pattern_continuation", "embedded_instruction",
                "malformed_instruction", "underspecified_request", "rhetorical_question")
  VERSIONS = ("full", "stripped", "embedded")
  ABLATED_CATEGORIES = ("indirect_request", "embedded_instruction", "pattern_continuation")

  @dataclass(frozen=True)
  class Item:
      id: str; category: str; language: str; control: bool
      prompt: str; literal_reading: str; intended_reading: str
      literal_continuation: str | None; intended_continuation: str | None
      versions: dict[str, str]          # version name -> prompt text (always has "full")

  def load_all(root: Path = Path("stimuli")) -> list[Item]
  def validate(items: list[Item]) -> None          # raises ValueError
  def freeze(root: Path = Path("stimuli")) -> str   # writes CHECKSUMS, returns combined sha256
  def current_checksum(root: Path = Path("stimuli")) -> str  # reads CHECKSUMS
  def make_versions(prompt: str, category: str, ablation: dict) -> dict[str, str]
  ```

- [ ] **Step 1: Write ablation config and one seed stimulus file**

`stimuli/ablation.yaml`:
```yaml
politeness: ["please", "kindly", "thanks", "thank you", "if you don't mind", "por favor", "obrigado", "obrigada"]
strip_punctuation: "?!.,;:"
embedding:
  en: "The afternoon had been slow at the office and the printer jammed twice. {prompt} Later the rain started and most people left early, which nobody minded."
  pt: "A tarde no escritório tinha sido lenta e a impressora travou duas vezes. {prompt} Mais tarde começou a chover e a maioria foi embora cedo, o que ninguém lamentou."
```

`stimuli/en/indirect_request.yaml`:
```yaml
category: indirect_request
language: en
items:
  - id: en_ir_01
    control: false
    prompt: "Can you tell me what year Brazil became independent?"
    literal_reading: "A yes/no question about the addressee's ability."
    intended_reading: "A request to state the year."
    literal_continuation: "Yes, I can."
    intended_continuation: "Brazil became independent in 1822."
  - id: en_ir_c01
    control: true
    prompt: "What year did Brazil become independent?"
    literal_reading: "A request to state the year."
    intended_reading: "A request to state the year."
    literal_continuation: null
    intended_continuation: null
```

- [ ] **Step 2: Write the failing tests**

`tests/test_stimuli.py`:
```python
from pathlib import Path
import textwrap
import pytest
from gricean import stimuli as S

ABL = {
    "politeness": ["please"],
    "strip_punctuation": "?!.,",
    "embedding": {"en": "Before. {prompt} After.", "pt": "Antes. {prompt} Depois."},
}

def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))

@pytest.fixture
def root(tmp_path: Path) -> Path:
    write(tmp_path, "ablation.yaml", """
        politeness: ["please"]
        strip_punctuation: "?!.,"
        embedding:
          en: "Before. {prompt} After."
          pt: "Antes. {prompt} Depois."
    """)
    write(tmp_path, "en/indirect_request.yaml", """
        category: indirect_request
        language: en
        items:
          - id: en_ir_01
            control: false
            prompt: "Can you please tell me the time?"
            literal_reading: "ability"
            intended_reading: "request"
            literal_continuation: "Yes."
            intended_continuation: "It is noon."
          - id: en_ir_c01
            control: true
            prompt: "What time is it?"
            literal_reading: "request"
            intended_reading: "request"
            literal_continuation: null
            intended_continuation: null
    """)
    write(tmp_path, "en/rhetorical_question.yaml", """
        category: rhetorical_question
        language: en
        items:
          - id: en_rq_01
            control: false
            prompt: "Who likes waiting?"
            literal_reading: "question"
            intended_reading: "complaint"
            literal_continuation: "Some people do."
            intended_continuation: "Nobody, waiting is frustrating."
    """)
    return tmp_path

def test_load_all_builds_items_with_versions(root):
    items = S.load_all(root)
    ids = {i.id for i in items}
    assert ids == {"en_ir_01", "en_ir_c01", "en_rq_01"}
    ir = next(i for i in items if i.id == "en_ir_01")
    assert set(ir.versions) == {"full", "stripped", "embedded"}
    assert ir.versions["full"] == "Can you please tell me the time?"
    assert ir.versions["stripped"] == "Can you tell me the time"
    assert ir.versions["embedded"] == "Before. Can you please tell me the time? After."
    rq = next(i for i in items if i.id == "en_rq_01")
    assert set(rq.versions) == {"full"}          # not an ablated category

def test_controls_get_only_full_version(root):
    items = S.load_all(root)
    c = next(i for i in items if i.control)
    assert set(c.versions) == {"full"}

def test_validate_rejects_duplicate_ids(root):
    write(root, "pt/indirect_request.yaml", """
        category: indirect_request
        language: pt
        items:
          - id: en_ir_01
            control: false
            prompt: "Você pode me dizer as horas?"
            literal_reading: "a"
            intended_reading: "b"
            literal_continuation: "Posso."
            intended_continuation: "É meio-dia."
    """)
    with pytest.raises(ValueError, match="duplicate id"):
        S.load_all(root)

def test_validate_rejects_dissociation_item_without_continuations(root):
    write(root, "en/malformed_instruction.yaml", """
        category: malformed_instruction
        language: en
        items:
          - id: en_mi_01
            control: false
            prompt: "tel me the tiem"
            literal_reading: "a"
            intended_reading: "b"
            literal_continuation: null
            intended_continuation: "It is noon."
    """)
    with pytest.raises(ValueError, match="en_mi_01"):
        S.load_all(root)

def test_validate_rejects_unknown_category(root):
    write(root, "en/jokes.yaml", """
        category: jokes
        language: en
        items: []
    """)
    with pytest.raises(ValueError, match="jokes"):
        S.load_all(root)

def test_freeze_writes_checksums_and_is_stable(root):
    a = S.freeze(root)
    assert (root / "CHECKSUMS").exists()
    assert S.current_checksum(root) == a
    assert S.freeze(root) == a
    (root / "en/rhetorical_question.yaml").write_text(
        (root / "en/rhetorical_question.yaml").read_text().replace("waiting", "queuing"))
    assert S.freeze(root) != a
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_stimuli.py -v`
Expected: FAIL with `ImportError` / `AttributeError` on `gricean.stimuli`.

- [ ] **Step 4: Implement stimuli.py**

`src/gricean/stimuli.py`:
```python
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
import yaml

CATEGORIES = (
    "indirect_request", "pattern_continuation", "embedded_instruction",
    "malformed_instruction", "underspecified_request", "rhetorical_question",
)
VERSIONS = ("full", "stripped", "embedded")
ABLATED_CATEGORIES = ("indirect_request", "embedded_instruction", "pattern_continuation")
LANGUAGES = ("en", "pt")


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    language: str
    control: bool
    prompt: str
    literal_reading: str
    intended_reading: str
    literal_continuation: str | None
    intended_continuation: str | None
    versions: dict[str, str]


def _strip(prompt: str, ablation: dict) -> str:
    text = prompt
    for marker in sorted(ablation["politeness"], key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(marker) + r"\b", "", text, flags=re.IGNORECASE)
    text = text.translate(str.maketrans("", "", ablation["strip_punctuation"]))
    return re.sub(r"\s+", " ", text).strip()


def make_versions(prompt: str, category: str, ablation: dict, language: str = "en",
                  control: bool = False) -> dict[str, str]:
    versions = {"full": prompt}
    if control or category not in ABLATED_CATEGORIES:
        return versions
    versions["stripped"] = _strip(prompt, ablation)
    versions["embedded"] = ablation["embedding"][language].format(prompt=prompt)
    return versions


def _load_ablation(root: Path) -> dict:
    return yaml.safe_load((root / "ablation.yaml").read_text())


def _stimulus_files(root: Path) -> list[Path]:
    return sorted(p for lang in LANGUAGES for p in (root / lang).glob("*.yaml"))


def load_all(root: Path = Path("stimuli")) -> list[Item]:
    ablation = _load_ablation(root)
    items: list[Item] = []
    for path in _stimulus_files(root):
        doc = yaml.safe_load(path.read_text())
        category, language = doc["category"], doc["language"]
        if category not in CATEGORIES:
            raise ValueError(f"{path}: unknown category {category!r}")
        if language not in LANGUAGES:
            raise ValueError(f"{path}: unknown language {language!r}")
        for raw in doc["items"]:
            control = bool(raw.get("control", False))
            items.append(Item(
                id=raw["id"], category=category, language=language, control=control,
                prompt=raw["prompt"], literal_reading=raw["literal_reading"],
                intended_reading=raw["intended_reading"],
                literal_continuation=raw.get("literal_continuation"),
                intended_continuation=raw.get("intended_continuation"),
                versions=make_versions(raw["prompt"], category, ablation, language, control),
            ))
    validate(items)
    return items


def validate(items: list[Item]) -> None:
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"duplicate id {it.id!r}")
        seen.add(it.id)
        if not it.control and not (it.literal_continuation and it.intended_continuation):
            raise ValueError(f"{it.id}: dissociation items need both continuations")


def _digest(root: Path) -> tuple[list[tuple[str, str]], str]:
    files = [root / "ablation.yaml", *_stimulus_files(root)]
    rows = [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    combined = hashlib.sha256("\n".join(f"{n} {h}" for n, h in rows).encode()).hexdigest()
    return rows, combined


def freeze(root: Path = Path("stimuli")) -> str:
    load_all(root)  # validates
    rows, combined = _digest(root)
    lines = [f"{h}  {n}" for n, h in rows] + [f"{combined}  COMBINED"]
    (root / "CHECKSUMS").write_text("\n".join(lines) + "\n")
    return combined


def current_checksum(root: Path = Path("stimuli")) -> str:
    last = (root / "CHECKSUMS").read_text().strip().splitlines()[-1]
    return last.split()[0]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_stimuli.py -v`
Expected: 6 passed.

- [ ] **Step 6: Freeze the seed stimuli and commit**

Run: `uv run python -c "from gricean.stimuli import freeze; print(freeze())"`
Expected: prints a 64-char hex digest; `stimuli/CHECKSUMS` created.

```bash
git add src/gricean/stimuli.py stimuli tests/test_stimuli.py
git commit -m "feat: stimuli loading, ablation versions, freeze"
```

---

### Task 3: LM protocol, FakeLM, Hugging Face backend, registry

**Files:**
- Create: `src/gricean/lm.py`, `configs/models.yaml`, `tests/conftest.py`, `tests/test_lm.py`

**Interfaces:**
- Produces:
  ```python
  class LM(Protocol):
      name: str
      def format(self, prompt: str, template: bool) -> str
      def generate(self, prompt: str, *, max_new_tokens: int, temperature: float | None, seed: int | None) -> str
      def continuation_logprob(self, prompt: str, continuation: str) -> tuple[float, int]

  class FakeLM:  # deterministic; generate returns f"[{name}] {prompt[:20]}"; logprobs from a rules dict
  class HFLM:    # transformers backend; template modes: "none" | "chat" | "alpaca"
  @dataclass class ModelSpec: name, repo, revision, family, template, adapter (str|None)
  def load_registry(path=Path("configs/models.yaml")) -> dict[str, ModelSpec]
  def load_model(spec: ModelSpec, device: str | None = None) -> HFLM
  ALPACA = "### Instruction:\n{prompt}\n\n### Response:\n"
  ```

- [ ] **Step 1: Write configs/models.yaml**

```yaml
# revision: fill with the exact commit hash from the Hugging Face model page before running.
models:
  olmo_base:
    repo: allenai/OLMo-2-0425-1B
    revision: main
    family: olmo
    template: none
  olmo_sft:
    repo: allenai/OLMo-2-0425-1B-SFT
    revision: main
    family: olmo
    template: chat
  olmo_dpo:
    repo: allenai/OLMo-2-0425-1B-DPO
    revision: main
    family: olmo
    template: chat
  olmo_instruct:
    repo: allenai/OLMo-2-0425-1B-Instruct
    revision: main
    family: olmo
    template: chat
  qwen_base:
    repo: Qwen/Qwen2.5-1.5B
    revision: main
    family: qwen
    template: none
  qwen_instruct:
    repo: Qwen/Qwen2.5-1.5B-Instruct
    revision: main
    family: qwen
    template: chat
  smollm_base:
    repo: HuggingFaceTB/SmolLM2-360M
    revision: main
    family: smollm
    template: alpaca
ladders:
  olmo: [olmo_base, olmo_sft, olmo_dpo, olmo_instruct]
  qwen: [qwen_base, qwen_instruct]
  smollm: [smollm_base]
neither_source:
  olmo: qwen_base
  qwen: olmo_base
  smollm: olmo_base
```

- [ ] **Step 2: Write conftest and failing tests**

`tests/conftest.py`:
```python
import pytest
from gricean.lm import FakeLM

@pytest.fixture
def fake_lm():
    return FakeLM("fake", logprob_rules={"It is noon.": -1.0, "Yes.": -3.0})
```

`tests/test_lm.py`:
```python
from pathlib import Path
import pytest
from gricean import lm as L

def test_fake_generate_is_deterministic(fake_lm):
    a = fake_lm.generate("Hello there", max_new_tokens=5, temperature=0.7, seed=1)
    b = fake_lm.generate("Hello there", max_new_tokens=5, temperature=0.7, seed=1)
    assert a == b and a.startswith("[fake]")

def test_fake_logprob_uses_rules(fake_lm):
    lp, n = fake_lm.continuation_logprob("Can you tell me the time?", "It is noon.")
    assert lp == -1.0 and n == 3
    lp2, _ = fake_lm.continuation_logprob("x", "unknown text")
    assert lp2 == -10.0 * 2

def test_fake_format_template_flag(fake_lm):
    assert fake_lm.format("hi", template=False) == "hi"
    assert fake_lm.format("hi", template=True) == "<user>hi</user><assistant>"

def test_registry_loads_specs_and_ladders():
    reg = L.load_registry(Path("configs/models.yaml"))
    assert reg["olmo_sft"].template == "chat"
    assert reg["olmo_base"].family == "olmo"
    assert L.ladders(Path("configs/models.yaml"))["olmo"] == ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]
    assert L.neither_source(Path("configs/models.yaml"))["olmo"] == "qwen_base"

def test_alpaca_format():
    spec = L.ModelSpec(name="s", repo="r", revision="main", family="smollm", template="alpaca", adapter=None)
    assert L.render_template(spec, "Do X", tokenizer=None) == "### Instruction:\nDo X\n\n### Response:\n"

@pytest.mark.slow
def test_hf_logprob_boundary_space():
    spec = L.ModelSpec(name="tiny", repo="sshleifer/tiny-gpt2", revision="main",
                       family="test", template="none", adapter=None)
    m = L.load_model(spec, device="cpu")
    lp_a, n_a = m.continuation_logprob("The cat", " sat")
    lp_b, n_b = m.continuation_logprob("The cat", "sat")
    assert n_a >= 1 and n_b >= 1
    assert lp_a != lp_b  # tokenized at the true boundary, not by string concatenation
    out = m.generate("The cat", max_new_tokens=3, temperature=None, seed=None)
    assert isinstance(out, str)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_lm.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.lm`.

- [ ] **Step 4: Implement lm.py**

`src/gricean/lm.py`:
```python
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import yaml

ALPACA = "### Instruction:\n{prompt}\n\n### Response:\n"


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
    return {name: ModelSpec(name=name, **{k: v for k, v in m.items()})
            for name, m in _cfg(path)["models"].items()}


def ladders(path: Path = Path("configs/models.yaml")) -> dict[str, list[str]]:
    return _cfg(path)["ladders"]


def neither_source(path: Path = Path("configs/models.yaml")) -> dict[str, str]:
    return _cfg(path)["neither_source"]


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
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)
        if temperature is None:
            kwargs.update(do_sample=False)
        else:
            kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
        with torch.no_grad():
            out = self.model.generate(ids, **kwargs)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def continuation_logprob(self, prompt: str, continuation: str) -> tuple[float, int]:
        """Tokenize prompt and continuation separately so the boundary matches training."""
        torch = self._torch
        p_ids = self._ids(prompt, special=True)
        c_ids = self._ids(continuation, special=False)
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
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(spec.repo, revision=spec.revision, torch_dtype=dtype)
    if spec.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec.adapter)
    model.to(device).eval()
    return HFLM(spec, model, tok, device)


def spec_stamp(spec: ModelSpec) -> str:
    """Stable identifier of a model configuration, for cache stamps."""
    return hashlib.sha256(
        f"{spec.repo}@{spec.revision}|{spec.template}|{spec.adapter}".encode()).hexdigest()[:16]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_lm.py -v`
Expected: 5 passed, 1 deselected (slow).

Run once with network: `uv run pytest tests/test_lm.py -v -m slow`
Expected: `test_hf_logprob_boundary_space` PASS (downloads a tiny model).

- [ ] **Step 6: Commit**

```bash
git add src/gricean/lm.py configs/models.yaml tests/conftest.py tests/test_lm.py
git commit -m "feat: LM protocol with FakeLM and HF backend; model registry"
```

---

### Task 4: JSONL I/O and per-item idempotency stamps

**Files:**
- Create: `src/gricean/io.py`, `tests/test_io.py`

**Interfaces:**
- Produces:
  ```python
  def read_jsonl(path: Path) -> list[dict]
  def write_jsonl(path: Path, rows: list[dict]) -> None        # atomic, sorted by (item_id, version, mode, seed)
  def item_stamp(*parts: str) -> str                            # sha256 of joined parts, 16 hex
  def merge_fresh(path: Path, key_fields: tuple[str, ...], stamp_field: str,
                  wanted: dict[tuple, str]) -> tuple[list[dict], list[tuple]]
      # returns (rows to keep unchanged, keys that must be recomputed)
  ```

- [ ] **Step 1: Write the failing tests**

`tests/test_io.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.io`.

- [ ] **Step 3: Implement io.py**

`src/gricean/io.py`:
```python
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

SORT_FIELDS = ("item_id", "version", "mode", "seed")


def _sort_key(row: dict):
    return tuple("" if row.get(f) is None else str(row.get(f)) for f in SORT_FIELDS)


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in sorted(rows, key=_sort_key):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def item_stamp(*parts: str) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def merge_fresh(path: Path, key_fields: tuple[str, ...], stamp_field: str,
                wanted: dict[tuple, str]) -> tuple[list[dict], list[tuple]]:
    existing = {tuple(r.get(f) for f in key_fields): r for r in read_jsonl(path)}
    keep, todo = [], []
    for key, stamp in wanted.items():
        row = existing.get(key)
        if row is not None and row.get(stamp_field) == stamp:
            keep.append(row)
        else:
            todo.append(key)
    return keep, todo
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_io.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/io.py tests/test_io.py
git commit -m "feat: jsonl io with per-item freshness stamps"
```

---

### Task 5: Generate stage

**Files:**
- Create: `src/gricean/generate.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: `stimuli.load_all`, `stimuli.current_checksum`, `lm.LM`, `lm.spec_stamp`, `io.*`.
- Produces:
  ```python
  def response_path(model_name: str, template: bool, root: Path = Path("data/responses")) -> Path
      # data/responses/{model_name}__{tmpl|raw}.jsonl
  def run_generate(model: LM, spec: ModelSpec, items: list[Item], *, template: bool,
                   gen_cfg: dict, stimuli_checksum: str, root: Path = Path("data/responses")) -> int
      # returns number of rows (re)computed
  ```
  Row schema: `{item_id, category, language, control, version, model, template, mode ("greedy"|"sample"), seed (int|None), prompt_text, response, stamp, stimuli_checksum}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_generate.py`:
```python
from pathlib import Path
from gricean import generate as G
from gricean.io import read_jsonl
from gricean.lm import FakeLM, ModelSpec
from gricean.stimuli import Item

GEN = {"max_new_tokens": 150, "greedy": True, "samples": {"n": 2, "temperature": 0.7, "seeds": [1, 2]}}

def items():
    return [
        Item(id="a", category="indirect_request", language="en", control=False,
             prompt="Can you tell me the time?", literal_reading="", intended_reading="",
             literal_continuation="Yes.", intended_continuation="Noon.",
             versions={"full": "Can you tell me the time?", "stripped": "Can you tell me the time"}),
        Item(id="c", category="indirect_request", language="en", control=True,
             prompt="What time is it?", literal_reading="", intended_reading="",
             literal_continuation=None, intended_continuation=None, versions={"full": "What time is it?"}),
    ]

def spec():
    return ModelSpec(name="fake", repo="r", revision="v1", family="f", template="chat")

def test_generates_all_rows_and_uses_template(tmp_path: Path):
    lm = FakeLM("fake")
    n = G.run_generate(lm, spec(), items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(G.response_path("fake", True, tmp_path))
    # item a: 2 versions x (1 greedy + 2 samples) = 6 ; item c: 1 version x 3 = 3
    assert n == 9 and len(rows) == 9
    assert all(r["prompt_text"].startswith("<user>") for r in rows)
    greedy = [r for r in rows if r["mode"] == "greedy"]
    assert all(r["seed"] is None for r in greedy)
    assert {r["seed"] for r in rows if r["mode"] == "sample"} == {1, 2}

def test_raw_path_does_not_apply_template(tmp_path: Path):
    G.run_generate(FakeLM("fake"), spec(), items(), template=False, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(G.response_path("fake", False, tmp_path))
    assert all(not r["prompt_text"].startswith("<user>") for r in rows)

def test_second_run_is_noop_and_changed_item_recomputed(tmp_path: Path):
    lm = FakeLM("fake")
    G.run_generate(lm, spec(), items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    assert G.run_generate(lm, spec(), items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path) == 0
    changed = items()
    changed[1] = Item(**{**changed[1].__dict__, "versions": {"full": "What time is it now?"}})
    n = G.run_generate(lm, spec(), changed, template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    assert n == 3
    rows = read_jsonl(G.response_path("fake", True, tmp_path))
    assert len(rows) == 9

def test_model_revision_change_recomputes_everything(tmp_path: Path):
    lm = FakeLM("fake")
    G.run_generate(lm, spec(), items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    s2 = ModelSpec(name="fake", repo="r", revision="v2", family="f", template="chat")
    assert G.run_generate(lm, s2, items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path) == 9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.generate`.

- [ ] **Step 3: Implement generate.py**

`src/gricean/generate.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/generate.py tests/test_generate.py
git commit -m "feat: generate stage with idempotent per-item rows"
```

---

### Task 6: Neither-continuation sampling

**Files:**
- Create: `src/gricean/neither.py`, `tests/test_neither.py`

**Interfaces:**
- Consumes: `lm.LM`, `stimuli.Item`, `io.*`.
- Produces:
  ```python
  def neither_path(family: str, root: Path = Path("stimuli/neither")) -> Path   # stimuli/neither/{family}.jsonl
  def token_overlap(a: str, b: str) -> float          # Jaccard over lowercase word sets
  def truncate(text: str, max_tokens: int = 20) -> str  # cut at first newline, then to max_tokens words
  def sample_neither(source: LM, source_spec: ModelSpec, family: str, items: list[Item], *,
                     cfg: dict, stimuli_checksum: str, root: Path = Path("stimuli/neither")) -> int
  def load_neither(family: str, root=...) -> dict[str, str]   # item_id -> chosen continuation
  ```
  Row schema: `{item_id, family, source_model, source_revision, seeds, samples: [str], chosen: str, overlap: float, flag: bool, stamp, stimuli_checksum}`. `flag` is true when the best sample still overlaps ≥ 0.5 with either continuation; the author reviews flagged rows.

- [ ] **Step 1: Write the failing tests**

`tests/test_neither.py`:
```python
from pathlib import Path
from gricean import neither as N
from gricean.io import read_jsonl
from gricean.lm import FakeLM, ModelSpec
from gricean.stimuli import Item

CFG = {"n": 3, "temperature": 1.0, "max_new_tokens": 20, "seeds": [11, 12, 13]}

def item(i="a", intended="Brazil became independent in 1822."):
    return Item(id=i, category="indirect_request", language="en", control=False,
                prompt="Can you tell me what year Brazil became independent?",
                literal_reading="", intended_reading="", literal_continuation="Yes, I can.",
                intended_continuation=intended, versions={"full": "Can you tell me what year Brazil became independent?"})

def test_truncate_cuts_at_newline_then_words():
    assert N.truncate("one two\nthree", 20) == "one two"
    assert N.truncate(" ".join(str(i) for i in range(30)), 5) == "0 1 2 3 4"

def test_overlap_is_jaccard():
    assert N.token_overlap("Yes, I can.", "yes i can") == 1.0
    assert N.token_overlap("apple", "pear") == 0.0

def test_picks_lowest_overlap_and_flags_when_all_overlap(tmp_path: Path):
    class Src(FakeLM):
        def generate(self, prompt, *, max_new_tokens, temperature, seed):
            return {11: "Brazil became independent in 1822.", 12: "Can you tell me what year Chile\nmore",
                    13: "Brazil became independent long ago."}[seed]
    src = Src("qwen_base")
    spec = ModelSpec(name="qwen_base", repo="q", revision="r1", family="qwen", template="none")
    n = N.sample_neither(src, spec, "olmo", [item()], cfg=CFG, stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(N.neither_path("olmo", tmp_path))
    assert n == 1 and len(rows) == 1
    r = rows[0]
    assert r["chosen"] == "Can you tell me what year Chile"
    assert r["flag"] is False
    assert r["source_model"] == "qwen_base" and r["seeds"] == [11, 12, 13]

def test_flags_when_every_sample_answers(tmp_path: Path):
    src = FakeLM("qwen_base", canned={"Can you tell me what year Brazil became independent?": "Brazil became independent in 1822."})
    spec = ModelSpec(name="qwen_base", repo="q", revision="r1", family="qwen", template="none")
    N.sample_neither(src, spec, "olmo", [item()], cfg=CFG, stimuli_checksum="abc", root=tmp_path)
    r = read_jsonl(N.neither_path("olmo", tmp_path))[0]
    assert r["flag"] is True
    assert N.load_neither("olmo", tmp_path) == {"a": "Brazil became independent in 1822."}

def test_controls_are_skipped_and_rerun_is_noop(tmp_path: Path):
    c = Item(id="c", category="indirect_request", language="en", control=True, prompt="x",
             literal_reading="", intended_reading="", literal_continuation=None,
             intended_continuation=None, versions={"full": "x"})
    src = FakeLM("qwen_base")
    spec = ModelSpec(name="qwen_base", repo="q", revision="r1", family="qwen", template="none")
    assert N.sample_neither(src, spec, "olmo", [item(), c], cfg=CFG, stimuli_checksum="abc", root=tmp_path) == 1
    assert N.sample_neither(src, spec, "olmo", [item(), c], cfg=CFG, stimuli_checksum="abc", root=tmp_path) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_neither.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.neither`.

- [ ] **Step 3: Implement neither.py**

`src/gricean/neither.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_neither.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/neither.py tests/test_neither.py
git commit -m "feat: cross-sourced neither continuations with overlap filter"
```

---

### Task 7: Log-prob scoring stage

**Files:**
- Create: `src/gricean/logprobs.py`, `tests/test_logprobs.py`

**Interfaces:**
- Consumes: `lm.LM`, `neither.load_neither`, `stimuli.Item`, `io.*`.
- Produces:
  ```python
  def logprob_path(model_name: str, template: bool, root=Path("data/logprobs")) -> Path
  def three_way(lp: dict[str, tuple[float, int]]) -> dict   # {"norm": {k: lp/n}, "share": {k: softmax over norm}, "log_odds": norm[intended]-norm[literal]}
  def run_logprobs(model: LM, spec: ModelSpec, items: list[Item], neither: dict[str, str], *,
                   template: bool, stimuli_checksum: str, root=Path("data/logprobs")) -> int
  ```
  Row schema: `{item_id, category, language, version, model, template, mode:"logprob", seed:None, lp_literal, lp_intended, lp_neither, n_literal, n_intended, n_neither, norm_literal, norm_intended, norm_neither, share_literal, share_intended, share_neither, log_odds, stamp, stimuli_checksum}`. Controls excluded.

- [ ] **Step 1: Write the failing tests**

`tests/test_logprobs.py`:
```python
import math
from pathlib import Path
import pytest
from gricean import logprobs as LP
from gricean.io import read_jsonl
from gricean.lm import FakeLM, ModelSpec
from gricean.stimuli import Item

def item():
    return Item(id="a", category="indirect_request", language="en", control=False,
                prompt="Can you tell me the time?", literal_reading="", intended_reading="",
                literal_continuation="Yes.", intended_continuation="It is noon.",
                versions={"full": "Can you tell me the time?", "stripped": "Can you tell me the time"})

def control():
    return Item(id="c", category="indirect_request", language="en", control=True, prompt="x",
                literal_reading="", intended_reading="", literal_continuation=None,
                intended_continuation=None, versions={"full": "x"})

def test_three_way_normalizes_and_softmaxes():
    out = LP.three_way({"literal": (-6.0, 2), "intended": (-3.0, 3), "neither": (-4.0, 4)})
    assert out["norm"] == {"literal": -3.0, "intended": -1.0, "neither": -1.0}
    assert out["log_odds"] == 2.0
    s = out["share"]
    assert math.isclose(sum(s.values()), 1.0)
    assert math.isclose(s["intended"], s["neither"]) and s["literal"] < s["intended"]

def test_run_writes_rows_excluding_controls(tmp_path: Path):
    lm = FakeLM("fake", logprob_rules={"Yes.": -3.0, "It is noon.": -1.5, "blah blah": -2.0})
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="chat")
    n = LP.run_logprobs(lm, spec, [item(), control()], {"a": "blah blah"}, template=True,
                        stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(LP.logprob_path("fake", True, tmp_path))
    assert n == 2 and {r["version"] for r in rows} == {"full", "stripped"}
    r = rows[0]
    assert r["lp_intended"] == -1.5 and r["n_intended"] == 3
    assert math.isclose(r["log_odds"], (-1.5 / 3) - (-3.0 / 1))
    assert math.isclose(r["share_literal"] + r["share_intended"] + r["share_neither"], 1.0)

def test_missing_neither_raises(tmp_path: Path):
    lm = FakeLM("fake")
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="none")
    with pytest.raises(KeyError, match="a"):
        LP.run_logprobs(lm, spec, [item()], {}, template=False, stimuli_checksum="abc", root=tmp_path)

def test_rerun_noop(tmp_path: Path):
    lm = FakeLM("fake")
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="none")
    LP.run_logprobs(lm, spec, [item()], {"a": "x"}, template=False, stimuli_checksum="abc", root=tmp_path)
    assert LP.run_logprobs(lm, spec, [item()], {"a": "x"}, template=False, stimuli_checksum="abc", root=tmp_path) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logprobs.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.logprobs`.

- [ ] **Step 3: Implement logprobs.py**

`src/gricean/logprobs.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_logprobs.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/logprobs.py tests/test_logprobs.py
git commit -m "feat: three-way continuation scoring stage"
```

---

### Task 8: Coding sheet export/load and kappa

**Files:**
- Create: `src/gricean/coding.py`, `tests/test_coding.py`

**Interfaces:**
- Consumes: `io.read_jsonl`, `generate.response_path`.
- Produces:
  ```python
  INTENT = ("literal", "intended", "neither"); CORRECT = ("correct", "incorrect", "na")
  def export_sheet(response_root: Path, out: Path, *, coder: str, sample_frac: float | None, seed: int) -> int
      # one CSV row per response; columns: row_id, item_id, category, language, control, version, model, template,
      # mode, seed, prompt_text, response, intent, correctness. Blind: model/template columns are moved to a
      # separate key file out.with_suffix(".key.csv") when sample_frac is given (coder 2).
  def load_sheet(path: Path, key: Path | None = None) -> pandas.DataFrame   # raises ValueError on blank/invalid cells
  def cohen_kappa(a: list[str], b: list[str]) -> float
  ```

- [ ] **Step 1: Write the failing tests**

`tests/test_coding.py`:
```python
from pathlib import Path
import math
import pandas as pd
import pytest
from gricean import coding as C
from gricean.io import write_jsonl

def make_responses(root: Path):
    rows = []
    for model in ("m_base", "m_sft"):
        for i in range(4):
            rows.append({"item_id": f"a{i}", "category": "indirect_request", "language": "en",
                         "control": False, "version": "full", "model": model, "template": True,
                         "mode": "greedy", "seed": None, "prompt_text": "p", "response": f"r{i}{model}",
                         "stamp": "s", "stimuli_checksum": "c"})
        write_jsonl(root / f"{model}__tmpl.jsonl", rows[-4:])

def test_export_full_sheet_has_all_rows_and_blank_rubric(tmp_path: Path):
    make_responses(tmp_path / "resp")
    n = C.export_sheet(tmp_path / "resp", tmp_path / "coder1.csv", coder="coder1", sample_frac=None, seed=0)
    df = pd.read_csv(tmp_path / "coder1.csv")
    assert n == 8 and len(df) == 8
    assert df["intent"].isna().all() and "model" in df.columns

def test_export_blind_sample_hides_model(tmp_path: Path):
    make_responses(tmp_path / "resp")
    n = C.export_sheet(tmp_path / "resp", tmp_path / "coder2.csv", coder="coder2", sample_frac=0.5, seed=0)
    df = pd.read_csv(tmp_path / "coder2.csv")
    key = pd.read_csv(tmp_path / "coder2.key.csv")
    assert n == 4 and "model" not in df.columns and set(key.columns) >= {"row_id", "model", "template"}

def test_load_rejects_blank_or_invalid(tmp_path: Path):
    make_responses(tmp_path / "resp")
    C.export_sheet(tmp_path / "resp", tmp_path / "c.csv", coder="c", sample_frac=None, seed=0)
    df = pd.read_csv(tmp_path / "c.csv")
    df["intent"] = "intended"; df["correctness"] = "correct"
    df.loc[0, "intent"] = "intendd"
    df.to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="row_id"):
        C.load_sheet(tmp_path / "c.csv")
    df.loc[0, "intent"] = None
    df.to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="blank"):
        C.load_sheet(tmp_path / "c.csv")

def test_load_merges_key_back(tmp_path: Path):
    make_responses(tmp_path / "resp")
    C.export_sheet(tmp_path / "resp", tmp_path / "c2.csv", coder="c2", sample_frac=0.5, seed=0)
    df = pd.read_csv(tmp_path / "c2.csv")
    df["intent"] = "neither"; df["correctness"] = "na"
    df.to_csv(tmp_path / "c2.csv", index=False)
    out = C.load_sheet(tmp_path / "c2.csv", key=tmp_path / "c2.key.csv")
    assert "model" in out.columns and len(out) == 4

def test_cohen_kappa_known_value():
    a = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["n"] * 10 + ["y"] * 15
    # po = 25/50 = 0.5 ; pa(y)=30/50, pb(y)=35/50 ; pe = 0.6*0.7 + 0.4*0.3 = 0.54
    assert math.isclose(C.cohen_kappa(a, b), (0.5 - 0.54) / (1 - 0.54))
    assert C.cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coding.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.coding`.

- [ ] **Step 3: Implement coding.py**

`src/gricean/coding.py`:
```python
from __future__ import annotations
from collections import Counter
from pathlib import Path
import pandas as pd
from .io import read_jsonl

INTENT = ("literal", "intended", "neither")
CORRECT = ("correct", "incorrect", "na")
PUBLIC = ["row_id", "item_id", "category", "language", "control", "version", "mode", "seed",
          "prompt_text", "response"]
HIDDEN = ["model", "template"]


def _all_responses(response_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(response_root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    df["row_id"] = [f"{r.model}|{int(r.template)}|{r.item_id}|{r.version}|{r.mode}|{r.seed}"
                    for r in df.itertuples()]
    return df


def export_sheet(response_root: Path, out: Path, *, coder: str, sample_frac: float | None,
                 seed: int) -> int:
    df = _all_responses(response_root)
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=seed).sort_values("row_id")
        df[["row_id", *HIDDEN]].to_csv(Path(out).with_suffix(".key.csv"), index=False)
        cols = PUBLIC
    else:
        cols = PUBLIC + HIDDEN
    sheet = df[cols].copy()
    sheet["coder"] = coder
    sheet["intent"] = pd.NA
    sheet["correctness"] = pd.NA
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False)
    return len(sheet)


def load_sheet(path: Path, key: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col, allowed in (("intent", INTENT), ("correctness", CORRECT)):
        blank = df[df[col].str.strip() == ""]
        if len(blank):
            raise ValueError(f"blank {col} in row_id(s): {blank['row_id'].tolist()[:5]}")
        bad = df[~df[col].str.strip().str.lower().isin(allowed)]
        if len(bad):
            raise ValueError(f"invalid {col} {bad[col].tolist()[:5]} at row_id(s) {bad['row_id'].tolist()[:5]}")
        df[col] = df[col].str.strip().str.lower()
    if key is not None:
        k = pd.read_csv(key, dtype=str)
        df = df.merge(k, on="row_id", how="left")
    df["control"] = df["control"].str.lower() == "true"
    if "template" in df.columns:
        df["template"] = df["template"].astype(str).str.lower() == "true"
    return df


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coding.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/coding.py tests/test_coding.py
git commit -m "feat: coding sheet export/load and Cohen's kappa"
```

---

### Task 9: Report stage (tables, figure, printed examples)

**Files:**
- Create: `src/gricean/report.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `coding.load_sheet`, `coding.cohen_kappa`, `io.read_jsonl`, `lm.ladders`.
- Produces:
  ```python
  def intent_table(coded: pd.DataFrame, models: list[str]) -> pd.DataFrame
      # index (category, version), columns MultiIndex (model, intent) -> rate in [0,1]; only template=True rows for tuned, raw for base
  def logodds_table(lp_root: Path, models: list[str]) -> pd.DataFrame
      # index (category, version), columns (model): mean log_odds, ci_low, ci_high (bootstrap 1000, seed 0), share_neither
  def examples(resp_root: Path, models: list[str], item_ids: list[str]) -> str   # markdown block, greedy full-version per model
  def ladder_figure(intent: pd.DataFrame, logodds: pd.DataFrame, models: list[str], out: Path) -> None
  def build(coded_path: Path, key_path: Path | None, coder2_path: Path | None, resp_root, lp_root, models, out_dir) -> dict
      # writes out_dir/intent.csv, intent.tex, logodds.csv, logodds.tex, ladder.png, examples.md, kappa.txt; returns summary
  ```

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
from pathlib import Path
import pandas as pd
from gricean import report as R
from gricean.io import write_jsonl

MODELS = ["m_base", "m_sft"]

def coded():
    rows = []
    for model, intents in (("m_base", ["neither", "literal", "literal", "intended"]),
                           ("m_sft", ["intended"] * 4)):
        for i, it in enumerate(intents):
            rows.append({"row_id": f"{model}|1|a{i}|full|greedy|", "item_id": f"a{i}",
                         "category": "indirect_request", "language": "en", "control": False,
                         "version": "full", "mode": "greedy", "seed": "", "prompt_text": "p",
                         "response": "r", "model": model, "template": True, "coder": "c1",
                         "intent": it, "correctness": "na"})
    return pd.DataFrame(rows)

def lp_rows(root: Path):
    for model, lo in (("m_base", [-1.0, -0.5]), ("m_sft", [2.0, 3.0])):
        write_jsonl(root / f"{model}__tmpl.jsonl", [
            {"item_id": f"a{i}", "category": "indirect_request", "language": "en", "version": "full",
             "model": model, "template": True, "mode": "logprob", "seed": None, "log_odds": v,
             "share_neither": 0.5, "share_literal": 0.25, "share_intended": 0.25, "stamp": "s",
             "stimuli_checksum": "c"} for i, v in enumerate(lo)])

def test_intent_table_rates():
    t = R.intent_table(coded(), MODELS)
    assert t.loc[("indirect_request", "full"), ("m_base", "intended")] == 0.25
    assert t.loc[("indirect_request", "full"), ("m_sft", "intended")] == 1.0
    assert t.loc[("indirect_request", "full"), ("m_base", "neither")] == 0.25

def test_logodds_table_means_and_ci(tmp_path: Path):
    lp_rows(tmp_path)
    t = R.logodds_table(tmp_path, MODELS)
    assert t.loc[("indirect_request", "full"), ("m_sft", "mean")] == 2.5
    assert t.loc[("indirect_request", "full"), ("m_sft", "ci_low")] <= 2.5 <= t.loc[("indirect_request", "full"), ("m_sft", "ci_high")]

def test_examples_markdown(tmp_path: Path):
    resp = tmp_path / "resp"
    for model in MODELS:
        write_jsonl(resp / f"{model}__tmpl.jsonl", [{"item_id": "a0", "category": "indirect_request",
            "language": "en", "control": False, "version": "full", "model": model, "template": True,
            "mode": "greedy", "seed": None, "prompt_text": "Can you tell me the time?",
            "response": f"answer from {model}", "stamp": "s", "stimuli_checksum": "c"}])
    md = R.examples(resp, MODELS, ["a0"])
    assert "answer from m_base" in md and "answer from m_sft" in md and "Can you tell me the time?" in md

def test_build_writes_all_outputs(tmp_path: Path):
    coded().to_csv(tmp_path / "coded.csv", index=False)
    lp_rows(tmp_path / "lp")
    resp = tmp_path / "resp"
    for model in MODELS:
        write_jsonl(resp / f"{model}__tmpl.jsonl", [{"item_id": "a0", "category": "indirect_request",
            "language": "en", "control": False, "version": "full", "model": model, "template": True,
            "mode": "greedy", "seed": None, "prompt_text": "p", "response": "r", "stamp": "s", "stimuli_checksum": "c"}])
    summary = R.build(tmp_path / "coded.csv", None, None, resp, tmp_path / "lp", MODELS, tmp_path / "paper")
    for name in ("intent.csv", "intent.tex", "logodds.csv", "logodds.tex", "ladder.png", "examples.md"):
        assert (tmp_path / "paper" / name).exists()
    assert summary["n_coded"] == 8 and summary["kappa"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.report`.

- [ ] **Step 3: Implement report.py**

`src/gricean/report.py`:
```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .coding import INTENT, cohen_kappa, load_sheet
from .io import read_jsonl

INDEX = ["category", "version"]


def intent_table(coded: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    df = coded[coded["model"].isin(models) & ~coded["control"]]
    counts = df.groupby(INDEX + ["model", "intent"]).size()
    totals = df.groupby(INDEX + ["model"]).size()
    rates = (counts / totals).rename("rate").reset_index()
    table = rates.pivot_table(index=INDEX, columns=["model", "intent"], values="rate", fill_value=0.0)
    cols = pd.MultiIndex.from_product([models, list(INTENT)])
    return table.reindex(columns=cols, fill_value=0.0)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def logodds_table(lp_root: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(lp_root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    df = df[df["model"].isin(models)]
    out = {}
    for (cat, ver, model), g in df.groupby(INDEX + ["model"]):
        x = g["log_odds"].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(x)
        out[(cat, ver)] = out.get((cat, ver), {})
        out[(cat, ver)].update({(model, "mean"): float(x.mean()), (model, "ci_low"): lo,
                                (model, "ci_high"): hi, (model, "share_neither"): float(g["share_neither"].mean())})
    table = pd.DataFrame.from_dict(out, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=INDEX)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    cols = pd.MultiIndex.from_product([models, ["mean", "ci_low", "ci_high", "share_neither"]])
    return table.reindex(columns=cols)


def examples(resp_root: Path, models: list[str], item_ids: list[str]) -> str:
    rows = []
    for path in sorted(Path(resp_root).glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    df = df[(df["mode"] == "greedy") & (df["version"] == "full") & df["model"].isin(models)]
    blocks = []
    for item_id in item_ids:
        sub = df[df["item_id"] == item_id]
        if sub.empty:
            continue
        prompt = sub.iloc[0]["prompt_text"]
        blocks.append(f"### {item_id}\n\n**Prompt:** {prompt}\n")
        for model in models:
            r = sub[sub["model"] == model]
            if not r.empty:
                blocks.append(f"**{model}:** {r.iloc[0]['response'].strip()}\n")
    return "\n".join(blocks)


def ladder_figure(intent: pd.DataFrame, logodds: pd.DataFrame, models: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    full_i = intent.xs("full", level="version")
    full_l = logodds.xs("full", level="version")
    for cat in full_i.index:
        axes[0].plot(models, [full_i.loc[cat, (m, "intended")] for m in models], marker="o", label=cat)
    for cat in full_l.index:
        axes[1].plot(models, [full_l.loc[cat, (m, "mean")] for m in models], marker="o", label=cat)
    axes[0].set_ylabel("rate of intended responses"); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("log-odds intended vs literal (per token)")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build(coded_path: Path, key_path: Path | None, coder2_path: Path | None, resp_root: Path,
          lp_root: Path, models: list[str], out_dir: Path, example_ids: list[str] | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coded = load_sheet(coded_path)
    intent = intent_table(coded, models)
    lo = logodds_table(lp_root, models)
    intent.to_csv(out_dir / "intent.csv"); (out_dir / "intent.tex").write_text(intent.to_latex(float_format="%.2f"))
    lo.to_csv(out_dir / "logodds.csv"); (out_dir / "logodds.tex").write_text(lo.to_latex(float_format="%.2f"))
    ladder_figure(intent, lo, models, out_dir / "ladder.png")
    ids = example_ids or sorted(coded["item_id"].unique())[:6]
    (out_dir / "examples.md").write_text(examples(resp_root, models, ids))
    kappa = None
    if coder2_path is not None:
        c2 = load_sheet(coder2_path, key=key_path)
        merged = coded.merge(c2[["row_id", "intent"]], on="row_id", suffixes=("", "_c2"))
        kappa = cohen_kappa(merged["intent"].tolist(), merged["intent_c2"].tolist())
        (out_dir / "kappa.txt").write_text(f"cohen_kappa_intent={kappa:.3f} n={len(merged)}\n")
    return {"n_coded": int(len(coded)), "kappa": kappa, "models": models}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/report.py tests/test_report.py
git commit -m "feat: report stage with tables, ladder figure and printed examples"
```

---

### Task 10: CLI wiring

**Files:**
- Create: `src/gricean/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: every stage module.
- Produces: `gricean freeze | neither --family F | generate --model M [--raw] | generate --all | logprobs --model M [--raw] | logprobs --all | sheet [--coder2 --frac 0.25] | report --ladder olmo`.
  Internal helper `run_stage(args, loader)` takes an injectable `loader(spec) -> LM` so tests use `FakeLM`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
from pathlib import Path
import shutil
import yaml
from gricean import cli
from gricean.lm import FakeLM

def project(tmp_path: Path) -> Path:
    shutil.copytree("stimuli", tmp_path / "stimuli", ignore=shutil.ignore_patterns("neither", "CHECKSUMS"))
    (tmp_path / "configs").mkdir()
    shutil.copy("configs/generation.yaml", tmp_path / "configs/generation.yaml")
    yaml.safe_dump({
        "models": {"b": {"repo": "x", "revision": "1", "family": "t", "template": "none"},
                   "s": {"repo": "y", "revision": "1", "family": "t", "template": "chat"},
                   "q": {"repo": "z", "revision": "1", "family": "u", "template": "none"}},
        "ladders": {"t": ["b", "s"], "u": ["q"]},
        "neither_source": {"t": "q", "u": "b"},
    }, (tmp_path / "configs/models.yaml").open("w"))
    return tmp_path

def fake_loader(spec):
    return FakeLM(spec.name, logprob_rules={})

def test_full_pipeline_with_fakes(tmp_path: Path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    assert cli.main(["freeze"]) == 0
    assert (root / "stimuli/CHECKSUMS").exists()
    assert cli.main(["neither", "--family", "t", "--family", "u"], loader=fake_loader) == 0
    assert (root / "stimuli/neither/t.jsonl").exists()
    assert (root / "stimuli/neither/u.jsonl").exists()
    assert cli.main(["generate", "--all"], loader=fake_loader) == 0
    assert (root / "data/responses/b__raw.jsonl").exists()
    assert (root / "data/responses/s__tmpl.jsonl").exists()
    assert (root / "data/responses/s__raw.jsonl").exists()     # tuned models also run raw
    assert not (root / "data/responses/b__tmpl.jsonl").exists()
    assert cli.main(["logprobs", "--all"], loader=fake_loader) == 0
    assert (root / "data/logprobs/s__tmpl.jsonl").exists()
    assert cli.main(["sheet"]) == 0
    assert (root / "data/coding/coder1.csv").exists()
    assert cli.main(["sheet", "--coder2", "--frac", "0.5"]) == 0
    assert (root / "data/coding/coder2.key.csv").exists()

def test_unknown_stage_returns_2(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(project(tmp_path))
    assert cli.main(["nope"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.cli`.

- [ ] **Step 3: Implement cli.py**

`src/gricean/cli.py`:
```python
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml
from . import coding, generate, logprobs, neither, report, stimuli
from .lm import ladders, load_model, load_registry, neither_source

CONFIGS = Path("configs")


def _gen_cfg() -> dict:
    return yaml.safe_load((CONFIGS / "generation.yaml").read_text())


def _models_for(args, reg) -> list[tuple[str, bool]]:
    """(model_name, template) pairs to run. Tuned models run templated and raw; base only raw."""
    names = list(reg) if args.all else [args.model]
    pairs = []
    for n in names:
        spec = reg[n]
        if spec.template == "none":
            pairs.append((n, False))
        elif args.all:
            pairs += [(n, True), (n, False)]
        else:
            pairs.append((n, not args.raw))
    return pairs


def cmd_freeze(args, loader) -> int:
    print(stimuli.freeze())
    return 0


def cmd_neither(args, loader) -> int:
    reg, items, cfg = load_registry(), stimuli.load_all(), _gen_cfg()["neither"]
    chk = stimuli.current_checksum()
    for family in args.family:
        src = reg[neither_source()[family]]
        n = neither.sample_neither(loader(src), src, family, items, cfg=cfg, stimuli_checksum=chk)
        print(f"neither[{family}] from {src.name}: {n} rows")
    return 0


def cmd_generate(args, loader) -> int:
    reg, items, cfg = load_registry(), stimuli.load_all(), _gen_cfg()
    chk = stimuli.current_checksum()
    for name, template in _models_for(args, reg):
        spec = reg[name]
        n = generate.run_generate(loader(spec), spec, items, template=template, gen_cfg=cfg, stimuli_checksum=chk)
        print(f"generate[{name}, template={template}]: {n} rows")
    return 0


def cmd_logprobs(args, loader) -> int:
    reg, items = load_registry(), stimuli.load_all()
    chk = stimuli.current_checksum()
    for name, template in _models_for(args, reg):
        spec = reg[name]
        nei = neither.load_neither(spec.family)
        n = logprobs.run_logprobs(loader(spec), spec, items, nei, template=template, stimuli_checksum=chk)
        print(f"logprobs[{name}, template={template}]: {n} rows")
    return 0


def cmd_sheet(args, loader) -> int:
    out = Path("data/coding") / ("coder2.csv" if args.coder2 else "coder1.csv")
    n = coding.export_sheet(Path("data/responses"), out, coder="coder2" if args.coder2 else "coder1",
                            sample_frac=args.frac if args.coder2 else None, seed=0)
    print(f"sheet {out}: {n} rows")
    return 0


def cmd_report(args, loader) -> int:
    models = ladders()[args.ladder]
    c2 = Path("data/coding/coder2.csv")
    summary = report.build(Path("data/coding/coder1.csv"), Path("data/coding/coder2.key.csv") if c2.exists() else None,
                           c2 if c2.exists() else None, Path("data/responses"), Path("data/logprobs"),
                           models, Path("paper") / args.ladder)
    print(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gricean")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    n = sub.add_parser("neither"); n.add_argument("--family", action="append", required=True); n.set_defaults(fn=cmd_neither)
    for name, fn in (("generate", cmd_generate), ("logprobs", cmd_logprobs)):
        s = sub.add_parser(name)
        s.add_argument("--model"); s.add_argument("--all", action="store_true"); s.add_argument("--raw", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("sheet"); s.add_argument("--coder2", action="store_true"); s.add_argument("--frac", type=float, default=0.25)
    s.set_defaults(fn=cmd_sheet)
    r = sub.add_parser("report"); r.add_argument("--ladder", default="olmo"); r.set_defaults(fn=cmd_report)
    return p


def main(argv: list[str] | None = None, loader=load_model) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:          # argparse rejects unknown subcommands with code 2
        return int(e.code or 0)
    if not getattr(args, "fn", None):
        build_parser().print_help()
        return 2
    return args.fn(args, loader)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the whole suite and commit**

Run: `uv run pytest -v`
Expected: all passed, 1 deselected.

```bash
git add src/gricean/cli.py tests/test_cli.py
git commit -m "feat: cli wiring for all pipeline stages"
```

---

### Task 11: Marimo notebooks 01, 02, 03, 05

**Files:**
- Create: `notebooks/01_stimuli.py`, `notebooks/02_responses.py`, `notebooks/03_logodds.py`, `notebooks/05_coding.py`, `tests/test_notebooks.py`

**Interfaces:**
- Consumes: package functions only. Rule: no logic in cells beyond calling package functions and rendering.

- [ ] **Step 1: Write the notebook-import test**

`tests/test_notebooks.py`:
```python
import importlib.util
from pathlib import Path
import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("*.py"))

@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_imports_and_is_marimo_app(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "app")
    src = path.read_text()
    assert "def " not in src.replace("def _(", "").replace("def __(", ""), \
        "notebooks must not define helper functions; put logic in the package"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_notebooks.py -v`
Expected: no tests collected (empty glob) or FAIL; either way proceed.

- [ ] **Step 3: Write the four notebooks**

`notebooks/01_stimuli.py`:
```python
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    from gricean.stimuli import load_all, current_checksum, CATEGORIES
    return CATEGORIES, Path, current_checksum, load_all, mo, pd


@app.cell
def _(current_checksum, load_all, mo, pd):
    items = load_all()
    checksum = current_checksum()
    df = pd.DataFrame([{"id": i.id, "category": i.category, "language": i.language, "control": i.control,
                        "n_versions": len(i.versions), "prompt": i.prompt} for i in items])
    mo.md(f"**Frozen stimuli** · checksum `{checksum[:12]}…` · {len(items)} items")
    return checksum, df, items


@app.cell
def _(CATEGORIES, mo):
    category = mo.ui.dropdown(list(CATEGORIES), value=CATEGORIES[0], label="category")
    category
    return (category,)


@app.cell
def _(category, df, mo):
    mo.ui.table(df[df["category"] == category.value], selection=None)
    return


@app.cell
def _(category, items, mo):
    rows = []
    for it in items:
        if it.category == category.value:
            for v, text in it.versions.items():
                rows.append(f"| {it.id} | {v} | {text} |")
    mo.md("| id | version | prompt |\n|---|---|---|\n" + "\n".join(rows))
    return


if __name__ == "__main__":
    app.run()
```

`notebooks/02_responses.py`:
```python
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    from gricean.io import read_jsonl
    from gricean.lm import ladders, load_registry
    from gricean.generate import response_path
    from gricean import cli
    return Path, cli, ladders, load_registry, mo, pd, read_jsonl, response_path


@app.cell
def _(ladders, mo):
    ladder = mo.ui.dropdown(list(ladders()), value="olmo", label="ladder")
    ladder
    return (ladder,)


@app.cell
def _(ladder, ladders, mo, pd, read_jsonl, response_path):
    models = ladders()[ladder.value]
    frames = []
    for m in models:
        for tmpl in (True, False):
            frames.extend(read_jsonl(response_path(m, tmpl)))
    resp = pd.DataFrame(frames)
    mo.md(f"{len(resp)} response rows on disk for ladder **{ladder.value}**" if len(resp)
          else "No responses yet. Use the run button below.")
    return models, resp


@app.cell
def _(mo, resp):
    item = mo.ui.dropdown(sorted(resp["item_id"].unique()) if len(resp) else [], label="item")
    version = mo.ui.dropdown(["full", "stripped", "embedded"], value="full", label="version")
    mo.hstack([item, version])
    return item, version


@app.cell
def _(item, mo, models, resp, version):
    if len(resp) and item.value:
        sub = resp[(resp["item_id"] == item.value) & (resp["version"] == version.value) & (resp["mode"] == "greedy")]
        cols = []
        for m in models:
            for tmpl in (True, False):
                r = sub[(sub["model"] == m) & (sub["template"] == tmpl)]
                if len(r):
                    cols.append(mo.md(f"**{m}** ({'template' if tmpl else 'raw'})\n\n{r.iloc[0]['response']}"))
        out = mo.hstack(cols, wrap=True)
    else:
        out = mo.md("")
    out
    return


@app.cell
def _(mo, models):
    run_model = mo.ui.dropdown(models, label="model to generate")
    run = mo.ui.run_button(label="run generation (slow)")
    mo.hstack([run_model, run])
    return run, run_model


@app.cell
def _(cli, mo, run, run_model):
    mo.stop(not run.value)
    rc = cli.main(["generate", "--model", run_model.value, "--raw"]) if run_model.value else 1
    mo.md(f"generate exit code {rc}; re-open the item dropdown to refresh.")
    return


if __name__ == "__main__":
    app.run()
```

`notebooks/03_logodds.py`:
```python
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from gricean.lm import ladders
    from gricean.report import logodds_table, ladder_figure, intent_table
    from gricean.coding import load_sheet
    return Path, intent_table, ladder_figure, ladders, load_sheet, logodds_table, mo


@app.cell
def _(ladders, mo):
    ladder = mo.ui.dropdown(list(ladders()), value="olmo", label="ladder")
    ladder
    return (ladder,)


@app.cell
def _(Path, ladder, ladders, logodds_table, mo):
    models = ladders()[ladder.value]
    try:
        lo = logodds_table(Path("data/logprobs"), models)
        out = mo.ui.table(lo.round(3).reset_index())
    except Exception as e:  # no data yet
        lo = None
        out = mo.md(f"No log-prob data yet: {e}")
    out
    return lo, models


@app.cell
def _(Path, intent_table, ladder_figure, lo, load_sheet, mo, models):
    coded_path = Path("data/coding/coder1.csv")
    if lo is not None and coded_path.exists():
        intent = intent_table(load_sheet(coded_path), models)
        fig_path = Path(".marimo") / "ladder_preview.png"
        ladder_figure(intent, lo, models, fig_path)
        out = mo.image(str(fig_path))
    else:
        out = mo.md("Ladder figure appears once both log-probs and the coded sheet exist.")
    out
    return


if __name__ == "__main__":
    app.run()
```

`notebooks/05_coding.py`:
```python
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from gricean.lm import ladders
    from gricean.coding import load_sheet, cohen_kappa
    from gricean.report import build
    return Path, build, cohen_kappa, ladders, load_sheet, mo


@app.cell
def _(Path, cohen_kappa, load_sheet, mo):
    c1, c2, key = Path("data/coding/coder1.csv"), Path("data/coding/coder2.csv"), Path("data/coding/coder2.key.csv")
    if c1.exists() and c2.exists():
        a = load_sheet(c1); b = load_sheet(c2, key=key)
        m = a.merge(b[["row_id", "intent"]], on="row_id", suffixes=("", "_c2"))
        out = mo.md(f"Cohen's kappa (intent), n={len(m)}: **{cohen_kappa(m['intent'].tolist(), m['intent_c2'].tolist()):.3f}**")
    else:
        out = mo.md("Both coder sheets are needed for kappa.")
    out
    return


@app.cell
def _(ladders, mo):
    ladder = mo.ui.dropdown(list(ladders()), value="olmo", label="ladder")
    go = mo.ui.run_button(label="build paper/ tables")
    mo.hstack([ladder, go])
    return go, ladder


@app.cell
def _(Path, build, go, ladder, ladders, mo):
    mo.stop(not go.value)
    c2 = Path("data/coding/coder2.csv")
    summary = build(Path("data/coding/coder1.csv"), Path("data/coding/coder2.key.csv") if c2.exists() else None,
                    c2 if c2.exists() else None, Path("data/responses"), Path("data/logprobs"),
                    ladders()[ladder.value], Path("paper") / ladder.value)
    mo.md(f"`{summary}`")
    return


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 4: Run the notebook test and open one notebook**

Run: `uv run pytest tests/test_notebooks.py -v`
Expected: 4 passed.

Run: `uv run marimo run notebooks/01_stimuli.py --headless --port 2718 &` then `sleep 5 && curl -s -o /dev/null -w "%{http_code}" http://localhost:2718 && kill %1`
Expected: `200`.

- [ ] **Step 5: Commit**

```bash
git add notebooks tests/test_notebooks.py
git commit -m "feat: marimo viewer notebooks for stimuli, responses, log-odds and coding"
```

---

### Task 12: Fine-tune data preparation and LoRA training

**Files:**
- Create: `src/gricean/finetune.py`, `configs/finetune.yaml`, `tests/test_finetune.py`
- Modify: `src/gricean/cli.py` (add `finetune` subcommand), `configs/models.yaml` (adapter entries)

**Interfaces:**
- Consumes: `lm.ALPACA`.
- Produces:
  ```python
  DOLLY_KEEP = ("open_qa", "general_qa", "brainstorming", "classification", "creative_writing")
  def prepare(records: list[dict], *, sizes: list[int], seed: int, shuffle_responses: bool) -> dict[str, list[dict]]
      # returns {"full": rows, "n500": rows, "n2000": rows, "format_only": rows} ; each row {"text": ALPACA + response + eos_placeholder}
  def load_dolly() -> list[dict]                                   # datasets.load_dataset, filtered to DOLLY_KEEP
  def train_lora(base_spec: ModelSpec, rows: list[dict], out_dir: Path, cfg: dict, device: str | None = None) -> Path
  ```
  Adapter dirs: `adapters/smollm_full`, `adapters/smollm_n500`, `adapters/smollm_n2000`, `adapters/smollm_format_only`. Each registered in `configs/models.yaml` as `smollm_<name>` with `adapter:` path and `template: alpaca`, and ladder `smollm: [smollm_base, smollm_n500, smollm_n2000, smollm_full, smollm_format_only]`.

- [ ] **Step 1: Write configs/finetune.yaml**

```yaml
base: smollm_base
seed: 0
sizes: [500, 2000]        # plus "full" (all kept rows) and "format_only" (full, responses shuffled)
max_seq_len: 512
epochs: 2
batch_size: 8
grad_accum: 2
lr: 2.0e-4
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_finetune.py`:
```python
from gricean import finetune as F
from gricean.lm import ALPACA

def records():
    return [{"instruction": f"q{i}", "response": f"a{i}", "context": "", "category": c}
            for i, c in enumerate(["open_qa", "closed_qa", "brainstorming", "summarization", "general_qa", "classification"] * 3)]

def test_filter_keeps_only_plain_categories():
    kept = F.filter_dolly(records())
    assert {r["category"] for r in kept} <= set(F.DOLLY_KEEP)
    assert len(kept) == 12

def test_prepare_subsets_and_format_only():
    kept = F.filter_dolly(records())
    out = F.prepare(kept, sizes=[4, 8], seed=0, shuffle_responses=True)
    assert set(out) == {"full", "n4", "n8", "format_only"}
    assert len(out["n4"]) == 4 and len(out["n8"]) == 8 and len(out["full"]) == 12
    assert out["n4"] == out["n8"][:4]                      # nested subsets
    assert out["full"][0]["text"].startswith(ALPACA.format(prompt=out["full"][0]["instruction"]))
    pairs_full = {(r["instruction"], r["response"]) for r in out["full"]}
    pairs_fo = {(r["instruction"], r["response"]) for r in out["format_only"]}
    assert pairs_full != pairs_fo and len(pairs_fo) == 12
    assert {r["response"] for r in out["format_only"]} == {r["response"] for r in out["full"]}

def test_prepare_is_deterministic():
    kept = F.filter_dolly(records())
    assert F.prepare(kept, sizes=[4], seed=1, shuffle_responses=True) == F.prepare(kept, sizes=[4], seed=1, shuffle_responses=True)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_finetune.py -v`
Expected: FAIL with `ModuleNotFoundError: gricean.finetune`.

- [ ] **Step 4: Implement finetune.py**

`src/gricean/finetune.py`:
```python
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
    model = AutoModelForCausalLM.from_pretrained(base_spec.repo, revision=base_spec.revision,
                                                 torch_dtype=torch.float32)
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_finetune.py -v`
Expected: 3 passed.

- [ ] **Step 6: Add the `finetune` CLI subcommand**

In `src/gricean/cli.py`, add after `cmd_report`:
```python
def cmd_finetune(args, loader) -> int:
    from . import finetune
    cfg = yaml.safe_load((CONFIGS / "finetune.yaml").read_text())
    reg = load_registry()
    base = reg[cfg["base"]]
    records = finetune.load_dolly()
    sets = finetune.prepare(records, sizes=cfg["sizes"], seed=cfg["seed"], shuffle_responses=True)
    names = args.set or list(sets)
    for name in names:
        out = Path("adapters") / f"smollm_{name}"
        finetune.write_jsonl(sets[name], out / "train.jsonl")
        finetune.train_lora(base, sets[name], out, cfg)
        print(f"finetune[{name}]: {len(sets[name])} rows -> {out}")
    return 0
```
and in `build_parser`:
```python
    f = sub.add_parser("finetune"); f.add_argument("--set", action="append"); f.set_defaults(fn=cmd_finetune)
```

- [ ] **Step 7: Register adapters in configs/models.yaml**

Append under `models:`:
```yaml
  smollm_n500:
    repo: HuggingFaceTB/SmolLM2-360M
    revision: main
    family: smollm
    template: alpaca
    adapter: adapters/smollm_n500
  smollm_n2000:
    repo: HuggingFaceTB/SmolLM2-360M
    revision: main
    family: smollm
    template: alpaca
    adapter: adapters/smollm_n2000
  smollm_full:
    repo: HuggingFaceTB/SmolLM2-360M
    revision: main
    family: smollm
    template: alpaca
    adapter: adapters/smollm_full
  smollm_format_only:
    repo: HuggingFaceTB/SmolLM2-360M
    revision: main
    family: smollm
    template: alpaca
    adapter: adapters/smollm_format_only
```
and change the ladder: `smollm: [smollm_base, smollm_n500, smollm_n2000, smollm_full, smollm_format_only]`.

Note: `smollm_base` keeps `template: alpaca` so base and adapters see the identical prompt format; the base's raw run comes from `--raw`. Add a `Makefile` target:
```make
finetune:
	uv run gricean finetune
```

- [ ] **Step 8: Run full suite, smoke-train on 16 rows, commit**

Run: `uv run pytest -v`
Expected: all passed.

Smoke run (network + a few minutes):
```bash
uv run python -c "
from pathlib import Path; import yaml
from gricean import finetune as F
from gricean.lm import load_registry
cfg = yaml.safe_load(open('configs/finetune.yaml')); cfg['epochs']=1
rows = F.prepare(F.load_dolly(), sizes=[16], seed=0, shuffle_responses=False)['n16']
F.train_lora(load_registry()['smollm_base'], rows, Path('/tmp/smoke_adapter'), cfg)
print('ok')"
```
Expected: prints `ok`; `/tmp/smoke_adapter/adapter_model.safetensors` exists.

```bash
git add src/gricean/finetune.py src/gricean/cli.py configs tests/test_finetune.py Makefile
git commit -m "feat: dolly preparation and LoRA fine-tuning with format-only and scale controls"
```

---

### Task 13: Notebook 04 (fine-tune viewer)

**Files:**
- Create: `notebooks/04_finetune.py`

- [ ] **Step 1: Write the notebook**

```python
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    from gricean.lm import ladders
    from gricean.report import logodds_table
    return Path, json, ladders, logodds_table, mo, pd


@app.cell
def _(Path, json, mo, pd):
    logs = {}
    for d in sorted(Path("adapters").glob("smollm_*")):
        f = d / "train_log.json"
        if f.exists():
            logs[d.name] = pd.DataFrame([r for r in json.loads(f.read_text()) if "loss" in r])
    out = mo.md("No adapters trained yet. Run `make finetune`.") if not logs else mo.md(
        "\n".join(f"- **{k}**: {len(v)} logged steps, final loss {v['loss'].iloc[-1]:.3f}" for k, v in logs.items()))
    out
    return (logs,)


@app.cell
def _(logs, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    for k, v in logs.items():
        ax.plot(v["step"], v["loss"], label=k)
    ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.legend(fontsize=7)
    mo.mpl.interactive(fig) if logs else mo.md("")
    return


@app.cell
def _(Path, ladders, logodds_table, mo):
    try:
        out = mo.ui.table(logodds_table(Path("data/logprobs"), ladders()["smollm"]).round(3).reset_index())
    except Exception as e:
        out = mo.md(f"No smollm log-prob data yet: {e}")
    out
    return


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 2: Run the notebook test and commit**

Run: `uv run pytest tests/test_notebooks.py -v`
Expected: 5 passed.

```bash
git add notebooks/04_finetune.py
git commit -m "feat: fine-tune viewer notebook"
```

---

### Task 14: Determinism check, README replication section, full-pipeline dry run

**Files:**
- Create: `scripts/determinism_check.py`
- Modify: `README.md`, `Makefile`

- [ ] **Step 1: Write the determinism script**

`scripts/determinism_check.py`:
```python
"""Greedy generation twice on the same machine; report whether outputs are byte-identical."""
import sys
from gricean.lm import load_model, load_registry
from gricean.stimuli import load_all

name = sys.argv[1] if len(sys.argv) > 1 else "olmo_base"
spec = load_registry()[name]
lm = load_model(spec)
items = [it for it in load_all() if not it.control][:5]
mismatch = 0
for it in items:
    a = lm.generate(lm.format(it.prompt, spec.template != "none"), max_new_tokens=60, temperature=None, seed=None)
    b = lm.generate(lm.format(it.prompt, spec.template != "none"), max_new_tokens=60, temperature=None, seed=None)
    mismatch += a != b
print(f"{name}: {len(items) - mismatch}/{len(items)} greedy outputs identical across two runs on {lm.device}")
sys.exit(1 if mismatch else 0)
```

- [ ] **Step 2: Extend README and Makefile**

Append to `README.md`:
```markdown
## Replication

1. `make install` (uv creates `.venv` from `uv.lock`).
2. `make test` (unit tests, no downloads).
3. Fill `revision:` in `configs/models.yaml` with the commit hashes you used.
4. `make freeze` — validates stimuli and writes `stimuli/CHECKSUMS`.
5. `make neither` — samples degenerate continuations (downloads two base models).
6. `make generate` then `make logprobs` — all models, templated and raw. Expect ~1–2 h on a 16 GB M3.
7. `make sheet` — writes `data/coding/coder1.csv`; code it. `uv run gricean sheet --coder2 --frac 0.25` for the blind sample.
8. `make finetune` — trains the four SmolLM2 adapters (~30 min each on M3); then rerun steps 6–7 for the `smollm` ladder.
9. `make report` — writes `paper/olmo/`. `uv run gricean report --ladder qwen` and `--ladder smollm` for the other tables.
10. `uv run python scripts/determinism_check.py olmo_base` — reports whether greedy decoding is deterministic on your device.

Committed `data/` lets you run steps 9 without any model.

## Notebooks

`uv run marimo edit notebooks/01_stimuli.py` (and 02–05). Notebooks only display; all logic lives in `src/gricean/`.
```

Add to `Makefile`:
```make
determinism:
	uv run python scripts/determinism_check.py olmo_base
```

- [ ] **Step 3: Dry run the whole pipeline end to end with real models on the seed stimuli**

Run, in order (network, ~20 min for 1B models on the two seed items):
```bash
make freeze
uv run gricean neither --family olmo
uv run gricean generate --model olmo_base
uv run gricean generate --model olmo_sft --raw && uv run gricean generate --model olmo_sft
uv run gricean logprobs --model olmo_base && uv run gricean logprobs --model olmo_sft
uv run gricean sheet
```
Expected: files under `data/responses/`, `data/logprobs/`, `data/coding/coder1.csv`; eyeball `data/responses/olmo_base__raw.jsonl` vs `olmo_sft__tmpl.jsonl` for the seed item: base should continue the text, SFT should answer. If MPS throws on OLMo2 half precision, set `dtype = torch.float32` for `mps` in `lm.load_model` and note it in the README.

- [ ] **Step 4: Commit**

```bash
git add scripts README.md Makefile
git commit -m "docs: replication steps and determinism check"
```

---

## Self-review notes

- **Spec coverage:** §3 models → Tasks 3, 12; §4 stimuli/versions/freeze → Task 2 (author writes remaining items into the same YAML shape); §5.1 → Tasks 5, 8; §5.2–5.3 → Tasks 6, 7; §6 tables/figure/examples → Task 9; §7 fine-tune + controls → Task 12; §8 layout, flow, idempotency, notebooks, replication package → Tasks 1, 4, 10, 11, 13, 14; §9 validation → unit tests throughout, integration dry run Task 14 step 3, determinism Task 14 step 1.
- **Review Focus mapping:** (1) Task 3 slow test; (2) Task 2 duplicate-id test; (3) Task 6 flag tests; (4) Task 8 blank/invalid tests; (5) Task 5 changed-item test.
- **Type consistency:** `Item.versions` dict used by generate/logprobs; `ModelSpec.template` string modes used by `render_template` and CLI `_models_for`; `KEY` tuple identical across stages; `read_jsonl` returns `[]` for missing files, relied on by report and notebooks.
- Portuguese items follow the same schema under `stimuli/pt/`; no separate code path, reports filter by `language` when the author calls `intent_table` on a language subset (add `coded[coded.language=="pt"]` in the notebook when needed).
