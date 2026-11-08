# OLMo Lineage (Spec Rev. 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-point the pipeline at one documented OLMo 2 1B lineage: stage-1 checkpoint as phase 1, five-rung official ladder, self-run fine-tune on stage 1, Pythia as the single `neither` source, Qwen and SmolLM2 out of the default run.

**Architecture:** Almost everything is configuration: `configs/models.yaml` gains the stage-1 and self-run entries and loses the SmolLM2 ladder, `configs/finetune.yaml` retargets the LoRA base, `configs/generation.yaml` raises `neither` sampling to five. Two small code changes carry the spec's new rules: a per-model `raw_control` flag so the self-run ladder skips the raw template pass, and an `adapter_prefix` so trained adapters land under `adapters/olmo_*`. Docs, Makefile and notebook 04 follow the config.

**Tech Stack:** Python 3.12, PyYAML, pytest, marimo. Nothing new.

**Spec:** `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md`, revision 2 (Revision log, §2, §3.1, §3.2, §3.4, §5.3, §7, §12). The pre-run plan `docs/superpowers/plans/2026-11-08-pre-run-tasks.md` still governs stimulus writing; this plan supersedes its Task 2 steps 5–6 and the family list in its Task 12.

## Global Constraints

- Open-source only; runs on a 16 GB Apple M3 (spec §2).
- Every checkpoint pinned to a revision hash via `make pin`; the stage-1 checkpoint is `allenai/OLMo-2-0425-1B` at branch `stage1-step1907359-tokens4001B` (spec §3.1, §12).
- Protocol invariants unchanged: `embedded` never chat-templated; untemplated prompts get `raw_separator`; decoding knobs pinned in `lm.DECODING`; flagged `neither` rows block scoring until approved.
- New rule (spec §3.4): the raw-text template control applies to the official ladder only; self-run models are never run raw.
- `neither` (spec §5.3): source `pythia_base` for every ladder, five samples per item at temperature 1.0, 20 tokens or first line break.
- Self-run (spec §7): base is the stage-1 checkpoint; "full" is all 10,544 Dolly pairs; sizes 500 and 2,000; format-only control; Alpaca template; adapters committed under `adapters/olmo_*`.
- Registry names fixed by spec §12: `olmo_stage1`, `olmo_stage1_alpaca`, `olmo_n500`, `olmo_n2000`, `olmo_full`, `olmo_format_only`, `pythia_base`; ladders `olmo` and `selfrun`.
- Notebooks display only; `uv run marimo check notebooks/` after edits. TDD; `uv run pytest`.

## Review Focus

- A model with `raw_control: false` and `template: none` is contradictory (a base model's only run *is* raw). Expected: `load_registry` rejects it. Pinned in Task 1.
- `olmo_stage1` and `olmo_stage1_alpaca` share repo and revision; their response files must not collide and their stamps must differ. Expected: rows keyed by model name, as now. Pinned in Task 1 (CLI test asserts both files exist).
- `make pin` on a branch-style revision (`stage1-step…`) must resolve to a sha, not be mistaken for one. Pinned in Task 1.
- `logprobs` for a self-run model must find `stimuli/neither/olmo.jsonl` through `family: olmo`. Pinned in Task 1 (CLI test runs logprobs for a self-run-style model).
- `generation.yaml` with `neither.n` larger than the seed list would silently sample fewer. Expected: a guard. Pinned in Task 3.

---

### Task 1: Registry rev. 2 and the `raw_control` flag

**Files:**
- Modify: `configs/models.yaml`
- Modify: `src/gricean/lm.py:28-35` (`ModelSpec`), `:42-43` (`load_registry`)
- Modify: `src/gricean/cli.py:19-32` (`_models_for`)
- Modify: `Makefile` (`neither` target)
- Test: `tests/test_lm.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `ModelSpec`, `load_registry`, `ladders`, `neither_source`, `pin_registry`, `_models_for`.
- Produces: `ModelSpec.raw_control: bool = True` (new optional field, last). `load_registry` raises `ValueError("<name>: raw_control: false needs a template")` when `template == "none"` and `raw_control` is false. `_models_for` with `--all` appends `(name, False)` for a templated model only when `spec.raw_control`. Registry contents as spec §12: `olmo_stage1` (template `none`), `olmo_stage1_alpaca` (template `alpaca`, `raw_control: false`), four `olmo_*` adapters (template `alpaca`, `raw_control: false`, `adapter: adapters/olmo_<set>`), `pythia_base`; `ladders: {olmo: [olmo_stage1, olmo_base, olmo_sft, olmo_dpo, olmo_instruct], selfrun: [olmo_stage1_alpaca, olmo_n500, olmo_n2000, olmo_full, olmo_format_only]}`; `neither_source: {olmo: pythia_base}`. `qwen_base` and `qwen_instruct` stay in `models` with a comment ("optional contrast run, not in any ladder"); every `smollm_*` entry and the `smollm` and `qwen` ladders are removed.

- [ ] **Step 1: Write the failing tests**

Replace `test_registry_loads_specs_and_ladders` in `tests/test_lm.py` with:

```python
CFG = Path("configs/models.yaml")

def test_registry_rev2_ladders_and_neither_source():
    reg, lad = L.load_registry(CFG), L.ladders(CFG)
    assert lad["olmo"] == ["olmo_stage1", "olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]
    assert lad["selfrun"] == ["olmo_stage1_alpaca", "olmo_n500", "olmo_n2000", "olmo_full", "olmo_format_only"]
    assert set(lad) == {"olmo", "selfrun"}
    assert L.neither_source(CFG) == {"olmo": "pythia_base"}
    s1, s1a = reg["olmo_stage1"], reg["olmo_stage1_alpaca"]
    assert (s1.repo, s1.revision) == (s1a.repo, s1a.revision) == ("allenai/OLMo-2-0425-1B", "stage1-step1907359-tokens4001B")
    assert s1.template == "none" and s1.raw_control is True
    assert s1a.template == "alpaca" and s1a.raw_control is False
    assert all(reg[n].family == "olmo" and reg[n].raw_control is False and reg[n].adapter == f"adapters/{n}"
               for n in lad["selfrun"][1:])
    assert not any(n.startswith("smollm") for n in reg)

def test_registry_rejects_raw_control_false_on_base_model(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("models:\n  b: {repo: x, revision: main, family: f, template: none, raw_control: false}\n")
    with pytest.raises(ValueError, match="raw_control"):
        L.load_registry(p)

def test_pin_registry_resolves_branch_style_revision(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("models:\n  s1: {repo: org/a, revision: stage1-step1907359-tokens4001B, family: f, template: none}\n")
    assert L.pin_registry(p, resolver=lambda repo, rev: "a" * 40) == {"s1": "a" * 40}
    assert L.load_registry(p)["s1"].revision == "a" * 40
```

In `tests/test_cli.py`, extend the `project()` fixture's registry with a self-run-style model
`"a": {"repo": "x", "revision": "1", "family": "t", "template": "alpaca", "raw_control": False}` in ladder `t`
(`"ladders": {"t": ["b", "s", "a"], "u": ["q"]}`), and add:

```python
def test_all_skips_raw_pass_for_raw_control_false(tmp_path: Path, monkeypatch):
    root = project(tmp_path); monkeypatch.chdir(root)
    cli.main(["freeze", "--partial"])
    assert cli.main(["neither", "--family", "t", "--family", "u"], loader=fake_loader) == 0
    assert cli.main(["generate", "--all"], loader=fake_loader) == 0
    assert (root / "data/responses/a__tmpl.jsonl").exists()
    assert not (root / "data/responses/a__raw.jsonl").exists()
    assert (root / "data/responses/s__raw.jsonl").exists()        # official-ladder tuned model still runs raw
    assert cli.main(["logprobs", "--all"], loader=fake_loader) == 0
    assert (root / "data/logprobs/a__tmpl.jsonl").exists()        # neither found via family
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_lm.py tests/test_cli.py -q`
Expected: FAIL. `ModelSpec.__init__() got an unexpected keyword argument 'raw_control'` for the CLI fixture; `KeyError: 'olmo_stage1'` and `AttributeError: 'ModelSpec' object has no attribute 'raw_control'` in `test_lm.py`.

- [ ] **Step 3: Implement**

`ModelSpec` gains `raw_control: bool = True` after `adapter`. `load_registry` validates the contradiction after constructing each spec. `_models_for`: in the `args.all` branch, `pairs += [(n, True)] + ([(n, False)] if spec.raw_control else [])`. Rewrite `configs/models.yaml` per Interfaces, keeping the `make pin` header comment and adding one comment line per group: stage 1 ("phase 1: 4.0T tokens, no mid-training; fp32 on the Hub"), base ("mid-training adds ≈8B FLAN tokens; descriptive rung"), self-run ("adapters trained by `make finetune` from olmo_stage1; alpaca template; no raw pass, spec §3.4"), pythia ("`neither` source, in no ladder"). Makefile: `neither:` → `uv run gricean neither --family olmo`.

- [ ] **Step 4: Run the full fast suite**

Run: `uv run pytest -q`
Expected: all pass. (`test_alpaca_format` builds a `ModelSpec` with `family="smollm"`; that is a free string and still passes.)

- [ ] **Step 5: Commit**

```bash
git add configs/models.yaml src/gricean/lm.py src/gricean/cli.py Makefile tests/test_lm.py tests/test_cli.py
git commit -m "feat: rev. 2 registry — OLMo stage-1 ladder, selfrun ladder, raw_control flag"
```

---

### Task 2: Self-run fine-tune on the stage-1 checkpoint

**Files:**
- Modify: `configs/finetune.yaml`
- Modify: `src/gricean/cli.py:96-110` (`cmd_finetune`)
- Modify: `notebooks/04_finetune.py:21`, `:47-49`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `finetune.load_dolly`, `finetune.prepare`, `finetune.train_lora(base_spec, rows, out_dir, cfg)`, registry from Task 1.
- Produces: `configs/finetune.yaml` with `base: olmo_stage1` and `adapter_prefix: olmo`; `cmd_finetune` writes to `Path("adapters") / f"{cfg['adapter_prefix']}_{name}"`. The directory names therefore equal the self-run registry names (`olmo_n500` …), which is what `adapter: adapters/olmo_n500` in Task 1 expects.

- [ ] **Step 1: Write the failing test**

```python
def test_finetune_writes_adapters_under_prefix(tmp_path: Path, monkeypatch):
    from gricean import finetune as F
    root = project(tmp_path); monkeypatch.chdir(root)
    shutil.copy(Path(__file__).parents[1] / "configs/finetune.yaml", root / "configs/finetune.yaml")
    cfg = yaml.safe_load((root / "configs/finetune.yaml").read_text())
    reg = yaml.safe_load((root / "configs/models.yaml").read_text())
    reg["models"][cfg["base"]] = {"repo": "x", "revision": "1", "family": "t", "template": "none"}
    yaml.safe_dump(reg, (root / "configs/models.yaml").open("w"))
    monkeypatch.setattr(F, "load_dolly", lambda: [
        {"category": "open_qa", "instruction": f"q{i}", "response": f"a{i}", "context": ""} for i in range(600)])
    monkeypatch.setattr(F, "train_lora", lambda base, rows, out, cfg: Path(out).mkdir(parents=True, exist_ok=True))
    assert cli.main(["finetune", "--set", "n500"]) == 0
    assert (root / "adapters/olmo_n500/train.jsonl").exists()
    assert cfg["base"] == "olmo_stage1"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_cli.py -k finetune_writes -q`
Expected: FAIL on `adapters/olmo_n500/train.jsonl` (the file lands under `smollm_n500`) or on `cfg["base"]`.

- [ ] **Step 3: Implement**

`configs/finetune.yaml`: `base: olmo_stage1`, add `adapter_prefix: olmo`, and amend the dtype comment to: "bf16; if MPS runs out of memory on the 1B model use batch_size 2 / grad_accum 8 (same effective batch)". `cmd_finetune` uses the prefix. Notebook 04: glob `"olmo_*"` and `ladders()["selfrun"]`, and the two message strings mention `selfrun`/`olmo` instead of `smollm`.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run marimo check notebooks/`
Expected: all pass; marimo prints nothing.

- [ ] **Step 5: Commit**

```bash
git add configs/finetune.yaml src/gricean/cli.py notebooks/04_finetune.py tests/test_cli.py
git commit -m "feat: self-run fine-tune targets the OLMo stage-1 checkpoint; adapters under adapters/olmo_*"
```

---

### Task 3: `neither` sampling to five, and a seed-count guard

**Files:**
- Modify: `configs/generation.yaml:10-14`
- Modify: `src/gricean/neither.py:40-56` (`sample_neither`)
- Test: `tests/test_neither.py`

**Interfaces:**
- Consumes: `sample_neither(source, source_spec, family, items, *, cfg, stimuli_checksum, root)`.
- Produces: `sample_neither` raises `ValueError("neither.n=<n> exceeds <k> seeds")` before any generation when `cfg["n"] > len(cfg["seeds"])`. `configs/generation.yaml` `neither:` block has `n: 5` and `seeds: [11, 12, 13, 14, 15]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sample_neither_rejects_more_samples_than_seeds(tmp_path):
    spec = ModelSpec(name="p", repo="q", revision="r1", family="pythia", template="none")
    cfg = dict(CFG, n=5, seeds=[11, 12, 13])
    with pytest.raises(ValueError, match="seeds"):
        N.sample_neither(FakeLM("p"), spec, "olmo", [item()], cfg=cfg, stimuli_checksum="abc", root=tmp_path)

def test_generation_config_samples_five_neither():
    cfg = yaml.safe_load(Path("configs/generation.yaml").read_text())["neither"]
    assert cfg["n"] == 5 and len(cfg["seeds"]) >= 5
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_neither.py -q`
Expected: FAIL. The first does not raise; the second sees `n == 3`.

- [ ] **Step 3: Implement**

One guard line at the top of `sample_neither`; edit the YAML.

- [ ] **Step 4: Run the suite and commit**

Run: `uv run pytest -q` → PASS.

```bash
git add configs/generation.yaml src/gricean/neither.py tests/test_neither.py
git commit -m "feat: five neither samples per item; guard n against the seed list"
```

---

### Task 4: Docs follow the config

**Files:**
- Modify: `README.md:13-26`, `CLAUDE.md` (Working rules "Protocol invariants", Current status), `docs/superpowers/plans/2026-11-08-pre-run-tasks.md` (note at top)

No code; no test. Verification is a read.

- [ ] **Step 1: README replication steps**

Step 5: "`make neither` — samples degenerate continuations from Pythia-1.4b (downloads it, 5.9 GB)." Step 6: "all models in the `olmo` ladder (stage 1 downloads as fp32, 5.9 GB), templated and raw; expect ~2–3 h on a 16 GB M3." Step 8: "`make finetune` — trains the four OLMo stage-1 adapters (bf16 LoRA; ≈5–6 h each for `full` and `format_only`, ≈1 h for `n2000`, minutes for `n500`; ≈12–15 h in total, run overnight); then rerun steps 6–7 for the `selfrun` ladder (`gricean generate --all` picks the adapters up once `adapters/olmo_*` exist)." Step 9: "`uv run gricean report --ladder selfrun` for the self-run table." Remove the mention of `--ladder qwen`. Add one line after the list: "The Qwen2.5 pair remains in `configs/models.yaml` as an optional contrast (`gricean generate --model qwen_base`); it is not in any ladder and not reported."

- [ ] **Step 2: CLAUDE.md**

Protocol invariants: append "self-run models (`raw_control: false`) never run raw; the raw template control is for the official ladder only". Current status: replace the open-task list with: date 2026-11-08; "spec rev. 2 (one OLMo lineage) implemented; stage-1 checkpoint is phase 1; `neither` from Pythia; stimuli still the two seed items; next: write the 78 items (pre-run plan Tasks 5–11), then `make freeze`, `make pin`, `make neither`, approval pass, `make generate`, `make logprobs`, `make sheet`, `make finetune`, `make report`." Keep the deferred-minors list; add the three from the pre-run review (quoted `revision:` values, no `--hub` bypass, seed rows recompute after pin).

- [ ] **Step 3: Pre-run plan note**

Insert after the header block of `docs/superpowers/plans/2026-11-08-pre-run-tasks.md`: "> **Superseded by spec rev. 2 (2026-11-08):** Task 2 steps 5–6 are decided (Pythia, documentary criterion); Task 12 step 2 runs `make neither` for the `olmo` family only; see `2026-11-08-olmo-lineage-rev2.md`."

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md docs/superpowers/plans/2026-11-08-pre-run-tasks.md
git commit -m "docs: replication steps, status and invariants for spec rev. 2"
```

---

### Task 5: Smoke the stage-1 checkpoint (author's machine, ~10 min, 5.9 GB download)

Not a code task; confirms the new rung loads and that raw prompting and the Alpaca template both work before the 78-item run.

- [ ] **Step 1:** `uv run gricean freeze --partial`
- [ ] **Step 2:** `uv run gricean generate --model olmo_stage1 --raw` → prints `generate[olmo_stage1, template=False]: N rows` and `data/responses/olmo_stage1__raw.jsonl` has a non-empty response for `en_ir_01`.
- [ ] **Step 3:** `uv run gricean generate --model olmo_stage1_alpaca` → `data/responses/olmo_stage1_alpaca__tmpl.jsonl` exists; its `prompt_text` starts with `### Instruction:`.
- [ ] **Step 4:** Do **not** commit the two response files (they will recompute after `make pin` and the real freeze).
