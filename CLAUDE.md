# gricean-devices

Simulation code for Pádua, "LLMs are Gricean machines" (Revista Texto Livre 2027 dossier).
Instruction fine-tuned LLMs as Gricean devices: dissociation stimuli where literal and intended
readings come apart, run through open base → SFT → DPO → Instruct checkpoints.

## Read first
- `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md` — the binding design (why each choice was made).
- `docs/superpowers/plans/2026-09-26-gricean-simulation.md` — the implementation plan the code follows.
- `README.md` — replication steps, in order, with expected runtimes on a 16 GB M3.
- Debate context (Hearer's Meaning exchange, anticipated objections):
  `~/Library/CloudStorage/Dropbox/Artigos_Academicos/_tools/claude-coreader/hearers-meaning/CLAUDE.md`.

## Working rules
- Open-source only; no paid APIs; everything must run on a 16 GB Apple M3.
- Notebooks display, the package computes: no helper `def`s in `notebooks/*.py` (a test enforces it);
  run `uv run marimo check notebooks/` after editing them.
- TDD: write the failing test first. `uv run pytest` (fast suite); `uv run pytest -m slow` downloads a tiny model.
- Stimuli are frozen before any model run (`make freeze`); every stage refuses drifted stimuli and
  any item still carrying the `TODO` placeholder (`stimuli.PLACEHOLDER`).
- Stages are idempotent per item; re-running recomputes only rows whose inputs changed.
- Protocol invariants baked into the code (do not silently change): `embedded` version is never
  chat-templated; tuned models also run raw as the template control, except self-run models
  (`raw_control: false`), which never run raw; untemplated prompts get
  `raw_separator` before scored continuations; decoding knobs are pinned in `lm.DECODING`;
  flagged `neither` rows block scoring until `approved: true`.
- Commit `data/`, `stimuli/neither/`, `adapters/` so tables regenerate without models.

## Current status (2026-11-08)
- PR #1 (pipeline) merged. PR #2 (`pre-run-tasks`) holds: freeze quota/ablation gates with `--partial`,
  `make pin`, empty-greedy marker, spec revision 2 and its implementation.
- Spec rev. 2: one documented OLMo 2 1B lineage. Phase 1 = stage-1 checkpoint
  (`stage1-step1907359-tokens4001B`, no instruction data documented); released base is a descriptive
  mid-training rung (≈8B FLAN tokens); SFT → DPO → Instruct as before. Self-run LoRA trains from stage 1
  (`adapters/olmo_*`, `selfrun` ladder, Alpaca template, no raw pass). `neither` from Pythia-1.4b, five
  samples, chosen by the documentary criterion (no instruction data in training). Qwen pair retired
  from the paper, kept as an optional contrast; SmolLM2 removed. GPT-3 → InstructGPT is quoted, not run.
- Stimuli: still the two seed items. Author's next task: write the 48 + 12 English items and the
  12 + 6 Portuguese mirror (`docs/superpowers/plans/2026-11-08-pre-run-tasks.md`, Tasks 5–11), validating
  with `uv run gricean freeze --partial`.
- Then: `make freeze`, `make pin`, `make neither`, approval pass over `stimuli/neither/olmo.jsonl`,
  `make generate`, `make logprobs`, `make sheet`, code, `make finetune` (overnight), rerun 6–7,
  `make report` (`--ladder olmo`, `--ladder selfrun`).
- Deferred minors from review: row ids render seeds as `1.0`/`nan`; Line 1 pools greedy with
  samples; rows of deleted stimuli persist until another row recomputes; responses notebook run
  button only does `--raw`; ladder figure omits CIs; `_strip` removes every period; `pin_registry`
  rejects quoted `revision: "main"`; no `--hub` flag to bypass a stale local cache; the
  stripped-equals-full ablation check cannot fire alone while `strip_punctuation` covers `?!.`.
