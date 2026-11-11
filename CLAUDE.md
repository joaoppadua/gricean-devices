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

## Current status (2026-11-11)
- PRs #1–#4 are merged into `main`: pipeline; pre-run gates + spec revision 2; the 78-item stimulus
  set; removal of the seed dry-run data. Branch `pre-run` carries the `neither` set and the fixes below.
- Spec rev. 2: one documented OLMo 2 1B lineage. Phase 1 = stage-1 checkpoint
  (`stage1-step1907359-tokens4001B`, no instruction data documented); released base is a descriptive
  mid-training rung (≈8B FLAN tokens); SFT → DPO → Instruct as before. Self-run LoRA trains from stage 1
  (`adapters/olmo_*`, `selfrun` ladder, Alpaca template, no raw pass; it simulates SFT only, no DPO).
  `neither` from Pythia-1.4b, five samples, documentary criterion. Qwen pair retired to an optional
  contrast; SmolLM2 removed. GPT-3 → InstructGPT is quoted, not run.
- **Stimuli are frozen** (2026-11-10, combined checksum `157edf6d…`): 78 items, intended and literal
  continuations one short sentence each. The author wrote every Portuguese item and most English
  ones; 23 English items were drafted with Claude Code and reviewed by the author. Models have run on
  this set (`make neither`), so do not edit `stimuli/**/*.yaml` or `stimuli/ablation.yaml`.
- **Models are pinned**: `configs/models.yaml` holds commit hashes (stage-1 branch = `9d3e4365…`).
- **`neither` is done**: `stimuli/neither/olmo.jsonl`, 60 rows, rule `containment-v3` (leading line
  breaks are skipped before truncation; the first run lost 143 of 300 samples to them, spec §5.3).
  No row is flagged. Author's pass replaced `chosen` in `en_pc_02`, `en_rq_03` and `en_ir_07` with
  another sample of the same row; `en_ir_02` ("Wellington, I think you said.") is kept on purpose.
- `data/` is empty by design until `make generate`: the seed rows would have survived the real run
  with a pre-freeze checksum, because response stamps cover prompt, mode, seed, model and decoding but
  not the stimuli checksum (a provenance label that nothing reads).
- Weights cached locally at the pinned revisions: OLMo stage-1, base, SFT, DPO (ships as
  `pytorch_model.bin`), Instruct; Pythia-1.4b. No download needed for the `olmo` ladder.
- Next: `make generate` (4,608 rows over nine passes: five `olmo` rungs, raw controls for the three
  tuned ones, and `olmo_stage1_alpaca`), `make logprobs` (990 rows), together ≈2–3 h; `make sheet`,
  code; `make finetune` (overnight, ≈12–15 h); rerun generate/logprobs for `selfrun`; `make report`
  (`--ladder olmo`, `--ladder selfrun`).
- Tests must not assert on the repo's own evolving state (real stimuli, `configs/models.yaml`): two
  broke that way, at the full set and at the pin. Build the fixture in `tmp_path` instead.
- Deferred minors from review: row ids render seeds as `1.0`/`nan`; Line 1 pools greedy with
  samples; rows of deleted stimuli persist until another row recomputes; responses notebook run
  button only does `--raw`; ladder figure omits CIs; `_strip` removes every period; `pin_registry`
  rejects quoted `revision: "main"`; no `--hub` flag to bypass a stale local cache; the
  stripped-equals-full ablation check cannot fire alone while `strip_punctuation` covers `?!.`;
  `selfrun` report writes an empty `template_control.csv` without warning; spec §5.3 says
  "non-empty" where the code requires two words; the Qwen contrast can generate but not score.
