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

## Current status (2026-11-08, evening)
- PR #1 (pipeline) and PR #2 (pre-run gates + spec revision 2) are merged into `main`.
- Spec rev. 2: one documented OLMo 2 1B lineage. Phase 1 = stage-1 checkpoint
  (`stage1-step1907359-tokens4001B`, no instruction data documented); released base is a descriptive
  mid-training rung (≈8B FLAN tokens); SFT → DPO → Instruct as before. Self-run LoRA trains from stage 1
  (`adapters/olmo_*`, `selfrun` ladder, Alpaca template, no raw pass; it simulates SFT only, no DPO).
  `neither` from Pythia-1.4b, five samples, documentary criterion. Qwen pair retired to an optional
  contrast; SmolLM2 removed. GPT-3 → InstructGPT is quoted, not run.
- **Branch `stimuli`** (3 commits ahead of `main`, pushed) is where the author writes the items.
  All 12 category files exist with every item pre-numbered (`en_<abbr>_01..08`, `_c01..c02`;
  `pt_<abbr>_01..02`, `_c01`); unfilled fields read `TODO`. Item `_01` and `_c01` of every file are
  written as the pattern (English: the spec's canonical examples; Portuguese: native drafts to revise).
  24 of 78 items written. Placeholder items load, do not count toward quotas, and block the full
  freeze and every model stage; `uv run gricean freeze --partial` passes meanwhile.
- Author's loop per file: fill the `TODO`s, `uv run gricean freeze --partial`, check the quota report
  (`stimuli.check_quotas` message), commit per category (plan Tasks 5–11). Portuguese politeness
  markers for stripping are only `por favor`/`obrigado`/`obrigada`; extend `stimuli/ablation.yaml`
  before the final freeze if the pt items need more.
- Weights cached locally: OLMo stage-1, base, SFT, DPO (ships as `pytorch_model.bin`), Instruct;
  Pythia-1.4b. Nothing else to download before `make pin`.
- Then: `make freeze`, `make pin`, `make neither`, approval pass over `stimuli/neither/olmo.jsonl`,
  `make generate`, `make logprobs`, `make sheet`, code, `make finetune` (overnight, ≈12–15 h), rerun
  generate/logprobs for `selfrun`, `make report` (`--ladder olmo`, `--ladder selfrun`).
- Deferred minors from review: row ids render seeds as `1.0`/`nan`; Line 1 pools greedy with
  samples; rows of deleted stimuli persist until another row recomputes; responses notebook run
  button only does `--raw`; ladder figure omits CIs; `_strip` removes every period; `pin_registry`
  rejects quoted `revision: "main"`; no `--hub` flag to bypass a stale local cache; the
  stripped-equals-full ablation check cannot fire alone while `strip_punctuation` covers `?!.`;
  `selfrun` report writes an empty `template_control.csv` without warning; spec §5.3 says
  "non-empty" where the code requires two words; the Qwen contrast can generate but not score.
