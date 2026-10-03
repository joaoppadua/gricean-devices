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
- Stimuli are frozen before any model run (`make freeze`); every stage refuses drifted stimuli.
- Stages are idempotent per item; re-running recomputes only rows whose inputs changed.
- Protocol invariants baked into the code (do not silently change): `embedded` version is never
  chat-templated; tuned models also run raw as the template control; untemplated prompts get
  `raw_separator` before scored continuations; decoding knobs are pinned in `lm.DECODING`;
  flagged `neither` rows block scoring until `approved: true`.
- Commit `data/`, `stimuli/neither/`, `adapters/` so tables regenerate without models.

## Current status (2026-09-27)
- Pipeline complete and reviewed (PR #1, branch `pipeline` → `main`).
- Seed stimuli only (`stimuli/en/indirect_request.yaml`, 2 items). Seed dry run on OLMo base + SFT committed.
- Author's open tasks: (1) write the 48 + 12 English items and the 12 + 6 Portuguese mirror;
  (2) choose the `neither` source model — Qwen2.5-1.5B base answered the seed question in 2 of 3
  samples; a GPT-3-era base (e.g. `EleutherAI/pythia-1.4b`) is the suggested alternative
  (edit `neither_source` in `configs/models.yaml`); (3) decide whether printed base examples use
  greedy (empty on the seed prompt: EOS is the top first token) or sampled responses.
- Then: pin `revision:` hashes in `configs/models.yaml`, `make freeze`, `make neither`,
  `make generate`, `make logprobs`, `make sheet`, code, `make finetune`, `make report`.
- Deferred minors from review: row ids render seeds as `1.0`/`nan`; Line 1 pools greedy with
  samples; rows of deleted stimuli persist until another row recomputes; responses notebook run
  button only does `--raw`; ladder figure omits CIs; `_strip` removes every period.
