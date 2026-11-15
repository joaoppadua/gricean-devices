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
  run `uv run marimo check notebooks/` after editing them. Every notebook's first cell calls
  `enter_repo_root(mo.notebook_dir())` (also enforced): package paths are relative to the repo root,
  and the VS Code marimo extension starts the kernel inside `notebooks/`.
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
- PRs #1–#8 are merged into `main`: pipeline; pre-run gates + spec revision 2; the 78-item stimulus
  set; removal of the seed dry-run data; the `neither` set and the pre-run fixes below; the first
  real run (`data/responses`, `data/logprobs`); notebooks that run from any folder; the Line 1
  subset.
- Spec rev. 2: one documented OLMo 2 1B lineage. Phase 1 = stage-1 checkpoint
  (`stage1-step1907359-tokens4001B`, no instruction data documented); released base is a descriptive
  mid-training rung (≈8B FLAN tokens); SFT → DPO → Instruct as before. Self-run LoRA trains from stage 1
  (`adapters/olmo_*`, `selfrun` ladder, Alpaca template, no raw pass; it simulates SFT only, no DPO).
  `neither` from Pythia-1.4b, five samples, documentary criterion. Qwen pair retired to an optional
  contrast; SmolLM2 removed. GPT-3 → InstructGPT is quoted, not run.
- **Stimuli are frozen** (2026-11-10, combined checksum `157edf6d…`): 78 items, intended and literal
  continuations one short sentence each. Authorship (spec §4): 24 pattern items (`_01` and `_c01` of
  every file) written with Claude Code in the design phase; 31 written by the author (25 en, 6 pt);
  23 English items drafted by Claude Code and reviewed by the author. Models have run on
  this set (`make neither`), so do not edit `stimuli/**/*.yaml` or `stimuli/ablation.yaml`.
- **Models are pinned**: `configs/models.yaml` holds commit hashes (stage-1 branch = `9d3e4365…`).
- **`neither` is done**: `stimuli/neither/olmo.jsonl`, 60 rows, rule `containment-v3` (leading line
  breaks are skipped before truncation; the first run lost 143 of 300 samples to them, spec §5.3).
  No row is flagged. Author's pass replaced `chosen` in `en_pc_02`, `en_rq_03` and `en_ir_07` with
  another sample of the same row; `en_ir_02` ("Wellington, I think you said.") is kept on purpose.
- **The `olmo` run is done** (2026-11-11): `data/responses` holds 4,608 rows over nine passes (five
  `olmo` rungs, raw controls for the three tuned ones, and `olmo_stage1_alpaca`); `data/logprobs`
  holds 990 rows. Generation took ≈4.5–5 h, scoring under 10 min. Audit after the run: every stamp
  recomputes from the frozen stimuli and pinned config; no non-finite score; stage-1 raw and stage-1
  Alpaca agree exactly on the untemplated `embedded` version (120 responses, 90 scores).
- Features of that data, not defects: 385 empty and 11 whitespace-only responses (raw passes on
  `full`; SFT templated on `embedded`, 58 of 120; `olmo_base`, 55 of 552), which spec §5.1 treats as
  a finding; `olmo_stage1_alpaca` answers and then invents further `### Instruction:` pairs up to
  the 150-token cap, so coding needs a rule for run-on text.
- Response stamps cover prompt, mode, seed, model and decoding but not the stimuli checksum (a
  provenance label that nothing reads). That is why the seed dry-run rows were deleted before the
  run: they would have survived it with a pre-freeze checksum.
- Weights cached locally at the pinned revisions: OLMo stage-1, base, SFT, DPO (ships as
  `pytorch_model.bin`), Instruct; Pythia-1.4b. No download needed for the `olmo` ladder.
- **Line 1 codes a subset** (spec §5.1, amended 2026-11-11 before any coding): the greedy response
  of each model's main condition, 828 of 4,608 rows now and 1,380 of 6,816 after `make finetune`.
  `coding.select_coded` is the rule. Empty responses are written to the sheet already coded
  `neither` / `na` (41 rows); coder 2's 25% is drawn from the hand-coded rows (197). Sampled
  responses and the raw template control are not coded; the template control is reported from
  Line 2 (`logodds_template_control.csv`). Do not change the `row_id` format once coding starts:
  it is the key that carries codes across re-exports.
- **The sheet is written** (2026-11-11): `data/coding/coder1.csv`, 828 rows, committed before any
  code was entered by hand (787 blank, 41 empty responses coded by rule). `git diff` on that file
  shows what coding changed; only `intent` and `correctness` should differ.
- **First pass coded and frozen** (2026-11-14): commit `5d265c0` holds the 787 hand codes as
  entered (through a temporary coding page that is not in the repo, spec §5.1). A consistency check
  flagged 71 rows in 17 groups (`data/coding/coder1_check.csv`); the author decides them in the
  page. `na` was never used by hand; the 41 rule-coded rows carry it.
- **Revision after the check** (2026-11-14): the author decided all 71 flagged rows (35 revised,
  36 kept) and revised one comparison row; `coder1.csv` holds the revised codes and
  `coder1_check.csv` the decisions, rules and reasons. `git diff 5d265c0 -- data/coding/coder1.csv`
  is the record (37 rows, counting `en_ir_02` base full, made consistent with its stripped twin at
  the author's request in `74d5280`).
- **Embedded rule and second check** (2026-11-14): the author dropped the blanket "every
  `embedded` response is `neither`" for a narrow rule (spec §5.1 step 7: task done = `intended`,
  item's literal reading = `literal`, frame story only = `neither`). A second check flagged 38
  embedded rows in 5 groups (`data/coding/coder1_check2.csv`, groups `g18`–`g22`); the author
  decided them in the page (32 revised, 6 kept, `b8ed514`) and wants the codes to be theirs: flag
  and describe, do not code. Four leftover codes were then entered at the author's explicit
  request (role `after_recheck` in `coder1_check2.csv`, spec §5.1 step 10). 67 rows now differ
  from the first pass. On Line 1 as coded, tuned models go from 24 of 24 intended (`full`,
  `stripped`) to 1, 7 and 7 of 24 (`embedded`; 0, 2, 2 before the second check), still the
  opposite of the §5.4 prediction for that version.
- **Line 1 coding by coder 1 is closed** unless the author reopens it. Branch `first-pass-codes`
  holds all of it and needs a PR (merge commit, not squash: the spec cites its hashes).
- **First report on coded data** (2026-11-14): `paper/olmo/` regenerated. Tables are per language
  (`*_pt.csv` for Portuguese; the table functions refuse to pool). `make report` uses `intent` only;
  `correctness` is in no table yet. No kappa until coder 2 exists.
- **Self-run adapters trained** (2026-11-14/15): `adapters/olmo_{n500,n2000,full,format_only}`
  (weights, training rows, loss log). MPS memory grows with every new batch shape, so training pads
  batches to multiples of 64 tokens and releases the cache after each optimizer step
  (`finetune.collate`, `ReleaseMpsCache`); batch 2 × accumulation 8; Dolly revision pinned in
  `configs/finetune.yaml`. Measured: 11 min, 40 min, 3 h 33 min, 3 h 30 min; 3.2–3.8 GiB (spec §7).
  Train one process per adapter (`gricean finetune --set <name>`), smallest first.
- **Self-run passes done** (2026-11-15): 2,208 responses and 480 scored rows for the four adapters,
  Alpaca template only, two hours. Run them per model (`--model`), not `--all`, which loads all
  nine finished models to find nothing to do. No empty response, no non-finite score.
- **Second batch of the sheet** (2026-11-15): `coder1.csv` has 1,380 rows; the 552 new ones
  (138 per adapter) were committed blank at `5505381` and wait for the author. `paper/selfrun/`
  holds Line 2 only until they are coded (the report skips Line 1 while any `intent` is blank).
- Next: the author codes the 552 rows (coding page, second batch; first batch read-only there);
  copy the codes into `coder1.csv`; `uv run gricean report --ladder selfrun` and `--ladder olmo`;
  then `uv run gricean sheet --coder2 --frac 0.25`, once, for the second coder.
- Tests must not assert on the repo's own evolving state (real stimuli, `configs/models.yaml`): two
  broke that way, at the full set and at the pin. Build the fixture in `tmp_path` instead.
- Deferred minors from review: row ids render seeds as `1.0`/`nan`; rows of deleted stimuli persist
  until another row recomputes; responses notebook run button only does `--raw`; ladder figure omits
  CIs; `_strip` removes every period; `pin_registry` rejects quoted `revision: "main"`; no `--hub`
  flag to bypass a stale local cache; the stripped-equals-full ablation check cannot fire alone
  while `strip_punctuation` covers `?!.`; spec §5.3 says "non-empty" where the code requires two
  words; the Qwen contrast can generate but not score.
