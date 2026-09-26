# Design spec: computational simulation for "LLMs are Gricean machines"

Date: 2026-09-26
Status: draft for author review
Paper: Pádua, "LLMs are Gricean machines", Revista Texto Livre 2027 dossier

## 1. Purpose

The paper claims that instruction fine-tuned LLMs operate as Gricean devices:
they respond to what a user's prompt is *for* (the intended meaning) rather
than to what it literally says. The simulation must produce controlled
evidence for this claim that survives two anticipated objections from the
Hearer's Meaning debate (Hansen & Terkourafi 2023; Terkourafi & Hansen 2026):

- **Uptake objection.** Behavioural output only shows interpretation, not the
  mental process. Answer: a dissociation design. Where surface-form matching
  and intent-tracking predict different responses, the response pattern
  licenses inference to a latent intent-like representation, the standard
  cognitive-science logic for latent variables.
- **Co-optation objection.** The model derives "what the user wants" from
  public cues only, so intent-attribution is redundant (sources 1–6). Answer:
  cue ablation. If intent-tracking survives removal of individual surface cues
  in tuned models but not in base models, no single cue does the work;
  "derived from cues" is not "redundant".

### Scope decision on the paper's second prong

The summary makes two claims: (1) tuned models recognize user intention;
(2) they produce responses whose communicative import users recognize in turn.
The simulation measures only (1). Rationale (author's): an instruction-tuned
model remains a next-token predictor, so the only observable trace of having
recognized an intention *is* the relevant completion. The single object we
measure, the completion, is simultaneously evidence of recognition and an act
of production. The paper reads one measurement in both directions; prong (2)
is argued conceptually, not simulated separately.

## 2. Constraints

1. Outputs must fit a traditional journal article: printable text versions,
   small tables, one or two figures.
2. Everything open-source and runnable on an ordinary laptop (reference
   machine: Apple M3, 16 GB). No paid APIs, no subscription-only models.
3. Full replication package: pinned model revisions, pinned Python
   dependencies, committed intermediate data, one-command regeneration of
   tables.

GPT-3 is excluded: its base models were removed from OpenAI's API in
January 2024 and were never open. The "legacy" comparison is an open base
model.

## 3. Models

### 3.1 Main ladder: OLMo 2 1B (AI2, April 2025 release)

Same architecture, tokenizer and pretraining at every stage; Apache 2.0;
fine-tuning data (Tülu 3) public.

| Stage | Checkpoint | Adds | Role in argument |
|---|---|---|---|
| Base | `allenai/OLMo-2-0425-1B` | pretraining only | legacy analogue |
| SFT | `allenai/OLMo-2-0425-1B-SFT` | supervised instruction tuning | first Gricean stage |
| DPO | `allenai/OLMo-2-0425-1B-DPO` | preference optimization on human comparisons | RLHF analogue; theoretical endpoint |
| Instruct | `allenai/OLMo-2-0425-1B-Instruct` | RL with verifiable rewards (math/code correctness) | **control column** |

The Instruct stage trains against correctness checkers, not human judgements
of adequacy, so it is not part of the Gricean mechanism. It is kept as a
control: the prediction is that intent-tracking rises at SFT and DPO and stays
flat at Instruct, localizing the effect in the stages that use human feedback
about adequacy. If the column is uninformative it is dropped from the paper
but kept in the repo.

### 3.2 Robustness pair

`Qwen/Qwen2.5-1.5B` and `Qwen/Qwen2.5-1.5B-Instruct`, run once on the same
stimuli, reported as one extra table. Shows the effect is not an OLMo quirk.

### 3.3 Self-run fine-tune (enacted mechanism)

See §7.

### 3.4 Prompt formatting

Base models receive raw text. Tuned models receive their chat template. This
is part of the mechanism, not a confound, but tuned models are *also* run on
raw text (no template) so the paper can show the template alone does not
explain the effect.

### 3.5 Inference stack

Python, Hugging Face `transformers` on Apple MPS (CPU fallback elsewhere),
half precision. Every checkpoint pinned to a revision hash in
`configs/models.yaml`.

## 4. Stimuli

Hand-written by the author, English first, one YAML file per category under
`stimuli/`. Each item records: `id`, `category`, `prompt`, `literal_reading`,
`intended_reading`, `literal_continuation`, `intended_continuation`,
`language`, `version` (see cue ablation).

### 4.1 Dissociation categories (8 items each, 48 total)

| Category | Literal reading | Intended reading | Example |
|---|---|---|---|
| Indirect request | yes/no question | perform the task | "Can you tell me what year Brazil became independent?" |
| Pattern continuation | continue the list of tasks | do the one task | "Explain the moon landing to a 6-year-old" (Ouyang et al. 2022, Fig. 1) |
| Embedded instruction | continue the narrative | do the task inside it | "She asked: 'summarize this paragraph for me.' The paragraph was: ..." |
| Malformed instruction | garbled text, no task | repair and do the task | "tel me hw many days febuary has" |
| Underspecified request | ambiguous or incomplete | pick the plausible intent | "Translate 'obrigado'." |
| Rhetorical question | answer the question | acknowledge the point | "Who would ever want to wait three hours for a bus?" |

### 4.2 Controls (2 per category, 12 total)

Items where literal and intended readings coincide ("What year did Brazil
become independent?"). If a tuned model scores well on dissociation items only
because it answers everything, controls will not separate it from base; if it
tracks intent, both move together.

### 4.3 Cue ablation

Applied to indirect request, embedded instruction and pattern continuation.
Each item exists in three versions:

- `full`: as written;
- `stripped`: punctuation and politeness markers removed;
- `embedded`: placed inside an unrelated prose paragraph, no chat template
  for any model.

### 4.4 Portuguese subset

Two dissociation items per category (12) plus one control per category (6),
*mirrored* (written natively, not translated). Reported separately as a generalization check, never pooled with
English.

### 4.5 Freezing

`gricean.stimuli.freeze` validates the YAML and writes `stimuli/CHECKSUMS`.
Stimuli are frozen and committed before any model is run.

## 5. Measures

Two independent lines, both from the frozen stimuli.

### 5.1 Line 1: generated responses, human-coded

For every model stage × stimulus version: one greedy response (printed in the
paper) and three sampled responses (temperature 0.7, fixed seeds; feed the
counts). Max 150 new tokens.

Rubric, per response:

- **Intent** (primary): `literal` / `intended` / `neither`. `neither` covers
  refusals, off-topic text and degenerate loops. It is reported as its own
  count, never folded into `literal`.
- **Correctness** (secondary, only if the task was addressed): `correct` /
  `incorrect`. Kept separate so the paper can distinguish intent-tracking from
  knowledge.

Coder 1: the author. Coder 2: an independent coder on a 25% random sample,
blind to model stage. Report Cohen's kappa. The coding sheet is a CSV emitted
by `gricean.coding.export` and read back by `gricean.coding.load`.

### 5.2 Line 2: three-way continuation scoring, no judge

For each dissociation item (controls are excluded, since their readings
coincide), three continuations:

- `literal`: hand-written ("Yes, I can.");
- `intended`: hand-written ("Brazil became independent in 1822.");
- `neither`: **sampled, cross-sourced** (see 5.3).

Each model scores each continuation given the prompt: sum of token
log-probabilities, length-normalized (per token). Reported:

- **Headline:** log-odds of `intended` over `literal`, per category × stage,
  with item-level confidence intervals.
- **Floor check:** normalized three-way probability share. Guards against a
  headline built on two negligible probabilities when the model's mass sits
  on the degenerate continuation.

### 5.3 The `neither` continuation

Not hand-written. Sampled from a base model *other than the one being scored*
to avoid the circularity of scoring a model against its own output:

| Scored family | `neither` sampled from |
|---|---|
| OLMo 2 ladder | Qwen2.5-1.5B base |
| Qwen robustness pair | OLMo-2-0425-1B base |
| SmolLM2 self-run | OLMo-2-0425-1B base |

Sampling: temperature 1.0, three samples per item, truncated at 20 tokens or
first line break. The script keeps the sample with lowest token overlap with
the `literal` and `intended` continuations. The author does a quick pass over
`stimuli/neither/*.jsonl` to reject any that accidentally answer the
question. Seeds, revision hashes and settings are stored with the file; the
set is frozen alongside the hand-written stimuli.

### 5.4 Predictions stated in advance

- Intent rate and log-odds rise sharply base → SFT, rise or hold at DPO, stay
  flat at Instruct.
- Controls stay high across stages.
- Cue ablation degrades base more than tuned models.
- Self-run: clear but smaller effect than OLMo SFT; present on indirect
  request and pattern continuation, weaker on malformed and underspecified;
  absent in the format-only control; grows with data scale.

## 6. What the paper reports

- Table: intended / literal / neither rates by category × stage (Line 1).
- Table: mean log-odds and three-way shares by category × stage (Line 2).
- Figure: both measures across the four OLMo stages.
- Table: cue-ablation results (intended rate by version × stage).
- Table: robustness pair; table: self-run fine-tune with controls.
- Printed examples: for selected items, the greedy response at each stage
  side by side. Portuguese examples from the mirrored subset.

## 7. Self-run fine-tune

Purpose: a reader can enact the mechanism, not only observe checkpoints.

- **Base:** `HuggingFaceTB/SmolLM2-360M` (Apache 2.0).
- **Data:** `databricks/databricks-dolly-15k` (CC BY-SA), plain-instruction
  categories only (drop context-dependent ones), ≈8,000 pairs. Enacts SFT
  only.
- **Method:** LoRA via `peft` on the same Hugging Face `transformers` backend
  used for inference (portable; adapters load directly into the evaluated
  model), 1–2 epochs, fixed seed, all hyperparameters in
  `configs/finetune.yaml`. Prompt format for training and evaluation is a
  fixed Alpaca-style template, since the base tokenizer has no chat template. Adapter weights committed under
  `adapters/` so replicators can skip training.
- **Comparison:** SmolLM2 base vs base + adapter on the full stimulus set,
  both measurement lines, same table format as OLMo.

Controls:

1. **Format-only:** LoRA on the same prompts with responses shuffled across
   items. Learns chat format and "answer now" without any prompt–response
   relation. Intent-tracking here would mean the mechanism is format, not
   intention.
2. **Data-scale curve:** adapters on 500, 2,000 and 8,000 pairs.

Training runs as a CLI (`gricean finetune`) so it can go in the background.

## 8. Repository layout and data flow

```
gricean-devices/
  stimuli/              hand-written YAML per category; pt/ mirror; neither/ samples; CHECKSUMS
  configs/              models.yaml (repo, revision hash), generation.yaml, finetune.yaml
  src/gricean/
    stimuli.py          load, validate, freeze
    models.py           load registered checkpoint; apply template or raw
    generate.py         greedy + sampled responses -> data/responses/
    neither.py          cross-sourced degenerate continuations -> stimuli/neither/
    logprobs.py         three-way scoring -> data/logprobs/
    coding.py           export coding sheet; load coded sheet; kappa
    finetune.py         LoRA runs (full, format-only, scale curve) -> adapters/
    report.py           tables and figures -> paper/
    cli.py              one subcommand per stage
  notebooks/            marimo notebooks (see 8.2)
  data/                 JSONL outputs, one file per model x stimulus version; committed
  adapters/             committed LoRA weights
  paper/                tables (CSV + LaTeX), figures, printed example blocks
  tests/
  Makefile              make all; expected runtime documented in README
```

### 8.1 Flow

freeze stimuli → sample `neither` → generate responses (every model × version)
→ score log-probs → export coding sheet → author codes → report.

Each step is idempotent: it hashes its inputs and skips work whose inputs are
unchanged, so adding one stimulus reruns only that stimulus.

### 8.2 Marimo notebooks as the front end

Rule: **notebooks display, the package computes.** Every function a notebook
calls lives in `src/gricean/` and is unit-tested there. Notebooks are plain
`.py` files (git-friendly, runnable headless with `marimo run`). A test
imports each notebook to catch drift.

Heavy steps never run implicitly on reactivity: every model call writes to
disk, notebooks read cached files, and a heavy step runs only behind an
explicit "run" button (with `mo.persistent_cache` on top). Training stays a
CLI; the notebook watches the loss log.

| Notebook | Shows |
|---|---|
| `01_stimuli.py` | browse frozen set by category and version; checksums |
| `02_responses.py` | pick an item, greedy response at every stage side by side; run generation for a model |
| `03_logodds.py` | three-way shares, per-category tables, ladder figure |
| `04_finetune.py` | loss curves; base vs adapter vs format-only vs scale curve |
| `05_coding.py` | kappa, final tables, export to `paper/` |

### 8.3 Replication package

Lockfile (uv or pip-tools) pinning Python packages; `configs/models.yaml`
pinning model revision hashes; `make all`; README with expected runtime on a
16 GB Mac. Data files are committed, so tables regenerate without running any
model.

## 9. Validation

- Unit tests: stimulus validation, chat-template application, log-prob
  normalization, three-way share, kappa. Use a tiny stub model so tests run in
  seconds.
- Integration test: one stimulus through the full pipeline with the smallest
  model.
- Determinism check: greedy generation compared across two runs on the same
  machine. If MPS proves nondeterministic, the paper says so and reports
  sampled means only.

## 10. Out of scope

No benchmarks, no larger models, no API models, no mechanistic
interpretability, no LLM-as-judge. Portuguese is a generalization check, not a
parallel study. Prong (2) of the summary is argued, not simulated (§1).

## 11. Open items for the author

- Write the 48 + 12 English items and the Portuguese mirror.
- Recruit the second coder.
- Decide, after seeing results, whether the Instruct column stays in the
  paper.
