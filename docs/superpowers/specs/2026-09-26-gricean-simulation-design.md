# Design spec: computational simulation for "LLMs are Gricean machines"

Date: 2026-09-26, revised 2026-11-08, amended 2026-11-11, notes added 2026-11-12, 2026-11-14, 2026-11-15 and 2026-11-16, amended 2026-11-18
Status: revision 2 for author review
Paper: Pádua, "LLMs are Gricean machines", Revista Texto Livre 2027 dossier

## Revision log

**2026-11-18 (amendment): coder 2 is an agent, not a person.** The text of
§5.1 said "an independent coder" and §11 said "recruit the second coder",
which reads as a person; the author's design was an agent all along, and the
spec now says so. Coder 2 is a Claude Haiku model run from Claude Code, in a
fresh context, with the codebook and the rows of the blind sheet and nothing
else (§5.1, step 15). The author chose a small model on purpose: coder 2
stands for an ordinary coder, not for an expert one. Three passages change
with it. Constraint 2 of §3 (no paid APIs, no subscription-only models)
applies to the models the experiment is about, not to the coding of their
outputs. "No LLM-as-judge" in §10 holds for every code that feeds a table:
those are the author's alone, and coder 2's codes feed the agreement measure
only. The open item of §11 is closed. This amendment was written before
coder 2 gave any code.

**2026-11-18 (note): coder 2 has coded.** Eight Haiku agents coded the 335
rows of the blind sample on 2026-11-18 (§5.1, step 15). On `intent`, Cohen's
kappa between the author and coder 2 is 0.775, with the same code on 290
rows (86.6%). On `correctness`, by the measure the author then fixed (step
16), kappa is 0.855 on the 290 rows with the same `intent` and 0.833 on all
335.

**2026-11-18 (note): codebook and sheet for coder 2.** The codebook is
`docs/codebook-coder2.md`; §5.1, step 14, says how it was compiled, which
four unwritten conventions the author confirmed, and why it carries no
example from the responses. Before any draw, the sheet for coder 2 was
changed so that it is blind in fact: the `row_id` it used to show began with
the model's name. It now has an opaque key, an order that does not follow
the model, and the two readings of each item. No sheet had been written for
coder 2 before this change, and coder 1's sheet is untouched.

**2026-11-16 (note): Line 1 is coded for the self-run ladder.** The author
coded the 552 greedy responses of the four adapters on 2026-11-15. The codes
were committed as entered, a third check flagged 40 of them, and the author
decided each one on 2026-11-16 (§5.1, steps 11 to 13). 26 rows differ from
the codes as entered. The rubric is unchanged; step 13 lists what the
decisions add to it. `paper/selfrun/` now holds the Line 1 tables, per
language. On the codes as they stand, for the 48 English dissociation items
in the `full` version, `intended` is 24 at Stage 1 under the Alpaca template
and 37, 36 and 37 for the adapters trained on 500, 2,000 and all pairs. The
format-only adapter is `intended` on none: all 48 are `neither`, and all 12
controls too. In the `embedded` version no rung is `intended` on any of the
24 items. Against §5.4: the effect of the adapters and its absence in the
format-only control are there; growth with data scale is not, between 500
pairs and all of them; `embedded` goes the same way as on the official
ladder. `correctness` is still in no table, and there is no second coder
yet.

**2026-11-15 (note): the self-run adapters are trained.** The four adapters
of §7 were trained on 2026-11-14 and 2026-11-15 with the settings recorded
below, one process each, in the order `n500`, `n2000`, `full`,
`format_only`. §7 now gives the measured times, memory and losses next to
the estimates made beforehand. No setting changed between the rehearsal and
the run.

**2026-11-14 (note, made before any adapter was trained): self-run
resources.** Two rehearsals of the 500-pair adapter ran out of memory, the
first at batch 4, the second at batch 2. The cause was MPS memory growing
with every new batch shape. §7 now records the measured figures and the three
settings used (batch 2 with accumulation 8, padding to multiples of 64
tokens, cache release); none changes the loss. The Dolly revision is pinned.
A third rehearsal completed and its adapter is the `n500` adapter.

**2026-11-14 (amendment, made after the first pass and the first check; it
clarifies the rubric): the `embedded` version.** The rubric had no stated
rule for a response to the `embedded` version that continues the frame story.
In the first check the author gave, as the reason for three groups, that
every `embedded` response is `neither`. That rule was not applied to the
whole sheet, and after discussion the author replaced it with a narrower one
(§5.1, step 7): the task carried out is `intended`, the item's own literal
reading is `literal`, a continuation of the frame story alone is `neither`.
What was known when the rule was fixed: the author had coded every row. On
those codes the tuned models are `intended` on 24 of 24 English dissociation
items in `full` and in `stripped`, and on 0 to 2 of 24 in `embedded`, which
goes against the §5.4 prediction for that version. A second check flagged 38
`embedded` rows against the narrower rule; the author decides them.

**2026-11-14 (note, made after the first pass of coding and before any
revision): consistency check.** Coder 1 finished the first pass on
2026-11-14. It is committed as entered (`5d265c0`). A check for internal
consistency then flagged 71 rows in which the same or nearly the same
response carries different codes. Coder 1 decides each flagged row; every
revision is a diff against the first pass and carries a reason. The procedure
is at the end of §5.1. The rubric is unchanged.

**2026-11-12 (note, made before any response was coded): coding
interface.** Coder 1 enters the codes through a temporary web page, described
at the end of §5.1. The page is a convenience for reading the rows and for
avoiding typing errors. It is not part of the experiment: it is not in this
repository, the rubric is unchanged, and the record remains
`data/coding/coder1.csv`.

**2026-11-11 (amendment to rev. 2, made after generation and scoring and
before any response was coded): Line 1 codes a subset.** Unlike the
amendments below, this one changes the design.

- Line 1 hand-codes the greedy response of each model's main condition
  instead of every generated response (§5.1): 828 of the 4,608 responses of
  the first run, and 1,380 of 6,816 once the self-run adapters are added.
  Reason: one coder cannot code several thousand rows.
- The subset is fixed by design variables only (decoding mode and pass). No
  row is chosen at random or by its content, and every item is still coded
  at every rung.
- What was known when the rule was fixed: the author had seen descriptive
  means of Line 2 and had read greedy responses for some items in the
  notebook. No response had been coded.
- The three sampled responses per cell stay in `data/responses`, uncoded.
- The raw template control of the tuned models (§3.4) is reported from
  Line 2 only (§6).
- Empty responses are coded `neither` by rule, and the second coder's sample
  is drawn from the hand-coded rows (§5.1).

**2026-11-11 (amendments to rev. 2, made before any scored run).** These
record what was done; none changes the design.

- §4 states how the stimulus set was actually written. It was not
  hand-written by the author alone: 24 pattern items and 23 further English
  items were drafted with Claude Code, and the continuations were
  length-matched.
- §4.5 records the rev. 1 dry run on two seed items and the deletion of its
  rows before the freeze.
- §5.3 corrects the truncation rule for `neither` samples (leading line
  breaks are skipped).

**2026-11-08 (rev. 2): one documented lineage.** Checking the training-data
documentation of the models in rev. 1 showed that every 2024-era "base"
checkpoint in the design had seen instruction data before any post-training:
the released `OLMo-2-0425-1B` includes a 50B-token mid-training stage whose
mix is 16.6% decontaminated FLAN (≈8B tokens of instruction–response pairs;
OLMo 2 report, Tables 5 and 13); the Qwen2 report states that "high-quality
multi-task instruction data is integrated into the Qwen2 pre-training
process", and Qwen2.5 adds synthetic pretraining data generated by
Qwen2-72B-Instruct. Phase 1 of the hypothesis (§1) therefore needs a
checkpoint whose *absence* of instruction data is documented, not assumed.
Changes:

- Phase-1 stand-in is the final **stage-1** OLMo 2 1B checkpoint
  (`stage1-step1907359-tokens4001B`, 4.0T tokens, pretraining mix only).
  The released base becomes a gradient rung with a known 8B-token dose of
  instruction data (§3.1).
- The Qwen robustness pair is retired from the paper (§3.2).
- The self-run fine-tune moves from SmolLM2-360M onto the same OLMo stage-1
  checkpoint, so the whole paper rests on one lineage (§7).
- The `neither` source is chosen by a documentary criterion (no instruction
  data in training), which selects Pythia-1.4b for every ladder (§5.3).
- GPT-3 → InstructGPT (Ouyang et al. 2022) is the paper's historical anchor
  for the transition, quoted, not run (§2).

## 1. Purpose

The paper claims that instruction fine-tuned LLMs operate as Gricean devices:
they respond to what a user's prompt is *for* (the intended meaning) rather
than to what it literally says. The hypothesis has two phases (author's
formulation, rev. 2):

- **Phase 1.** Trained on trillions of tokens but with no instruction
  training, a model has vast knowledge and text-completion ability yet cannot
  take a simple human turn, because it does not represent what the speaker
  wants by saying what they said.
- **Phase 2.** Train the same model to follow instructions, even under the
  weight-change caution built into PPO-like methods, and it behaves like a
  human co-interactant.

The first documented instance of this transition is GPT-3 → InstructGPT
(Ouyang et al. 2022, Fig. 1); the simulation reproduces it on an open lineage
where both phases are documented (§3). Scope note (added rev. 2): both
measurement lines are behavioural. They show whether output tracks the
speaker's intended reading; the step from intent-tracking to "theory of mind"
is the paper's argument and is stated as such, not as a result.

The simulation must produce controlled
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
   This applies to the models that are the object of the experiments
   (clarified 2026-11-18). It does not apply to the coding of their outputs:
   coder 2 of §5.1 is a hosted model. Its codes are committed, so every
   table and the agreement measure regenerate without it.
3. Full replication package: pinned model revisions, pinned Python
   dependencies, committed intermediate data, one-command regeneration of
   tables.

GPT-3 is excluded as a subject: `davinci` and `text-davinci-003` were removed
from OpenAI's API in January 2024, `gpt-3.5-turbo` is a chat endpoint that
never exposed continuation log-probabilities, and all were paid and
un-pinnable. GPT-3 → InstructGPT remains the paper's historical anchor: the
published pairs in Ouyang et al. (2022) are quoted, and the pattern-
continuation category (§4.1) uses their Fig. 1 prompt. The "legacy"
comparison that is actually run is the OLMo 2 stage-1 checkpoint (§3.1).

A further constraint (rev. 2): every checkpoint standing in for phase 1 must
have **documented** training data with no instruction-tuning datasets. This
rules out every 2024-era released "base" model checked so far (see Revision
log) and is the reason the design uses a pre-mid-training checkpoint.

The criterion is literal (author's decision, rev. 2): no training on
prompt–response datasets. It does not exclude web text that *resembles* answers. OLMo-mix-1124 is mostly
DCLM-Baseline, whose quality filter is a fastText classifier trained with
OpenHermes-2.5 and r/ExplainLikeImFive as positive examples (DCLM, Li et al.
2024), so stage-1 web data is tilted toward explanatory prose. The paper
discloses this. It is phase-1 material in the hypothesis' own terms: GPT-3
also saw forum Q&A, and what phase 2 adds is training on the turn structure,
not exposure to answer-like text.

## 3. Models

### 3.1 Main ladder: OLMo 2 1B (AI2, April 2025 release)

Same architecture, tokenizer and pretraining at every stage; Apache 2.0;
every training mix (OLMo-mix-1124, Dolmino-mix-1124, Tülu 3) public, so each
rung's exposure to instruction data is a documented quantity.

| Stage | Checkpoint | Adds | Instruction data seen | Role in argument |
|---|---|---|---|---|
| Stage 1 | `allenai/OLMo-2-0425-1B` @ `stage1-step1907359-tokens4001B` | 4.0T tokens of pretraining mix | none documented | **phase 1**; GPT-3 analogue |
| Base | `allenai/OLMo-2-0425-1B` @ `main` | 50B-token mid-training (Dolmino) | ≈8B tokens FLAN (16.6% of mix) | exposure gradient, no post-training |
| SFT | `allenai/OLMo-2-0425-1B-SFT` | supervised instruction tuning (Tülu 3) | full SFT set | first Gricean stage |
| DPO | `allenai/OLMo-2-0425-1B-DPO` | preference optimization on human comparisons | + preference pairs | RLHF analogue; **phase 2** endpoint |
| Instruct | `allenai/OLMo-2-0425-1B-Instruct` | RL with verifiable rewards (math/code correctness) | + RLVR | **control column** |

The headline contrast is Stage 1 → DPO. The Base rung is **descriptive**: it
separates "saw some instruction-formatted text during pretraining" from "was
trained to follow instructions". Stage 1 → Base is not FLAN alone: mid-training
also anneals the learning rate and adds curated math, encyclopedic and Q&A
data (OLMo 2 report §2.4.2), and for the 1B model the released `main` is a
single 50B-token run (report, Table 9 caption), so the step is reported as one
column, or as a dose note under the figure, never decomposed. No intermediate
mid-training checkpoints are run. Stage-1 weights are the same repository
under a different revision; the tokenizer is identical and has no chat
template, so raw prompting is unchanged. Stage-1 safetensors are fp32 (5.9 GB
download, ≈3 GB in bf16 at run time).

The Instruct stage trains against correctness checkers, not human judgements
of adequacy, so it is not part of the Gricean mechanism. It is kept as a
control: the prediction is that intent-tracking rises at SFT and DPO and stays
flat at Instruct, localizing the effect in the stages that use human feedback
about adequacy. If the column is uninformative it is dropped from the paper
but kept in the repo.

### 3.2 Robustness pair (retired in rev. 2)

`Qwen/Qwen2.5-1.5B` cannot stand in for phase 1: its own technical reports
document instruction data and instruct-generated synthetic data in
pretraining, with undisclosed volume. The pair is removed from the paper. The
registry entries stay in `configs/models.yaml` as an optional contrast run
(a "base" with undisclosed instruction exposure), not reported by default.
Generalization beyond OLMo is carried by the self-run fine-tune (§7), which
enacts the transition on the stage-1 checkpoint with author-controlled data.

### 3.3 Self-run fine-tune (enacted mechanism)

See §7.

### 3.4 Prompt formatting

Base models receive raw text. Tuned models receive their chat template. This
is part of the mechanism, not a confound, but tuned models are *also* run on
raw text (no template) so the paper can show the template alone does not
explain the effect. The raw passes are generated and scored; the paper
reports them from Line 2 (§5.2), and they are not hand-coded (§5.1).

The raw-text control applies to the official ladder (§3.1) only. The self-run
ladder (§7) is exempt (author's decision, rev. 2): it already holds format
constant by running its baseline and every adapter under the same Alpaca
template, and the format-only adapter is its format control. Running the
five self-run conditions raw would add ≈2,700 responses to hand-code and ≈6 h
of generation for a question §7 answers by design.

### 3.5 Inference stack

Python, Hugging Face `transformers` on Apple MPS (CPU fallback elsewhere),
half precision. Every checkpoint pinned to a revision hash in
`configs/models.yaml`.

## 4. Stimuli

One YAML file per category and language under `stimuli/`. Each item records: `id`, `category`, `prompt`, `literal_reading`,
`intended_reading`, `literal_continuation`, `intended_continuation`,
`language`, `version` (see cue ablation).

**How the set was written.** The 78 items (60 English, 18 Portuguese) were
written by the author together with Claude Code (Anthropic's coding
assistant), in four parts:

- *Pattern items (24).* The first dissociation item and the first control of
  every category file, in both languages, were written with Claude Code
  during the design phase, as the pattern for the rest. The English ones are
  this spec's canonical examples (§4.1; the pattern-continuation one is from
  Ouyang et al. 2022, Fig. 1). The Portuguese ones are native drafts, which
  the author kept unchanged.
- *Author's items (31).* The author wrote 25 English items and the second
  dissociation item of every Portuguese file (6).
- *Drafted items (23).* The remaining English items were drafted by Claude
  Code, with the author's items as templates, and reviewed by the author:
  `en_mi_03`–`08`, `en_pc_05`–`08`, `en_ur_04`–`08`, `en_rq_05`–`08`, and the
  second control of those four categories.
- *Editing pass.* Claude Code corrected typos and YAML errors in the author's
  items; the deliberate damage in malformed-instruction prompts was left
  alone. With the author's approval it also replaced `en_ir_05`, rewrote
  `en_ei_07` so that the prompt quotes its article, and reworded `en_ei_02`,
  `en_ei_c02`, `en_rq_04`, `pt_ur_02` and the intended continuations of
  `pt_mi_02` and `pt_ir_02`.

The author reviewed every item before the freeze (§4.5).

**Length of the continuations.** Line 2 normalizes log-probabilities per
token (§5.2), so a long `intended` continuation scored against a short
`literal` one would shift the log-odds at every rung for a reason unrelated
to the hypothesis. Each continuation is therefore one short sentence; 20
intended continuations were cut to their first sentence or clause. Lengths
were counted with the OLMo tokenizer alone, with no model loaded. After the
cut, 3 of the 60 dissociation items have an `intended` continuation at least
three times as long as the `literal` one (maximum 3.4×), down from 19
(maximum 18.5×).

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
Stimuli are frozen and committed before any model is run. The set was frozen
on 2026-11-10 (combined checksum `157edf6d…`).

One qualification: the rev. 1 pipeline dry run generated responses for two
seed items, `en_ir_01` and `en_ir_c01`, before the rest of the set existed.
Their prompts are unchanged since. The dry-run rows were deleted before the
freeze, so every committed row was generated under the frozen set.

## 5. Measures

Two independent lines, both from the frozen stimuli.

### 5.1 Line 1: generated responses, human-coded

For every model stage × stimulus version: one greedy response (printed in the
paper) and three sampled responses (temperature 0.7, fixed seeds). Max 150
new tokens. An empty greedy response is printed as such
(`⟨empty: model emitted end-of-sequence⟩`); it is a finding (the base model
treats the prompt as complete), not a gap.

**What is coded** (amended 2026-11-11, before any coding): the greedy
response of each model's main condition, which is the templated pass for a
tuned model, the raw pass for a base model, and the untemplated `embedded`
version for both. That is one response per item × version × model, so the
item is the unit of every rate. Not coded: the sampled responses, which
repeat the same item at temperature 0.7 and were meant to feed the counts,
and the raw template control of the tuned models, which Line 2 reports
(§5.2, §6). Both stay in `data/responses` and can be coded later.

An empty or whitespace-only response is coded `neither` / `na` by rule when
the sheet is written; the response text stays in the sheet, so these rows
can be counted apart.

Rubric, per response:

- **Intent** (primary): `literal` / `intended` / `neither`. `neither` covers
  refusals, off-topic text and degenerate loops. It is reported as its own
  count, never folded into `literal`.
- **Correctness** (secondary, only if the task was addressed): `correct` /
  `incorrect`. Kept separate so the paper can distinguish intent-tracking from
  knowledge.

Coder 1: the author. Coder 2: an independent coder on a 25% random sample
of the rows coder 1 codes by hand (rule-coded empty responses are left out,
since no one judges them), blind to model stage. Report Cohen's kappa. The coding sheet is a CSV emitted
by `gricean.coding.export` and read back by `gricean.coding.load`.

**Coder 2 is an agent (stated 2026-11-18, before it gave any code).** It is
a Claude Haiku model, run from Claude Code in a fresh context that holds the
codebook and the rows of the blind sheet and nothing else; step 15 below is
the procedure. Three things follow.

- *Why a small model.* Coder 2 stands for an ordinary coder who applies the
  written codebook, not for an expert. The author chose Haiku over the more
  capable Claude models for that reason.
- *What it is for.* Coder 2 is a control on a control. Line 1 is already
  checked by Line 2, which uses no judge (§5.2); coder 2 checks the codes of
  Line 1 themselves. It is not an object of the experiment and gives no
  result of its own. Every code that feeds a table is the author's; coder
  2's codes feed the agreement measure and nothing else.
- *How it is reported.* Kappa is the agreement between the author and a
  language model that applied the author's written codebook, and the paper
  says so and names the model. It is not agreement between two human coders.
  The paper also states that this coder is an instruction-tuned model, the
  kind of system the paper is about.

**Coding interface (a convenience, not part of the experiment; note of
2026-11-12).** Coder 1 enters the codes through a temporary web page instead
of a spreadsheet. The page is not in this repository and nothing in the
replication depends on it: the record is `data/coding/coder1.csv`, and the
page can be deleted once the codes are in the sheet. The rubric above is the
only coding rule. What the page does:

- It shows one row at a time: `stimulus_text`, the response in full, and the
  item's two readings with their example continuations, as written in the
  frozen stimulus file. In a response it highlights the template markers
  (`### Instruction:` and the like); it hides no text.
- It takes `intent` and `correctness` from buttons that carry the values
  `gricean.coding.load_sheet` accepts, so a code cannot be mistyped.
- It orders rows by stimulus and version and, within those, by a hash of
  `row_id`, so a row's position does not give its model away. `model`,
  `template` and `prompt_text` are hidden unless the coder turns them on, and
  a code entered while they are visible is marked, so that the number of such
  codes can be reported. This does not make coder 1 blind: the author knows
  the design, and some responses show their stage by themselves (the Alpaca
  run-on text, for one). Blindness is claimed for coder 2 only.
- It shows how many rows are coded and no running tally of the codes.
- It shows the rule-coded empty responses locked.
- One shortcut, off by default, fills `na` when the coder chooses `neither`
  and correctness is still blank. The coder can change it.
- Flags and notes stay in the page. Only `intent` and `correctness` are copied
  into the sheet, keyed by `row_id`. The transfer is checked with `git diff`
  on the sheet, which must show changes in those two columns only. The
  reasons for revisions are the one exception: they go into the repository
  (next paragraph).

**First pass, consistency check and revision (note of 2026-11-14).**

1. *First pass.* Coder 1 coded the 787 rows between 2026-11-12 and
   2026-11-14, every one with the model hidden. The codes were copied into
   `data/coding/coder1.csv` and committed as entered, before any check
   (`5d265c0`, "First-pass codes for Line 1"). That commit is the frozen first
   pass.
2. *Check.* Claude Code then compared the coded responses for internal
   consistency only: rows whose responses are identical or nearly identical
   and whose codes differ. It did not judge whether a code fits the rubric,
   and it proposed no code. It compared the text of every pair of responses
   to the same item, and of responses with the same opening across items;
   it then read the opening of every `embedded` row and of every row of the
   three untuned passes (base, stage 1, stage 1 with the Alpaca template).
   The `full` and `stripped` rows of the tuned models were compared by text
   similarity only. The check was not blind: Claude Code had the model of
   each row before it. It used the text alone, and the groups are described
   to the coder without naming a model.
3. *Result.* 71 rows in 17 groups, plus 5 rows shown for comparison. They are
   listed in `data/coding/coder1_check.csv` with the reason each group was
   flagged and the first-pass codes. That list was committed before any
   decision. A by-product: of 59 sets of character-identical responses to the
   same item, 58 had received identical codes.
4. *Decision.* Coder 1 goes through the flagged rows in the coding page, with
   the model hidden by default as before, and either keeps the first code or
   changes it. The page shows the rows of a group side by side, with the
   reason they were flagged and the first-pass code of each. The coder
   records a rule adopted for a whole group, a reason for a single row, or
   both.
5. *Record.* The revised codes are copied into `coder1.csv` in a later
   commit, so `git diff 5d265c0 -- data/coding/coder1.csv` shows exactly what
   the revision changed. The same commit fills the decision columns of
   `coder1_check.csv` (`decision`, `revised_intent`, `revised_correctness`,
   `group_rule`, `reason`). A row revised without having been flagged is
   added to that file with its reason. `make report` reads the sheet as it
   stands, so the tables and the kappa use the revised codes; the first pass
   stays available at `5d265c0`.
6. *Outcome of the first check.* The author revised 35 of the 71 flagged rows
   and kept 36, and revised one comparison row with its group (`ea30344`). A
   re-run of the comparison on the revised codes found one pair of identical
   responses left with different correctness (`en_ir_02`, base, `full` and
   `stripped`). At the author's request Claude Code entered the code of the
   `full` row, following the reason the author had given for the `stripped`
   one (`74d5280`; the row is in `coder1_check.csv` with the role
   `added_after_recheck`). 37 rows differ from the first pass at that point.
7. *Rule for the `embedded` version* (amendment of 2026-11-14, see the
   Revision log for what was known). A response that carries out the task of
   the embedded prompt is `intended`. A response that gives the item's own
   literal reading is `literal`. A response that only continues the frame
   story (the office, the rain, the printer) is `neither`. Correctness stays
   a separate code. The rule replaces the one recorded for groups `g07`,
   `g08` and `g16` of the first check, which said that every `embedded`
   response is `neither`. It leaves one case for the author to settle in the
   second check: a response that continues the narrative and carries out the
   embedded task inside it.
8. *Second check.* Claude Code read every hand-coded `embedded` row (158)
   against that rule and flagged 38 rows in 5 groups, with 11 rows for
   comparison. They are listed in `data/coding/coder1_check2.csv`, committed
   before any decision, with the first-pass codes and the codes at the start
   of this check. Unlike the first check, this one rests on a judgment of
   what a response does. Each flag therefore states that judgment and no
   code, and the codes remain the author's. The check was not blind. The
   author decides in the coding page as before, and the record is again one
   commit of `coder1.csv` and of the decision columns of `coder1_check2.csv`.
9. *Outcome of the second check.* The author revised 32 of the 38 flagged
   rows and kept 6 (`b8ed514`). For the case step 7 left open, the author's
   rule for the group reads "Changed most to intended per the narrow rule":
   a response that carries out the embedded task inside the continued story
   is `intended`. 68 rows differ from the first pass at that point
   (`git diff 5d265c0 -- data/coding/coder1.csv`). The rule moves the result
   it was fixed after: on the 24 English dissociation items of the ablated
   categories, `intended` in the `embedded` version goes from 0, 2 and 2
   (SFT, DPO, Instruct) before the second check to 1, 6 and 6 after it, and
   stays at 0 for Stage 1 and Base. The paper reports the `embedded` column
   with this history.
10. *After the second check.* Claude Code read the revised `embedded` rows
    once more and reported three points; the author decided each, and Claude
    Code entered the four codes at the author's request (rows with the role
    `after_recheck` in `coder1_check2.csv`, in the commit that follows
    `b9fcd06`). `en_ei_01`, DPO and Instruct, go back from `literal` to
    `intended`, with the other rows that carry out the task inside the story.
    `en_ei_08`, Base, goes from `literal` to `neither`, with the two rows of
    the item that loop on the same kind of sentence. `pt_ir_01`, DPO, becomes
    `incorrect`: its answer is wrong. 67 rows then differ from the first
    pass, and `intended` in the `embedded` version stands at 1, 7 and 7 of 24
    for SFT, DPO and Instruct.
11. *Second batch (2026-11-15).* After the self-run adapters were trained
    and run (§7), the sheet was written again: 1,380 rows, the 828 of the
    first batch with every cell unchanged and 552 new ones, the greedy
    responses of the four adapters (138 each). The new rows were committed
    blank (`5505381`) before any of them was coded. None is empty, so none
    is coded by rule. The author codes them in the same page, under the
    rubric as it stands after steps 7 to 10; the page still hides the model,
    though the coder knows that the open rows come from the adapters. The
    rows of the first batch are shown there for comparison and cannot be
    edited. A consistency check of this batch, if one is made, follows
    steps 2 to 5.
12. *Third check (2026-11-15), on the second batch.* The author coded the
    552 rows on 2026-11-15; they were committed as entered (`3ff8884`)
    before anything was checked. At the author's request Claude Code then
    read all 552 against the rubric as it stands and against the first
    batch, looking for two things: the same text, or the same kind of text,
    under different codes, and codes that look like slips of attention.
    It flagged 40 rows in 13 groups (`data/coding/coder1_check3.csv`,
    groups `g23` to `g35`, with 25 rows shown for comparison, 5 of them
    from the first batch). The list was committed before any decision and
    proposes no code. Five groups are single rows or pairs with identical
    or near-identical text; the others are judgments about kinds of
    response (an invented figure, a wrong fact, a city described without
    its weather, the narrative of the prompt repeated). The author decides
    each row in the page, as in steps 3 to 5, and the decisions go into the
    same file.
13. *Outcome of the third check (2026-11-16).* The author revised 26 of the
    40 flagged rows and kept 14 (`bad2684`), with a rule for nine groups and
    a reason for five rows, all in `coder1_check3.csv`. What the decisions
    add to the rubric, in the author's terms:
    - A response unrelated to both readings is `neither`, and it is
      `correct` when it is well formed as language (`g23`, `g24`, `g28`).
      This is how the one sentence that the format-only adapter gives to
      almost every prompt is coded.
    - A response is `incorrect` when its information is obviously wrong. It
      stays `correct` when the information only lacks grounding, such as a
      definite waiting time or ticket price (`g31`, `g32`).
    - "Squarer root" is `incorrect` wherever it appears (`g25`).
    - A response that never reaches what was asked, such as the city
      described without its weather, is `neither` (`g30`).
    - In the `embedded` version, a loop that takes up the wording of the
      embedded prompt, or follows its pattern, is `literal` (`g33`, `g34`).
      A loop without that link, or a repetition of the narrative that
      continues nothing, is `neither` (`g34`, `g35`).

    One row shown for comparison (`en_ir_08`, `full`) was changed in the
    page as well. Claude Code then read the revised codes against these
    rules, a final audit. That row does talk about the weather, like the
    other answer to the same prompt, which is coded `intended`; the author
    restored its first code (`6d2fcf3`, role `after_recheck` in the file).
    The audit also asked again about `pt_rq_c01` ("Depends on the bank."),
    kept `literal` with a reason that concerns correctness only. The author
    confirmed `literal`. It is the only response to a control item coded
    `literal`. The audit found no other row at odds with the rules above,
    no identical text under two codes, and no `full` and `stripped` pair of
    one model with the same text and different codes. 26 rows of the second
    batch differ from the codes as entered
    (`git diff 3ff8884 -- data/coding/coder1.csv`).

    One convention holds in the batch without having been written as a
    rule, and coder 2's instructions need it: an answer in English to a
    Portuguese prompt is `incorrect` when it is coded `intended` or
    `literal` (one exception, a translation item whose answer gives the
    Portuguese word), and language is not considered under `neither`.
14. *Codebook and sheet for coder 2 (2026-11-18).* Coder 2 codes from
    `docs/codebook-coder2.md`, which Claude Code compiled from the rubric,
    from steps 7, 9, 10 and 13, and from the rules and reasons of the three
    check files. The author confirmed four conventions that held in the codes
    without having been written as rules: a response that does what a control
    item asks is `intended`; `na` is not used by hand, so every hand-coded
    row is `correct` or `incorrect`; under `neither`, `correct` means well
    formed as language and sensible in itself, relevance aside; and the
    language convention of step 13. The author also fixed the sense of
    "obviously wrong": wrong for an ordinary reader, not only for a
    specialist in the subject. The occasion was `en_mi_04`, `full`, whose
    response goes on to name the wrong mountain as the second tallest; its
    code, `correct`, stands.

    At the author's instruction the codebook gives no example taken from the
    responses and does not quote the check files. An example fixes the code
    of every sampled row with the same text, and one sentence alone accounts
    for 104 of the 1,339 hand-coded rows; agreement on such rows would not
    test the rule. The codebook names no model and no result.

    The sheet for coder 2 (`gricean sheet --coder2`) differs from coder 1's
    in three ways, all settled before the draw. Its `row_id` is an opaque
    key, a hash, because the `row_id` of the full sheet begins with the name
    of the model; `coder2.key.csv` leads back to it (`source_row_id`). Its
    rows are ordered by item and version and then by that key, so that the
    position of a row does not follow its model. And it carries the two
    readings and the example continuations of each item, which coder 1 saw
    in the coding page. The draw is as stated above: 25% of the rows coded
    by hand, seed 0.
15. *Coder 2's run (2026-11-18).* The procedure was fixed before the run.
    `gricean coder2 --batches DIR --n 8` writes the blind sheet as eight
    text files in a folder outside the repository, in the order of the
    sheet, without splitting an item: the codebook asks for the same codes
    on the same text within an item, so all rows of an item must be read
    together. A batch shows the columns of the sheet, the bookkeeping ones
    (`mode`, `seed`, `coder`) and the codes left out, and comes with a form
    that lists its `row_id`s with both codes blank. The folder also holds a
    copy of `docs/codebook-coder2.md`. Each batch goes to a separate agent
    (Claude Code's `haiku` setting) with the brief in
    `docs/coder2-brief.md`, the same for all but the batch number. The
    agents share nothing: eight fresh contexts, so no judgement carries
    from one batch to the next. A human coder would carry it. The
    difference can only make coder 2 less consistent from batch to batch.

    Blindness: the folder holds no key, no code of coder 1 and no model
    name, and the brief tells the agent to open no other file, to run no
    code and to write no rule that assigns codes. Nothing enforces this.
    Each agent reports the files it opened.

    `gricean coder2 --enter DIR` copies the forms into `coder2.csv`. It
    writes nothing unless the forms hold every row of the sheet once and
    every code is a valid one. A form that fails goes back to the agent
    that filled it, with the reason; no one else enters or changes a code
    of coder 2. The sheet is committed as entered, and doubts the agents
    wrote down are kept in `data/coding/coder2_notes.md`.

    *Outcome (written after the run).* The eight agents ran on 2026-11-18,
    about a minute each, and coded 43, 42, 42, 43, 43, 41, 40 and 41 rows.
    Each reported that it opened its codebook, its batch and its form and no
    other file. Every form passed the checks at the first entry, so none
    went back to its agent. All eight wrote notes: 70 of the 335 rows carry
    a doubt. The sheet is committed as entered (`ff28c89`), and no code was
    changed afterwards. `na` was not used.

    On `intent` the two coders give the same code to 290 of the 335 rows
    (86.6%), and Cohen's kappa is 0.775 (`paper/*/kappa.txt`; the number is
    the same in both folders, because it is computed on the whole sample
    and not per ladder). Coder 1 has 170 `intended`, 47 `literal` and 118
    `neither` on these rows; coder 2 has 177, 48 and 110. The largest cell
    of disagreement is 13 rows that are `neither` for coder 1 and
    `intended` for coder 2.
16. *Agreement on `correctness` (2026-11-18).* The author fixed the measure
    after coder 2's codes were entered and before anyone compared the two
    coders on `correctness`. It is reported twice. The main number is
    Cohen's kappa on the rows where the two coders gave the same `intent`:
    what `correct` means depends on the intent (under `neither` it means
    well formed as language; elsewhere it means right), so only on those
    rows do the two coders answer the same question. The second number is
    kappa on all rows of the sample, given so that the first can be seen
    not to depend on what it leaves out. `gricean.coding.agreement`
    computes both, and `make report` writes them to `kappa.txt` under the
    kappa on `intent`, each with its number of rows and of equal codes.

    *Outcome (written after the first computation).* On the 290 rows with
    the same `intent`, the two coders give the same `correctness` to 269,
    and kappa is 0.855. On all 335 rows they give the same `correctness` to
    307, and kappa is 0.833.

### 5.2 Line 2: three-way continuation scoring, no judge

For each dissociation item (controls are excluded, since their readings
coincide), three continuations:

- `literal`: written with the item, §4 ("Yes, I can.");
- `intended`: written with the item, §4 ("Brazil became independent in 1822.");
- `neither`: **sampled, cross-sourced** (see 5.3).

Each model scores each continuation given the prompt: sum of token
log-probabilities, length-normalized (per token). Reported:

- **Headline:** log-odds of `intended` over `literal`, per category × stage,
  with item-level confidence intervals.
- **Floor check:** normalized three-way probability share. Guards against a
  headline built on two negligible probabilities when the model's mass sits
  on the degenerate continuation.

### 5.3 The `neither` continuation

Not written with the items. Sampled from a model that is (a) outside every ladder being
scored, to avoid the circularity of scoring a model against its own output,
and (b) **documented** to have no instruction data in its training, so the
distractor represents what a phase-1 model says rather than a second answer.
Criterion (b) is fixed before sampling and is independent of how the
candidate behaves on the stimuli; the seed-item pilot (Qwen2.5 base answered
the question in 2 of 3 samples, Pythia did not) is confirmation, not the
reason.

| Scored ladder | `neither` sampled from |
|---|---|
| OLMo 2 ladder (§3.1) | `EleutherAI/pythia-1.4b` |
| OLMo stage-1 self-run (§7) | `EleutherAI/pythia-1.4b` |

Pythia-1.4b is trained on the Pile only (components fully listed, 2020, no
instruction-tuning datasets) and is the open GPT-3-era model, which matches
the paper's anchor. One source for every ladder makes `neither` identical
across tables. The `neither` text is a fixed distractor applied to every rung
and never enters the headline log-odds (§5.2); it affects only the floor
check, which is reported as a share, never used to exclude items.

Sampling: temperature 1.0, **five** samples per item (raised from three in
rev. 2 because Pythia returns empty completions on some prompts; the scorer
rejects an empty `neither`), truncated at 20 tokens or the first line break
after the text starts. Leading line breaks are skipped: Pythia opens most
continuations with one, and cutting there discarded 143 of 300 samples in the
first run (2026-11-11, before any scoring), so the rule was corrected and the
set re-sampled with the same seeds. The
script keeps the non-empty sample with lowest token overlap with the
`literal` and `intended` continuations. The author does a quick pass over
`stimuli/neither/*.jsonl` to reject any that accidentally answer the
question. Seeds, revision hashes and settings are stored with the file; the
set is frozen alongside the stimuli.

### 5.4 Predictions stated in advance

- Intent rate and log-odds are low at Stage 1, move little or not at all at
  Base (8B FLAN tokens inside 4T is not instruction training), rise sharply
  at SFT, rise or hold at DPO, stay flat at Instruct.
- Controls stay high across stages.
- Cue ablation degrades Stage 1 and Base more than SFT, DPO and Instruct.
- Self-run on Stage 1: clear but smaller effect than official SFT; present on
  indirect request and pattern continuation, weaker on malformed and
  underspecified; absent in the format-only control; grows with data scale.

## 6. What the paper reports

- Table: intended / literal / neither rates by category × stage (Line 1).
- Table: mean log-odds and three-way shares by category × stage (Line 2).
- Table: template control, the same Line 2 measures for SFT, DPO and
  Instruct on raw text (`logodds_template_control`).
- Figure: both measures across the five OLMo rungs (Stage 1 → Instruct).
- Table: cue-ablation results (intended rate by version × stage).
- Table: self-run fine-tune on Stage 1 with controls.
- Historical anchor: the GPT-3 / InstructGPT pairs quoted from Ouyang et al.
  (2022) beside the Stage 1 / SFT responses to the same prompt.
- Printed examples: for selected items, the greedy response at each stage
  side by side. Portuguese examples from the mirrored subset.

## 7. Self-run fine-tune

Purpose: a reader can enact the mechanism, not only observe checkpoints.

- **Base:** the OLMo 2 1B **stage-1** checkpoint of §3.1, same revision. One
  lineage for the whole paper: the official ladder shows the transition as
  released, the self-run enacts it from the same documented-zero weights.
  (Rev. 1 used `HuggingFaceTB/SmolLM2-360M` for runtime; its pretraining data
  was never checked for instruction content and the two-lineage design was
  harder to defend.)
- **Data:** `databricks/databricks-dolly-15k` (CC BY-SA), the five
  context-free categories (open_qa, general_qa, brainstorming,
  classification, creative_writing), 10,544 pairs; closed_qa,
  information_extraction and summarization are dropped because they need a
  context passage. "Full" means all 10,544 (rev. 1 said ≈8,000; the code
  already used the full filtered set). Enacts SFT only.
- **Method:** LoRA via `peft` on the same Hugging Face `transformers` backend
  used for inference (portable; adapters load directly into the evaluated
  model), 1–2 epochs, fixed seed, all hyperparameters in
  `configs/finetune.yaml`. Prompt format for training and evaluation is a
  fixed Alpaca-style template (the stage-1 tokenizer carries no chat
  template, and the official ladder's template is not used, so the self-run
  tests instruction data, not OLMo's template). Adapter weights committed
  under `adapters/` so replicators can skip training.
- **Comparison:** Stage 1 vs Stage 1 + adapter on the full stimulus set, both
  measurement lines, same table format as the official ladder. The Stage 1
  baseline in this table is run under the Alpaca template so format is held
  constant; its raw run belongs to §3.1.
- **Resources (16 GB M3, bf16 LoRA r=16 on attention projections; measured
  2026-11-14, before any adapter was trained):** ≈3 GiB for the weights, and
  about 2.4 GiB more for each sequence at the 384-token cap, so one step
  peaks at 12.5 GiB at batch 4, 8.0 GiB at batch 2 and 5.5 GiB at batch 1.
  Batch 4 runs out of memory; the run uses batch 2 with accumulation 8, the
  same effective batch of 16. Two further measures, neither of which changes
  a loss term, keep memory flat on MPS, where it grows with every new batch
  shape: batches are right-padded to multiples of 64 tokens (six shapes
  instead of one per batch), and the cache is released after each optimizer
  step. Rehearsed on the 500-pair adapter: 11 minutes, about 3.6 GiB held
  between steps. At that rate (≈1.3 s per batch) the full adapter takes ≈4 h,
  format-only the same, 2,000 pairs ≈45 min, ≈9 h in total. The adapters are
  trained one per process, smallest first.
- **Outcome of the run (2026-11-14 and 2026-11-15):** 40 min for 2,000 pairs
  (250 optimizer steps), 3 h 33 min for the full adapter and 3 h 30 min for
  format-only (1,318 steps each), 7 h 54 min of training in total with the
  rehearsal. Memory held between steps stayed between 3.2 and 3.8 GiB at all
  142 readings. Training loss, first and last logged value (means over 20
  steps): 2.30 → 2.09 (`n500`), 2.33 → 2.08 (`n2000`), 2.30 → 1.93 (`full`),
  2.61 → 2.18 (`format_only`). Format-only stays higher, as it should: its
  responses are shuffled, so the instruction does not predict them. The logs
  are in `adapters/olmo_*/train_log.json`.
- **Training data revision:** `databricks/databricks-dolly-15k` is pinned in
  `configs/finetune.yaml` (`dataset_revision`), like the model revisions.

Controls:

1. **Format-only:** LoRA on the same prompts with responses shuffled across
   items. Learns chat format and "answer now" without any prompt–response
   relation. Intent-tracking here would mean the mechanism is format, not
   intention.
2. **Data-scale curve:** adapters on 500, 2,000 and all 10,544 pairs.

Training runs as a CLI (`gricean finetune`) so it can go in the background.

## 8. Repository layout and data flow

```
gricean-devices/
  stimuli/              stimulus YAML per category; pt/ mirror; neither/ samples; CHECKSUMS
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
interpretability, no LLM-as-judge for any code that feeds a table (coder 2 of
§5.1 is an agent whose codes feed the agreement measure only; amended
2026-11-18). Portuguese is a generalization check, not a
parallel study. Prong (2) of the summary is argued, not simulated (§1).

## 11. Open items for the author

- Write the 48 + 12 English items and the Portuguese mirror.
- ~~Recruit the second coder.~~ Closed 2026-11-18: coder 2 is an agent (§5.1).
- Decide, after seeing results, whether the Instruct column stays in the
  paper.
- Decide, after seeing results, whether the Base (mid-training) rung is
  reported as a column or as a one-line dose note under the figure.
- Select the Ouyang et al. (2022) pairs to quote beside the OLMo responses.

## 12. Configuration implied by rev. 2 (for the implementation plan)

Registry (`configs/models.yaml`): `olmo_stage1` (repo `allenai/OLMo-2-0425-1B`,
revision `stage1-step1907359-tokens4001B`, template `none`);
`olmo_stage1_alpaca` (same weights, template `alpaca`, self-run baseline);
`olmo_n500`, `olmo_n2000`, `olmo_full`, `olmo_format_only` (same weights,
template `alpaca`, adapters under `adapters/olmo_*`); `pythia_base`
(`EleutherAI/pythia-1.4b`, template `none`, in no ladder). Ladders:
`olmo: [olmo_stage1, olmo_base, olmo_sft, olmo_dpo, olmo_instruct]`;
`selfrun: [olmo_stage1_alpaca, olmo_n500, olmo_n2000, olmo_full, olmo_format_only]`.
`neither_source`: `pythia_base` for both. The `smollm_*` entries and ladder
and the `qwen` ladder leave the default run; they may stay in the file as
optional. `configs/finetune.yaml`: `base: olmo_stage1`. `configs/generation.yaml`:
`neither.n: 5`, five seeds. `make pin` resolves the stage-1 branch name to its
commit sha like any other revision.
