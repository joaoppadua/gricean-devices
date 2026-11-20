# From the tables to the draft: a workflow

*Written 2026-11-20, when the experiment pipeline was complete. It is a
suggestion for the author's own phase of the work: reading the results,
exploring the data and drafting the paper.*

The rule behind every step: read first, explore second. If you start by
changing things in the notebooks, what you happen to see will shape the
story. The predictions were written down before the runs (spec §5.4), so
the story can be held to them.

The spec is `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md`.

## 1. Start from the predictions, not the data

Open spec §5.4. It has four predictions. For each one, write "held",
"failed" or "mixed", and name the table cell that says so. Use only the
committed files under `paper/`.

Two are already recorded in the spec as failed on Line 1:

- the `embedded` version (tuned models do worse there, not better);
- growth with data scale on `intent` (37, 36 and 37 `intended` responses
  for the adapters trained on 500, 2,000 and all pairs).

## 2. Read the report in this order

The order goes from the main claim to its controls to its illustrations.
Do it for `paper/olmo/` first, then for `paper/selfrun/`.

| Step | File | What it answers |
|---|---|---|
| 1 | `intent.csv`, `ladder.png` | Line 1: which reading each rung follows |
| 2 | `logodds.csv` | Line 2: the same question, with no coder |
| 3 | `logodds_template_control.csv` | Whether the chat template alone explains it (official ladder only) |
| 4 | `correctness.csv` | Whether a response that follows a reading is also right |
| 5 | `kappa.txt` | How far the two coders agree |
| 6 | `examples.md` | What the responses look like |

Files ending in `_pt` are the Portuguese subset. It is reported apart and
never pooled with English (spec §4.4).

Three things to keep in mind while reading:

- Every point rests on eight items per category. The intervals in the
  figure are per model; two rungs are compared on the same items, not by
  whether their intervals overlap.
- In `correctness.csv` an empty share means no response followed that
  reading. A zero means some did and none was `correct`.
- `incorrect` also covers a right answer followed by run-on text, and an
  English answer to a Portuguese prompt. A low share is not only wrong
  information.

## 3. Take the three open decisions early

They are listed in spec §11 and they change the shape of every table, so
settle them before drafting the results section.

- Does the Instruct column stay in the paper?
- Is the released base a column, or a one-line note under the figure?
- Which pairs from Ouyang et al. (2022) are quoted beside the OLMo
  responses?

## 4. Then open the notebooks, to understand

    uv run marimo edit notebooks/02_responses.py

The notebooks display; they do not produce the paper's numbers.

| Notebook | Use it to |
|---|---|
| `01_stimuli.py` | Reread the items by category |
| `02_responses.py` | Read one item across the rungs, in each version |
| `03_logodds.py` | See the Line 2 table and the figure, per ladder and language |
| `04_finetune.py` | See the training loss of the self-run adapters and their Line 2 table |
| `05_coding.py` | See agreement and the correctness table; its button rebuilds `paper/` from committed data |

The dropdowns only change what is shown. Three things would change the
committed data, so leave them alone on `main`:

- the "run generation" button in `02_responses.py`;
- anything under `configs/`;
- the stimuli (`stimuli/`), which are frozen.

To try something that touches those, make a branch first.

## 5. Keep a list of what exploration suggests

When a look at the data suggests a claim that no table makes, write it
down as a question. Each question then goes one of two ways:

- it stays in the paper as a remark, marked as made after seeing the data;
- or it becomes an analysis: the measure is fixed and committed first,
  computed second, and the outcome is committed apart. Every measure in
  this repository was added that way.

One question is already on the list. Among the self-run adapters, the
number of `intended` responses coded `correct` rises with data scale (23,
25 and 28, of about 37). The counts are small and carry no interval. If
the paper says correctness "grows" with scale, that sentence needs a
paired test on the same items first. The same holds for any sentence that
says a measure "rises" between two rungs.

## 6. Draft from the files

- **Numbers** come from the files under `paper/`, and only from them.
  `make report` and `uv run gricean report --ladder selfrun` regenerate
  them from committed data, with no model.
- **Quotations** come from `examples.md` and `examples_pt.md`. They were
  drawn at random with a fixed seed. If you need more, raise the numbers
  in `gricean.report.EXAMPLES`: the draw then adds examples and replaces
  none. Do not change the seed. If the paper quotes only some of the
  examples drawn, say so in one sentence.
- **Methods and limitations** come from the spec. Its revision log dates
  every change to the design, and each one is a candidate for the
  limitations section.

What the paper has to disclose, all of it recorded in the spec:

- Coder 2 is an instruction-tuned model, the kind of system the paper is
  about (§5.1).
- Coder 1, the author, was not blind to the model.
- Part of the stimulus set was drafted with Claude Code and reviewed by
  the author (§4).
- Each category has eight items, and the bootstrap intervals rest on
  them.
- Two predictions of §5.4 failed as stated.

Before the discussion section, reread the debate notes listed under "Read
first" in `CLAUDE.md`, as a check that each anticipated objection is
answered by something in the tables.

## 7. When the draft needs something new

A new table, a test, more examples or a change of wording in the spec is
one branch and one pull request into `main`, merged with a merge commit.
Nothing is computed on the real data before its definition is committed.
