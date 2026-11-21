# gricean-devices

Simulation code for Pádua, "LLMs are Gricean machines" (Revista Texto Livre, 2027).

See `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md` for the design.

## Quick start

    make install
    make test
    make all     # regenerates paper/ from committed data; no model needed

## Replication

1. `make install` (uv creates `.venv` from `uv.lock`).
2. `make test` (unit tests, no downloads).
3. `make pin` resolves every `revision: main` or branch name in `configs/models.yaml` to the commit hash in your cache or on the Hub (each line says which); commit the result. The committed file is already pinned (2026-11-10), so this prints `all revisions already pinned` unless you add a model.
4. `make freeze` — validates stimuli (schema, no `TODO` placeholders left, quotas, non-vacuous ablation) and writes `stimuli/CHECKSUMS`. The committed set of 78 items is frozen (2026-11-10); every model stage recomputes the checksum and refuses to run if a stimulus file or `stimuli/ablation.yaml` has changed since. While writing new items, `uv run gricean freeze --partial` skips the completeness checks (items whose fields read `TODO` load but do not count toward quotas); `neither`, `generate` and `logprobs` refuse to run while any `TODO` remains.
5. `make neither` — samples degenerate continuations from Pythia-1.4b (downloads it, 5.9 GB); five samples per item, each cut at the first line break after its text starts. The committed `stimuli/neither/olmo.jsonl` holds the 60 rows, including the author's pass (three `chosen` texts replaced by another sample of the same row), so a rerun recomputes nothing.
6. `make generate` then `make logprobs` — every model in the `olmo` ladder (stage 1 downloads as fp32, 5.9 GB), templated and raw. On a 16 GB M3 the committed run (2026-11-11) took ≈4.5–5 h to generate (4,608 responses over nine passes, 15–45 min each) and under 10 min to score (990 rows). Each pass writes its file only when it finishes, so an interrupted pass starts over while finished passes are kept.
7. `make sheet` — writes `data/coding/coder1.csv`; code it. The sheet holds the greedy response of each model's main condition (828 rows for the `olmo` run; sampled responses and the raw template control are not hand-coded, spec §5.1). Empty responses arrive coded `neither` / `na`; fill `intent` and `correctness` in the rest. Rerunning keeps the codes already entered. `uv run gricean sheet --coder2 --frac 0.25` writes the blind sample for the second coder, drawn from the hand-coded rows. The second coder is an agent, a Claude Haiku model (spec §5.1, step 15): `uv run gricean coder2 --batches DIR` writes the sample as text batches with blank forms, each batch is coded from `docs/codebook-coder2.md` under the brief in `docs/coder2-brief.md`, and `uv run gricean coder2 --enter DIR` copies the filled forms into `coder2.csv`. The committed `coder2.csv` holds those codes, so no step of the replication needs the hosted model.
8. `make finetune` — trains the four OLMo stage-1 adapters (bf16 LoRA, batch 2 × accumulation 8, batches padded to multiples of 64 tokens; measured on a 16 GB M3 in the committed run: 11 min for `n500`, 40 min for `n2000`, ≈3.5 h each for `full` and `format_only`, ≈8 h in total, run overnight). The committed `adapters/olmo_*` hold the weights, the training rows and the loss logs, so this step can be skipped. It trains `full` first; to train smallest first, one process per adapter: `for s in n500 n2000 full format_only; do uv run gricean finetune --set $s; done`. Then rerun steps 6–7, which pick up the `selfrun` ladder once `adapters/olmo_*` exist. Self-run models run under the Alpaca template only (no raw pass). Dolly is pinned in `configs/finetune.yaml` (`dataset_revision`).
9. `make report` — writes `paper/olmo/`. `uv run gricean report --ladder selfrun` for the self-run table. English tables carry plain names (`intent.csv`, `correctness.csv`, `logodds.csv`) and the first and last feed the figure; `correctness.csv` gives, for the responses coded `intended` and for those coded `literal`, their number and the share coded `correct` (spec §5.1, step 17); Portuguese is written separately as `*_pt.csv` and never pooled (spec §4.4). `examples.md` (and `examples_pt.md`) prints responses drawn at random with a fixed seed: one prompt of each category at every rung, and three responses of each rung (spec §6). `make report` then runs `uv run gricean results`, which rewrites the tables of `paper/RESULTS.md` from the files of both ladders. That page records the outcome of each prediction of spec §5.4 and the author's decisions; its text outside the marked tables is written by hand and is never changed by a command. `uv run gricean results --check` says whether its tables are up to date.
10. `uv run python scripts/determinism_check.py olmo_base` — reports whether greedy decoding is deterministic on your device. On the reference machine: 5 of 5 identical (2026-11-20, spec §9).
11. Post hoc, not part of the original design (spec §13.1): turn completion, English only. `make report` and `uv run gricean report --ladder selfrun` also write `termination.csv` (for each intent, the number of responses and the share that ended before the 150-token limit), `termination_correctness.csv` (the `intended` responses coded `correct` and `incorrect`, and how many of each reached the limit) and `termination.txt` (the tolerance, and on the self-run ladder the one exploratory comparison). They are computed from the committed `data/lengths/`, with no model and no tokenizer; `uv run gricean turns --ladder NAME` writes them alone. To redo the inputs: `uv run gricean turns --calibrate MODEL` for each of the ten models of the two ladders (generation; about 40 minutes in all), then `uv run gricean turns --count` (tokenizers only).

The Qwen2.5 pair stays in `configs/models.yaml` as an optional contrast (`uv run gricean generate --model qwen_base`); it is in no ladder and not reported.

Committed `data/` lets you run step 9 without any model. `make report` writes whatever the data supports:
Line 1 tables need a fully coded `coder1.csv`, Line 2 needs log-prob rows; anything missing is skipped with a warning.

Protocol notes baked into the code: the `embedded` stimulus version is never chat-templated for any model;
tuned models in the official ladder are also run raw as a template control (reported from Line 2, `paper/<ladder>/logodds_template_control.csv`; self-run models are exempt, `raw_control: false`); untemplated
prompts get `raw_separator` (a space) prefixed to scored continuations; every decoding knob is pinned
(`top_k=0, top_p=1, repetition_penalty=1`) so checkpoint defaults cannot leak in; `neither` rows flagged by the
overlap filter block scoring until you set `approved: true` on them (or edit `chosen`).

## Notebooks

`uv run marimo edit notebooks/01_stimuli.py` (and 02–05). Notebooks only display; all logic lives in `src/gricean/`.

A notebook runs all its cells when it opens, so there is nothing to start by hand: change a dropdown and the cells below it update. Notebooks read the files on disk when their cells run; after a `make` stage, reopen the notebook to see the new data.

The notebooks work from the terminal command above and from the VS Code marimo extension. Each one moves to the repo root in its first cell (`gricean.paths.enter_repo_root`), because every path in the package is relative to it and the extension starts notebooks inside `notebooks/`.
