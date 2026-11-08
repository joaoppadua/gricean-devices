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
3. Fill `revision:` in `configs/models.yaml` with the commit hashes you used.
4. `make freeze` — validates stimuli and writes `stimuli/CHECKSUMS`.
5. `make neither` — samples degenerate continuations (downloads two base models).
6. `make generate` then `make logprobs` — all models, templated and raw. Expect ~1–2 h on a 16 GB M3.
7. `make sheet` — writes `data/coding/coder1.csv`; code it. `uv run gricean sheet --coder2 --frac 0.25` for the blind sample.
8. `make finetune` — trains the four SmolLM2 adapters (bf16 LoRA; ~80 min for the full ~10k-row adapter on an M3, ~3 h for all four); then rerun steps 6–7 for the `smollm` ladder.
9. `make report` — writes `paper/olmo/`. `uv run gricean report --ladder qwen` and `--ladder smollm` for the other tables.
10. `uv run python scripts/determinism_check.py olmo_base` — reports whether greedy decoding is deterministic on your device.

Committed `data/` lets you run step 9 without any model. `make report` writes whatever the data supports:
Line 1 tables need a fully coded `coder1.csv`, Line 2 needs log-prob rows; anything missing is skipped with a warning.

Protocol notes baked into the code: the `embedded` stimulus version is never chat-templated for any model;
tuned models are also run raw as a template control (`paper/<ladder>/template_control.csv`); untemplated
prompts get `raw_separator` (a space) prefixed to scored continuations; every decoding knob is pinned
(`top_k=0, top_p=1, repetition_penalty=1`) so checkpoint defaults cannot leak in; `neither` rows flagged by the
overlap filter block scoring until you set `approved: true` on them (or edit `chosen`).

## Notebooks

`uv run marimo edit notebooks/01_stimuli.py` (and 02–05). Notebooks only display; all logic lives in `src/gricean/`.
