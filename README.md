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
3. `make pin` resolves every `revision: main` or branch name in `configs/models.yaml` to the commit hash in your cache or on the Hub (each line says which); commit the result. Pinning changes the model stamp, so committed seed rows recompute on the next generate/logprobs.
4. `make freeze` — validates stimuli (schema, no `TODO` placeholders left, quotas, non-vacuous ablation) and writes `stimuli/CHECKSUMS`. The category files are pre-filled with id-numbered skeletons whose fields read `TODO`; use `uv run gricean freeze --partial` while filling them (placeholder items load but do not count toward quotas), and note that `neither`, `generate` and `logprobs` refuse to run while any `TODO` remains.
5. `make neither` — samples degenerate continuations from Pythia-1.4b (downloads it, 5.9 GB); five samples per item.
6. `make generate` then `make logprobs` — every model in the `olmo` ladder (stage 1 downloads as fp32, 5.9 GB), templated and raw. Expect ~2–3 h on a 16 GB M3.
7. `make sheet` — writes `data/coding/coder1.csv`; code it. `uv run gricean sheet --coder2 --frac 0.25` for the blind sample.
8. `make finetune` — trains the four OLMo stage-1 adapters (bf16 LoRA; ≈5–6 h each for `full` and `format_only`, ≈1 h for `n2000`, minutes for `n500`; ≈12–15 h in total, run overnight); then rerun steps 6–7, which pick up the `selfrun` ladder once `adapters/olmo_*` exist. Self-run models run under the Alpaca template only (no raw pass).
9. `make report` — writes `paper/olmo/`. `uv run gricean report --ladder selfrun` for the self-run table.
10. `uv run python scripts/determinism_check.py olmo_base` — reports whether greedy decoding is deterministic on your device.

The Qwen2.5 pair stays in `configs/models.yaml` as an optional contrast (`uv run gricean generate --model qwen_base`); it is in no ladder and not reported.

Committed `data/` lets you run step 9 without any model. `make report` writes whatever the data supports:
Line 1 tables need a fully coded `coder1.csv`, Line 2 needs log-prob rows; anything missing is skipped with a warning.

Protocol notes baked into the code: the `embedded` stimulus version is never chat-templated for any model;
tuned models in the official ladder are also run raw as a template control (`paper/<ladder>/template_control.csv`; self-run models are exempt, `raw_control: false`); untemplated
prompts get `raw_separator` (a space) prefixed to scored continuations; every decoding knob is pinned
(`top_k=0, top_p=1, repetition_penalty=1`) so checkpoint defaults cannot leak in; `neither` rows flagged by the
overlap filter block scoring until you set `approved: true` on them (or edit `chosen`).

## Notebooks

`uv run marimo edit notebooks/01_stimuli.py` (and 02–05). Notebooks only display; all logic lives in `src/gricean/`.
