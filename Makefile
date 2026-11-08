.PHONY: install test pin freeze neither generate logprobs sheet report all finetune determinism
install:
	uv sync --extra dev
test:
	uv run pytest
pin:
	uv run python scripts/pin_revisions.py
freeze:
	uv run gricean freeze
neither:
	uv run gricean neither --family olmo --family qwen --family smollm
generate:
	uv run gricean generate --all
logprobs:
	uv run gricean logprobs --all
sheet:
	uv run gricean sheet
report:
	uv run gricean report
all: report
finetune:
	uv run gricean finetune
determinism:
	uv run python scripts/determinism_check.py olmo_base
