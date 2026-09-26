.PHONY: install test freeze neither generate logprobs sheet report all
install:
	uv sync --extra dev
test:
	uv run pytest
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
