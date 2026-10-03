from gricean import finetune as F
from gricean.lm import ALPACA


def records():
    return [{"instruction": f"q{i}", "response": f"a{i}", "context": "", "category": c}
            for i, c in enumerate(["open_qa", "closed_qa", "brainstorming", "summarization", "general_qa", "classification"] * 3)]


def test_filter_keeps_only_plain_categories():
    kept = F.filter_dolly(records())
    assert {r["category"] for r in kept} <= set(F.DOLLY_KEEP)
    assert len(kept) == 12


def test_prepare_subsets_and_format_only():
    kept = F.filter_dolly(records())
    out = F.prepare(kept, sizes=[4, 8], seed=0, shuffle_responses=True)
    assert set(out) == {"full", "n4", "n8", "format_only"}
    assert len(out["n4"]) == 4 and len(out["n8"]) == 8 and len(out["full"]) == 12
    assert out["n4"] == out["n8"][:4]                      # nested subsets
    assert out["full"][0]["text"].startswith(ALPACA.format(prompt=out["full"][0]["instruction"]))
    pairs_full = {(r["instruction"], r["response"]) for r in out["full"]}
    pairs_fo = {(r["instruction"], r["response"]) for r in out["format_only"]}
    assert pairs_full != pairs_fo and len(pairs_fo) == 12
    assert {r["response"] for r in out["format_only"]} == {r["response"] for r in out["full"]}


def test_prepare_is_deterministic():
    kept = F.filter_dolly(records())
    assert F.prepare(kept, sizes=[4], seed=1, shuffle_responses=True) == F.prepare(kept, sizes=[4], seed=1, shuffle_responses=True)


class _CharTok:
    eos_token = "E"
    eos_token_id = ord("E")
    def __call__(self, texts, truncation, max_length):
        ids = [[ord(c) for c in t][:max_length] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def test_tokenize_rows_keeps_eos_in_labels():
    out = F.tokenize_rows([{"text": "ab"}, {"text": "cde"}], _CharTok(), max_len=16)
    assert all(ex["labels"] == ex["input_ids"] for ex in out)
    assert all(ex["labels"][-1] == ord("E") for ex in out)
