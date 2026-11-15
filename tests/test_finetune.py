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


def test_load_dolly_asks_for_the_pinned_revision(monkeypatch):
    """The training data is pinned like the models: `load_dolly` passes the revision through to the hub."""
    import datasets
    seen = {}

    def fake_load_dataset(name, split, revision=None):
        seen.update(name=name, split=split, revision=revision)
        return [{"instruction": "q", "response": "a", "category": "open_qa", "context": ""},
                {"instruction": "q2", "response": "a2", "category": "closed_qa", "context": "c"}]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    rows = F.load_dolly("abc123")
    assert seen == {"name": "databricks/databricks-dolly-15k", "split": "train", "revision": "abc123"}
    assert [r["instruction"] for r in rows] == ["q"]          # still filtered to the context-free categories



def test_collate_pads_to_a_multiple_and_masks_the_padding():
    """Batches come in a few fixed lengths (MPS memory grows with every new shape); padding changes no loss term."""
    short = {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [5, 6, 7]}
    long = {"input_ids": list(range(1, 71)), "attention_mask": [1] * 70, "labels": list(range(1, 71))}
    batch = F.collate([short, long], pad_id=0, multiple=64)
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape == (2, 128)
    assert batch["input_ids"][0, :3].tolist() == [5, 6, 7] and batch["input_ids"][1, :70].tolist() == list(range(1, 71))
    assert batch["input_ids"][0, 3:].eq(0).all() and batch["attention_mask"][0].sum() == 3 and batch["attention_mask"][1].sum() == 70
    assert batch["labels"][0, :3].tolist() == [5, 6, 7] and batch["labels"][0, 3:].eq(-100).all() and batch["labels"][1, 70:].eq(-100).all()


def test_collate_leaves_a_batch_that_already_fits_the_multiple():
    row = {"input_ids": list(range(64)), "attention_mask": [1] * 64, "labels": list(range(64))}
    assert F.collate([row, row], pad_id=0, multiple=64)["input_ids"].shape == (2, 64)
