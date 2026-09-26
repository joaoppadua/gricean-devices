from pathlib import Path
from gricean import neither as N
from gricean.io import read_jsonl
from gricean.lm import FakeLM, ModelSpec
from gricean.stimuli import Item

CFG = {"n": 3, "temperature": 1.0, "max_new_tokens": 20, "seeds": [11, 12, 13]}


def item(i="a", intended="Brazil became independent in 1822."):
    return Item(id=i, category="indirect_request", language="en", control=False,
                prompt="Can you tell me what year Brazil became independent?",
                literal_reading="", intended_reading="", literal_continuation="Yes, I can.",
                intended_continuation=intended, versions={"full": "Can you tell me what year Brazil became independent?"})


def test_truncate_cuts_at_newline_then_words():
    assert N.truncate("one two\nthree", 20) == "one two"
    assert N.truncate(" ".join(str(i) for i in range(30)), 5) == "0 1 2 3 4"


def test_overlap_is_containment_of_reference_words():
    assert N.token_overlap("Yes, I can.", "yes i can") == 1.0
    assert N.token_overlap("apple", "pear") == 0.0
    # a long sample that contains the whole reference answer must score 1.0, not a diluted Jaccard
    long = "According to historical records, Brazil became independent from Portugal on September 7, 1822"
    assert N.token_overlap(long, "Brazil became independent in 1822.") == 0.8   # 4 of 5 reference words ("on" != "in")


def test_flags_long_sample_that_contains_the_answer(tmp_path: Path):
    src = FakeLM("qwen_base", canned={"Can you tell me what year Brazil became independent?":
                                      "According to historical records, Brazil became independent from Portugal on September 7, 1822"})
    spec = ModelSpec(name="qwen_base", repo="q", revision="r1", family="qwen", template="none")
    N.sample_neither(src, spec, "olmo", [item()], cfg=CFG, stimuli_checksum="abc", root=tmp_path)
    assert read_jsonl(N.neither_path("olmo", tmp_path))[0]["flag"] is True


def test_picks_lowest_overlap_and_flags_when_all_overlap(tmp_path: Path):
    class Src(FakeLM):
        def generate(self, prompt, *, max_new_tokens, temperature, seed):
            return {11: "Brazil became independent in 1822.", 12: "Can you tell me what year Chile\nmore",
                    13: "Brazil became independent long ago."}[seed]
    src = Src("qwen_base")
    spec = ModelSpec(name="qwen_base", repo="q", revision="r1", family="qwen", template="none")
    n = N.sample_neither(src, spec, "olmo", [item()], cfg=CFG, stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(N.neither_path("olmo", tmp_path))
    assert n == 1 and len(rows) == 1
    r = rows[0]
    assert r["chosen"] == "Can you tell me what year Chile"
    assert r["flag"] is False
    assert r["source_model"] == "qwen_base" and r["seeds"] == [11, 12, 13]


def test_flags_when_every_sample_answers(tmp_path: Path):
    src = FakeLM("qwen_base", canned={"Can you tell me what year Brazil became independent?": "Brazil became independent in 1822."})
    spec = ModelSpec(name="qwen_base", repo="q", revision="r1", family="qwen", template="none")
    N.sample_neither(src, spec, "olmo", [item()], cfg=CFG, stimuli_checksum="abc", root=tmp_path)
    r = read_jsonl(N.neither_path("olmo", tmp_path))[0]
    assert r["flag"] is True
    assert N.load_neither("olmo", tmp_path) == {"a": "Brazil became independent in 1822."}


def test_controls_are_skipped_and_rerun_is_noop(tmp_path: Path):
    c = Item(id="c", category="indirect_request", language="en", control=True, prompt="x",
             literal_reading="", intended_reading="", literal_continuation=None,
             intended_continuation=None, versions={"full": "x"})
    src = FakeLM("qwen_base")
    spec = ModelSpec(name="qwen_base", repo="q", revision="r1", family="qwen", template="none")
    assert N.sample_neither(src, spec, "olmo", [item(), c], cfg=CFG, stimuli_checksum="abc", root=tmp_path) == 1
    assert N.sample_neither(src, spec, "olmo", [item(), c], cfg=CFG, stimuli_checksum="abc", root=tmp_path) == 0
