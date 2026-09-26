from pathlib import Path
from gricean import generate as G
from gricean.io import read_jsonl
from gricean.lm import FakeLM, ModelSpec
from gricean.stimuli import Item

GEN = {"max_new_tokens": 150, "greedy": True, "samples": {"n": 2, "temperature": 0.7, "seeds": [1, 2]}}


def items():
    return [
        Item(id="a", category="indirect_request", language="en", control=False,
             prompt="Can you tell me the time?", literal_reading="", intended_reading="",
             literal_continuation="Yes.", intended_continuation="Noon.",
             versions={"full": "Can you tell me the time?", "stripped": "Can you tell me the time"}),
        Item(id="c", category="indirect_request", language="en", control=True,
             prompt="What time is it?", literal_reading="", intended_reading="",
             literal_continuation=None, intended_continuation=None, versions={"full": "What time is it?"}),
    ]


def spec():
    return ModelSpec(name="fake", repo="r", revision="v1", family="f", template="chat")


def test_generates_all_rows_and_uses_template(tmp_path: Path):
    lm = FakeLM("fake")
    n = G.run_generate(lm, spec(), items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(G.response_path("fake", True, tmp_path))
    # item a: 2 versions x (1 greedy + 2 samples) = 6 ; item c: 1 version x 3 = 3
    assert n == 9 and len(rows) == 9
    assert all(r["prompt_text"].startswith("<user>") for r in rows)
    greedy = [r for r in rows if r["mode"] == "greedy"]
    assert all(r["seed"] is None for r in greedy)
    assert {r["seed"] for r in rows if r["mode"] == "sample"} == {1, 2}


def test_raw_path_does_not_apply_template(tmp_path: Path):
    G.run_generate(FakeLM("fake"), spec(), items(), template=False, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(G.response_path("fake", False, tmp_path))
    assert all(not r["prompt_text"].startswith("<user>") for r in rows)


def test_second_run_is_noop_and_changed_item_recomputed(tmp_path: Path):
    lm = FakeLM("fake")
    G.run_generate(lm, spec(), items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    assert G.run_generate(lm, spec(), items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path) == 0
    changed = items()
    changed[1] = Item(**{**changed[1].__dict__, "versions": {"full": "What time is it now?"}})
    n = G.run_generate(lm, spec(), changed, template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    assert n == 3
    rows = read_jsonl(G.response_path("fake", True, tmp_path))
    assert len(rows) == 9


def test_model_revision_change_recomputes_everything(tmp_path: Path):
    lm = FakeLM("fake")
    G.run_generate(lm, spec(), items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path)
    s2 = ModelSpec(name="fake", repo="r", revision="v2", family="f", template="chat")
    assert G.run_generate(lm, s2, items(), template=True, gen_cfg=GEN, stimuli_checksum="abc", root=tmp_path) == 9
