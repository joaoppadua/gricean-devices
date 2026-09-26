import math
from pathlib import Path
import pytest
from gricean import logprobs as LP
from gricean.io import read_jsonl
from gricean.lm import FakeLM, ModelSpec
from gricean.stimuli import Item


def item():
    return Item(id="a", category="indirect_request", language="en", control=False,
                prompt="Can you tell me the time?", literal_reading="", intended_reading="",
                literal_continuation="Yes.", intended_continuation="It is noon.",
                versions={"full": "Can you tell me the time?", "stripped": "Can you tell me the time"})


def control():
    return Item(id="c", category="indirect_request", language="en", control=True, prompt="x",
                literal_reading="", intended_reading="", literal_continuation=None,
                intended_continuation=None, versions={"full": "x"})


def test_three_way_normalizes_and_softmaxes():
    out = LP.three_way({"literal": (-6.0, 2), "intended": (-3.0, 3), "neither": (-4.0, 4)})
    assert out["norm"] == {"literal": -3.0, "intended": -1.0, "neither": -1.0}
    assert out["log_odds"] == 2.0
    s = out["share"]
    assert math.isclose(sum(s.values()), 1.0)
    assert math.isclose(s["intended"], s["neither"]) and s["literal"] < s["intended"]


def test_run_writes_rows_excluding_controls(tmp_path: Path):
    lm = FakeLM("fake", logprob_rules={"Yes.": -3.0, "It is noon.": -1.5, "blah blah": -2.0})
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="chat")
    n = LP.run_logprobs(lm, spec, [item(), control()], {"a": "blah blah"}, template=True,
                        stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(LP.logprob_path("fake", True, tmp_path))
    assert n == 2 and {r["version"] for r in rows} == {"full", "stripped"}
    r = rows[0]
    assert r["lp_intended"] == -1.5 and r["n_intended"] == 3
    assert math.isclose(r["log_odds"], (-1.5 / 3) - (-3.0 / 1))
    assert math.isclose(r["share_literal"] + r["share_intended"] + r["share_neither"], 1.0)


def test_missing_neither_raises(tmp_path: Path):
    lm = FakeLM("fake")
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="none")
    with pytest.raises(KeyError, match="a"):
        LP.run_logprobs(lm, spec, [item()], {}, template=False, stimuli_checksum="abc", root=tmp_path)


def test_rerun_noop(tmp_path: Path):
    lm = FakeLM("fake")
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="none")
    LP.run_logprobs(lm, spec, [item()], {"a": "x"}, template=False, stimuli_checksum="abc", root=tmp_path)
    assert LP.run_logprobs(lm, spec, [item()], {"a": "x"}, template=False, stimuli_checksum="abc", root=tmp_path) == 0


def test_raw_prompts_prefix_continuations_with_separator(tmp_path: Path):
    lm = FakeLM("fake", logprob_rules={" Yes.": -3.0, " It is noon.": -1.5, " blah blah": -2.0})
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="none")
    LP.run_logprobs(lm, spec, [item()], {"a": "blah blah"}, template=False, stimuli_checksum="abc",
                    root=tmp_path, raw_separator=" ")
    r = read_jsonl(LP.logprob_path("fake", False, tmp_path))[0]
    assert r["lp_intended"] == -1.5 and r["lp_literal"] == -3.0 and r["raw_separator"] == " "


def test_templated_prompts_get_no_separator(tmp_path: Path):
    lm = FakeLM("fake", logprob_rules={"Yes.": -3.0, "It is noon.": -1.5})
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="chat")
    LP.run_logprobs(lm, spec, [item()], {"a": "x"}, template=True, stimuli_checksum="abc", root=tmp_path)
    rows = read_jsonl(LP.logprob_path("fake", True, tmp_path))
    assert all(r["lp_intended"] == -1.5 for r in rows)


def test_embedded_scored_raw_even_in_template_pass_and_skipped_in_tuned_raw_pass(tmp_path: Path):
    emb = Item(id="e", category="indirect_request", language="en", control=False, prompt="p",
               literal_reading="", intended_reading="", literal_continuation="Yes.",
               intended_continuation="It is noon.", versions={"full": "p", "embedded": "Before. p After."})
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="chat")
    LP.run_logprobs(FakeLM("fake"), spec, [emb], {"e": "x"}, template=True, stimuli_checksum="abc", root=tmp_path)
    rows = {r["version"]: r for r in read_jsonl(LP.logprob_path("fake", True, tmp_path))}
    assert rows["embedded"]["template"] is False and not rows["embedded"]["prompt_text"].startswith("<user>")
    assert rows["full"]["template"] is True
    LP.run_logprobs(FakeLM("fake"), spec, [emb], {"e": "x"}, template=False, stimuli_checksum="abc", root=tmp_path)
    assert {r["version"] for r in read_jsonl(LP.logprob_path("fake", False, tmp_path))} == {"full"}


def test_rejects_empty_continuation(tmp_path: Path):
    spec = ModelSpec(name="fake", repo="r", revision="v1", family="f", template="none")
    with pytest.raises(ValueError, match="empty"):
        LP.run_logprobs(FakeLM("fake"), spec, [item()], {"a": ""}, template=False, stimuli_checksum="abc", root=tmp_path)
