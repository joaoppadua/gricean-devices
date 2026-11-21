from pathlib import Path

import pandas as pd
import pytest
import yaml

from gricean import cli
from gricean import turns as T
from gricean.io import read_jsonl, write_jsonl

CAP = 10
MODELS = ["m_base", "m_sft"]
EXPECTED = {"m_base": False, "m_sft": True}


class Tok:
    """Words are tokens. With special tokens on, one more is counted, as a real tokenizer adds BOS."""

    def __init__(self, extra: int = 0):
        self.extra = extra                  # a tokenizer that counts differently, to tell two models apart

    def __call__(self, text, add_special_tokens=True):
        class Enc:
            pass
        enc = Enc()
        enc.input_ids = (["<s>"] if add_special_tokens else []) + text.split() + ["+"] * self.extra
        return enc


class TokenLM:
    """Generates fixed token ids. 0 is end-of-sequence; 9 decodes without a space, so it merges with the
    token before it and the decoded text has one word fewer than the tokens that made it."""

    name = "m"
    eos_ids = {0}
    tokenizer = Tok()
    PIECES = {1: " a", 2: " b", 9: "x"}

    def __init__(self, outputs: dict[str, list[int]]):
        self.outputs = outputs

    def format(self, prompt: str, template: bool) -> str:
        return f"<t>{prompt}" if template else prompt

    def generate_tokens(self, prompt, *, max_new_tokens, temperature, seed):
        assert temperature is None and seed is None             # calibration is greedy
        return self.outputs[prompt][:max_new_tokens]

    def decode(self, tokens):
        return "".join(self.PIECES[t] for t in tokens if t != 0)


def _resp(model, template, item, response, *, version="full", mode="greedy", seed=None, language="en",
          control=False, category="indirect_request"):
    return {"item_id": item, "category": category, "language": language, "control": control,
            "version": version, "model": model, "template": template, "mode": mode, "seed": seed,
            "prompt_text": "p", "stimulus_text": "p", "response": response, "stamp": "s",
            "stimuli_checksum": "c"}


def _rid(model, template, item, version="full", seed="None"):
    return f"{model}|{int(template)}|{item}|{version}|greedy|{seed}"


def _coded(model, template, item, intent, correctness="correct", *, version="full", language="en",
           control=False):
    return {"row_id": _rid(model, template, item, version), "item_id": item, "category": "indirect_request",
            "language": language, "control": control, "version": version, "mode": "greedy", "seed": "",
            "model": model, "template": template, "intent": intent, "correctness": correctness}


def test_count_tokens_leaves_special_tokens_out():
    assert T.count_tokens(Tok(), "one two three") == 3
    assert T.count_tokens(Tok(), "") == 0


def test_write_lengths_counts_the_english_coded_rows_with_each_models_tokenizer(tmp_path: Path):
    resp = tmp_path / "responses"
    write_jsonl(resp / "m_base__raw.jsonl", [
        _resp("m_base", False, "a0", "one two"),
        _resp("m_base", False, "a0", "sampled text here", mode="sample", seed=1),      # not coded
        _resp("m_base", False, "p0", "um dois", language="pt"),                        # not English
    ])
    write_jsonl(resp / "m_sft__tmpl.jsonl", [_resp("m_sft", True, "a0", "one two three")])
    write_jsonl(resp / "m_sft__raw.jsonl", [
        _resp("m_sft", False, "a0", "raw control"),                                    # template control: not coded
        _resp("m_sft", False, "a0", "", version="embedded"),                           # embedded is raw: coded
    ])
    toks = {"m_base": Tok(), "m_sft": Tok(extra=10)}
    n = T.write_lengths(resp, tmp_path / "lengths.csv", EXPECTED, toks.__getitem__)
    got = T.load_lengths(tmp_path / "lengths.csv")
    assert n == 3
    # ids as `coding._all_responses` writes them: with a sampled row in the data, a greedy seed reads `nan`
    assert got.to_dict() == {_rid("m_base", False, "a0", seed="nan"): 2, _rid("m_sft", True, "a0", seed="nan"): 13,
                             _rid("m_sft", False, "a0", "embedded", seed="nan"): 10}


def test_calibrate_records_true_count_end_of_sequence_and_recount():
    lm = TokenLM({"stops": [1, 2, 0],                 # ended after two tokens
                  "runs": [1] * 20,                   # cut at the limit
                  "merges": [1, 9] * 5,               # cut at the limit; decodes to five words
                  "last": [1] * (CAP - 1) + [0]})     # end-of-sequence is the last token allowed: it ended
    rows = T.calibrate(lm, ["stops", "runs", "merges", "last"], template=False, max_new_tokens=CAP)
    assert [(r["n_generated"], r["eos"], r["n_recount"]) for r in rows] == [
        (2, True, 2), (CAP, False, CAP), (CAP, False, 5), (CAP - 1, True, CAP - 1)]
    assert {(r["model"], r["template"]) for r in rows} == {("m", False)}
    assert [r["prompt"] for r in rows] == [0, 1, 2, 3]


def test_calibrate_formats_the_prompt_when_templated():
    lm = TokenLM({"<t>q": [1, 0]})
    assert T.calibrate(lm, ["q"], template=True, max_new_tokens=CAP)[0]["n_generated"] == 1


def _cal(*rows):
    return pd.DataFrame([{"eos": eos, "n_recount": n} for eos, n in rows])


def test_tolerance_is_the_largest_shortfall_among_responses_that_reached_the_limit():
    cal = _cal((False, CAP), (False, CAP - 3), (False, CAP - 1), (True, 2), (True, CAP - 9))
    assert T.tolerance(cal, CAP, min_at_limit=3) == 3


def test_tolerance_is_zero_when_the_recount_never_falls_short():
    assert T.tolerance(_cal((False, CAP), (False, CAP + 2), (True, 1)), CAP, min_at_limit=2) == 0


def test_tolerance_refuses_a_shortfall_beyond_the_bound():
    with pytest.raises(ValueError, match="larger than"):
        T.tolerance(_cal((False, CAP - 4), (False, CAP)), CAP, min_at_limit=2, max_tolerance=3)


def test_tolerance_refuses_too_few_responses_at_the_limit():
    with pytest.raises(ValueError, match="reached the limit"):
        T.tolerance(_cal((False, CAP), (True, 1)), CAP, min_at_limit=2)


def sheet():
    rows = [
        _coded("m_base", False, "a0", "intended", "incorrect"),      # 10 tokens: at the limit
        _coded("m_base", False, "a1", "intended", "incorrect"),      # 3: ended
        _coded("m_base", False, "a2", "neither"),                    # 9: inside the band at tolerance 1
        _coded("m_base", False, "a3", "literal"),                    # 0: an empty response ended
        _coded("m_sft", True, "a0", "intended"),                     # 4
        _coded("m_sft", True, "a1", "intended"),                     # 10: a long answer cut off
        _coded("m_sft", True, "a2", "intended", "incorrect"),        # 2
        _coded("m_sft", True, "a3", "intended"),                     # 5
        _coded("m_sft", False, "a0", "literal"),                     # template control: not in the main condition
        _coded("m_sft", True, "p0", "intended", language="pt"),      # Portuguese: not counted
    ]
    return pd.DataFrame(rows)


def lengths():
    n = {("m_base", False, "a0"): 10, ("m_base", False, "a1"): 3, ("m_base", False, "a2"): 9,
         ("m_base", False, "a3"): 0, ("m_sft", True, "a0"): 4, ("m_sft", True, "a1"): 10,
         ("m_sft", True, "a2"): 2, ("m_sft", True, "a3"): 5}
    return pd.Series({_rid(*k): v for k, v in n.items()}, name="n_tokens")


def test_flag_marks_ended_limit_and_band_on_english_main_rows_only():
    df = T.flag(sheet(), lengths(), MODELS, EXPECTED, cap=CAP, tolerance=1).set_index("row_id")
    assert len(df) == 8                                              # no Portuguese row, no template control
    assert not df.loc[_rid("m_base", False, "a0"), "ended"]
    assert not df.loc[_rid("m_base", False, "a2"), "ended"]          # 9 >= 10 - 1
    assert df.loc[_rid("m_base", False, "a2"), "in_band"]
    assert not df.loc[_rid("m_base", False, "a0"), "in_band"]
    assert df.loc[_rid("m_base", False, "a3"), "ended"]


def test_flag_refuses_a_coded_row_without_a_count():
    with pytest.raises(ValueError, match="no token count"):
        T.flag(sheet(), lengths().iloc[1:], MODELS, EXPECTED, cap=CAP, tolerance=0)


def test_termination_table_gives_n_and_share_ended_for_each_intent():
    t = T.termination_table(T.flag(sheet(), lengths(), MODELS, EXPECTED, cap=CAP, tolerance=0), MODELS)
    cell = ("indirect_request", "full")
    assert t.loc[cell, ("m_base", "intended", "n")] == 2
    assert t.loc[cell, ("m_base", "intended", "ended")] == 0.5
    assert t.loc[cell, ("m_base", "neither", "ended")] == 1.0        # 9 tokens ended when the tolerance is 0
    assert t.loc[cell, ("m_sft", "intended", "n")] == 4
    assert t.loc[cell, ("m_sft", "intended", "ended")] == 0.75
    assert t.loc[cell, ("m_sft", "literal", "n")] == 0
    assert pd.isna(t.loc[cell, ("m_sft", "literal", "ended")])       # no response: empty, not zero


def test_termination_correctness_table_splits_intended_responses_by_code():
    t = T.termination_correctness_table(
        T.flag(sheet(), lengths(), MODELS, EXPECTED, cap=CAP, tolerance=0), MODELS)
    cell = ("indirect_request", "full")
    assert t.loc[cell, ("m_base", "incorrect", "n")] == 2
    assert t.loc[cell, ("m_base", "incorrect", "at_limit")] == 1
    assert t.loc[cell, ("m_base", "correct", "n")] == 0
    assert t.loc[cell, ("m_sft", "correct", "n")] == 3
    assert t.loc[cell, ("m_sft", "correct", "at_limit")] == 1       # the long answer cut off
    assert t.loc[cell, ("m_sft", "incorrect", "at_limit")] == 0


def test_termination_correctness_table_refuses_na_under_intended():
    s = sheet()
    s.loc[0, "correctness"] = "na"
    with pytest.raises(ValueError, match="`na`"):
        T.termination_correctness_table(T.flag(s, lengths(), MODELS, EXPECTED, cap=CAP, tolerance=0), MODELS)


@pytest.mark.parametrize("only_a, only_b, p", [(7, 1, 0.0703125), (1, 7, 0.0703125), (8, 0, 0.0078125),
                                                (0, 0, 1.0), (3, 3, 1.0)])
def test_mcnemar_exact_is_a_two_sided_binomial_test_on_the_discordant_items(only_a, only_b, p):
    assert T.mcnemar_exact(only_a, only_b) == pytest.approx(p)


def test_paired_counts_complete_turns_on_the_same_dissociation_items():
    df = T.flag(sheet(), lengths(), MODELS, EXPECTED, cap=CAP, tolerance=0)
    got = T.paired(df, "m_base", "m_sft")
    # a0: base at the limit, sft complete; a1: base complete, sft cut off; a2: base neither, sft complete;
    # a3: base literal, sft complete
    assert got == {"n": 4, "both": 0, "only_first": 1, "only_second": 3, "neither": 0,
                   "p": pytest.approx(0.625)}


def test_paired_leaves_controls_and_other_versions_out():
    s = pd.concat([sheet(), pd.DataFrame([
        _coded("m_base", False, "c0", "intended", control=True), _coded("m_sft", True, "c0", "intended", control=True),
        _coded("m_base", False, "a0", "intended", version="stripped"),
        _coded("m_sft", True, "a0", "intended", version="stripped")])], ignore_index=True)
    more = pd.Series({_rid("m_base", False, "c0"): 1, _rid("m_sft", True, "c0"): 1,
                      _rid("m_base", False, "a0", "stripped"): 1, _rid("m_sft", True, "a0", "stripped"): 1})
    df = T.flag(s, pd.concat([lengths(), more]), MODELS, EXPECTED, cap=CAP, tolerance=0)
    assert T.paired(df, "m_base", "m_sft")["n"] == 4


def test_paired_refuses_models_coded_on_different_items():
    s = sheet()
    s = s[s["row_id"] != _rid("m_sft", True, "a3")]
    with pytest.raises(ValueError, match="same items"):
        T.paired(T.flag(s, lengths(), MODELS, EXPECTED, cap=CAP, tolerance=0), "m_base", "m_sft")


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    coded = root / "coder1.csv"
    s = sheet().assign(response="r", stimulus_text="p", prompt_text="p", coder="c1")
    s.to_csv(coded, index=False)
    lens = root / "lengths.csv"
    lengths().rename_axis("row_id").reset_index().to_csv(lens, index=False)
    cal = root / "calibration.jsonl"
    write_jsonl(cal, [{"model": "m", "template": False, "prompt": i, "n_generated": CAP, "eos": False,
                       "n_recount": CAP - (i == 0)} for i in range(3)]
                + [{"model": "m", "template": False, "prompt": 3, "n_generated": 9, "eos": True, "n_recount": 9},
                   {"model": "m", "template": False, "prompt": 4, "n_generated": 2, "eos": True, "n_recount": 2}])
    return coded, lens, cal


def test_build_writes_both_tables_and_the_record_in_english_only(tmp_path: Path):
    coded, lens, cal = _write_inputs(tmp_path)
    out = tmp_path / "paper"
    summary = T.build(coded, lens, cal, MODELS, out, EXPECTED, cap=CAP, comparison=("m_base", "m_sft"),
                      min_at_limit=3)
    assert summary["tolerance"] == 1
    assert sorted(p.name for p in out.iterdir()) == [
        "termination.csv", "termination.tex", "termination.txt",
        "termination_correctness.csv", "termination_correctness.tex"]
    record = (out / "termination.txt").read_text()
    assert "limit=10 tolerance=1" in record
    assert "calibration responses=5 reached_limit=3 ended=2 ended_inside_band=1" in record
    assert "coded responses=8 reached_limit=3 inside_band=1" in record
    assert "complete_turns m_base=1/4 m_sft=3/4" in record
    assert "m_base vs m_sft: n=4 both=0 only_first=1 only_second=3 neither=0 mcnemar_exact_p=0.6250" in record


def test_build_without_a_comparison_writes_none(tmp_path: Path):
    coded, lens, cal = _write_inputs(tmp_path)
    T.build(coded, lens, cal, MODELS, tmp_path / "paper", EXPECTED, cap=CAP, min_at_limit=3)
    assert "mcnemar" not in (tmp_path / "paper" / "termination.txt").read_text()


def test_calibration_prompts_are_twenty_distinct_texts():
    assert len(T.CALIBRATION_PROMPTS) == 20 == len(set(T.CALIBRATION_PROMPTS))


def _project(root: Path) -> None:
    (root / "configs").mkdir()
    yaml.safe_dump({"max_new_tokens": CAP}, (root / "configs/generation.yaml").open("w"))
    yaml.safe_dump({
        "models": {"m_base": {"repo": "x", "revision": "1", "family": "t", "template": "none"},
                   "m_sft": {"repo": "y", "revision": "1", "family": "t", "template": "chat"}},
        "ladders": {"t": MODELS}, "neither_source": {"t": "m_base"},
    }, (root / "configs/models.yaml").open("w"))
    words = {("m_base", False, "a0"): 10, ("m_base", False, "a1"): 3, ("m_base", False, "a2"): 9,
             ("m_base", False, "a3"): 0, ("m_sft", True, "a0"): 4, ("m_sft", True, "a1"): 10,
             ("m_sft", True, "a2"): 2, ("m_sft", True, "a3"): 5}
    for (model, template, item), n in words.items():
        path = root / "data/responses" / f"{model}__{'tmpl' if template else 'raw'}.jsonl"
        write_jsonl(path, read_jsonl(path) + [_resp(model, template, item, " ".join(["w"] * n))])
    (root / "data/coding").mkdir(parents=True)
    s = sheet().assign(response="r", stimulus_text="p", prompt_text="p", coder="c1")
    s[s["language"] == "en"].to_csv(root / "data/coding/coder1.csv", index=False)
    write_jsonl(root / "data/lengths/calibration.jsonl",
                [{"model": "m", "template": False, "prompt": i, "n_generated": CAP, "eos": False, "n_recount": CAP}
                 for i in range(T.MIN_AT_LIMIT)])


def test_cli_counts_then_writes_the_tables(tmp_path: Path, monkeypatch):
    _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(T, "load_tokenizer", lambda spec: Tok())
    assert cli.main(["turns", "--count"]) == 0
    assert len(T.load_lengths(tmp_path / "data/lengths/coded_en.csv")) == 8
    assert cli.main(["turns", "--ladder", "t"]) == 0
    assert "limit=10 tolerance=0" in (tmp_path / "paper/t/termination.txt").read_text()


def test_check_prompts_refuses_a_prompt_that_is_a_stimulus():
    T.check_prompts(["a", "b"], ["c"])
    with pytest.raises(ValueError, match="stimuli"):
        T.check_prompts(["a", "b"], ["b"])


def test_save_calibration_replaces_one_models_rows(tmp_path: Path):
    def row(model, prompt, n):
        return {"model": model, "template": False, "prompt": prompt, "n_generated": n, "eos": True, "n_recount": n}
    path = tmp_path / "cal.jsonl"
    T.save_calibration(path, "b", [row("b", 0, 1)])
    T.save_calibration(path, "a", [row("a", 1, 2), row("a", 0, 3)])
    T.save_calibration(path, "b", [row("b", 0, 9)])
    assert [(r["model"], r["prompt"], r["n_generated"]) for r in read_jsonl(path)] == [
        ("a", 0, 3), ("a", 1, 2), ("b", 0, 9)]


def test_cli_report_also_writes_the_turn_tables(tmp_path: Path, monkeypatch):
    _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(T, "load_tokenizer", lambda spec: Tok())
    assert cli.main(["turns", "--count"]) == 0
    with pytest.warns(UserWarning):                       # this project has no log-probs: Line 2 is skipped
        assert cli.main(["report", "--ladder", "t"]) == 0
    assert (tmp_path / "paper/t/termination.csv").exists()
    assert (tmp_path / "paper/t/intent.csv").exists()


def test_cli_report_without_counts_writes_no_turn_table(tmp_path: Path, monkeypatch):
    _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning):
        assert cli.main(["report", "--ladder", "t"]) == 0
    assert not (tmp_path / "paper/t/termination.csv").exists()


class AnyPromptLM(TokenLM):
    def __init__(self, name):
        self.name = name

    def generate_tokens(self, prompt, *, max_new_tokens, temperature, seed):
        return [1, 2, 0]


def test_cli_calibrate_runs_raw_and_the_template_a_model_has(tmp_path: Path, monkeypatch):
    _project(tmp_path)
    (tmp_path / "data/lengths/calibration.jsonl").unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.stimuli, "load_all", lambda: [])
    for name in MODELS:
        assert cli.main(["turns", "--calibrate", name], loader=lambda spec: AnyPromptLM(spec.name)) == 0
    rows = pd.DataFrame(read_jsonl(tmp_path / "data/lengths/calibration.jsonl"))
    assert rows.groupby(["model", "template"]).size().to_dict() == {
        ("m_base", False): 20, ("m_sft", False): 20, ("m_sft", True): 20}
