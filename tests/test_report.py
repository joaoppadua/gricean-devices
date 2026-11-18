from pathlib import Path
import pandas as pd
import pytest
from gricean import report as R
from gricean.io import write_jsonl

MODELS = ["m_base", "m_sft"]
EXPECTED = {"m_base": False, "m_sft": True}


def _row(model, template, i, intent, version="full", control=False):
    return {"row_id": f"{model}|{int(template)}|a{i}|{version}|greedy|", "item_id": f"a{i}",
            "category": "indirect_request", "language": "en", "control": control,
            "version": version, "mode": "greedy", "seed": "", "prompt_text": "p", "stimulus_text": "p",
            "response": "r", "model": model, "template": template, "coder": "c1",
            "intent": intent, "correctness": "na"}


def coded():
    rows = []
    for i, it in enumerate(["neither", "literal", "literal", "intended"]):
        rows.append(_row("m_base", False, i, it))
    for i in range(4):
        rows.append(_row("m_sft", True, i, "intended"))
        rows.append(_row("m_sft", False, i, "literal"))          # template control: must not pollute main
    rows.append(_row("m_base", False, 9, "intended", control=True))
    rows.append(_row("m_sft", True, 9, "intended", control=True))
    rows.append(_row("m_sft", False, 9, "literal", control=True))
    rows.append(_row("m_sft", False, 5, "intended", version="embedded"))   # embedded is always raw: belongs to main
    return pd.DataFrame(rows)


def lp_rows(root: Path):
    def rows(model, template, values, version="full"):
        return [{"item_id": f"a{i}", "category": "indirect_request", "language": "en", "version": version,
                 "model": model, "template": template, "mode": "logprob", "seed": None, "log_odds": v,
                 "share_neither": 0.5, "share_literal": 0.25, "share_intended": 0.25, "stamp": "s",
                 "stimuli_checksum": "c"} for i, v in enumerate(values)]
    write_jsonl(root / "m_base__raw.jsonl", rows("m_base", False, [-1.0, -0.5]))
    write_jsonl(root / "m_sft__tmpl.jsonl", rows("m_sft", True, [2.0, 3.0]) + rows("m_sft", False, [7.0], "embedded"))
    write_jsonl(root / "m_sft__raw.jsonl", rows("m_sft", False, [-100.0, -100.0]))


def resp_rows(resp: Path):
    write_jsonl(resp / "m_base__raw.jsonl", [{"item_id": "a0", "category": "indirect_request", "language": "en",
        "control": False, "version": "full", "model": "m_base", "template": False, "mode": "greedy", "seed": None,
        "prompt_text": "Can you tell me the time?", "stimulus_text": "Can you tell me the time?",
        "response": "answer from m_base", "stamp": "s", "stimuli_checksum": "c"}])
    write_jsonl(resp / "m_sft__raw.jsonl", [{"item_id": "a0", "category": "indirect_request", "language": "en",
        "control": False, "version": "full", "model": "m_sft", "template": False, "mode": "greedy", "seed": None,
        "prompt_text": "Can you tell me the time?", "stimulus_text": "Can you tell me the time?",
        "response": "raw answer", "stamp": "s", "stimuli_checksum": "c"}])
    write_jsonl(resp / "m_sft__tmpl.jsonl", [{"item_id": "a0", "category": "indirect_request", "language": "en",
        "control": False, "version": "full", "model": "m_sft", "template": True, "mode": "greedy", "seed": None,
        "prompt_text": "<user>Can you tell me the time?</user><assistant>", "stimulus_text": "Can you tell me the time?",
        "response": "answer from m_sft", "stamp": "s", "stimuli_checksum": "c"}])


def test_intent_table_uses_main_condition_only_and_reports_controls():
    t = R.intent_table(coded(), MODELS, EXPECTED)
    assert t.loc[("indirect_request", "full"), ("m_base", "intended")] == 0.25
    assert t.loc[("indirect_request", "full"), ("m_base", "neither")] == 0.25
    assert t.loc[("indirect_request", "full"), ("m_sft", "intended")] == 1.0      # raw rows excluded
    assert t.loc[("indirect_request", "control"), ("m_sft", "intended")] == 1.0
    assert t.loc[("indirect_request", "control"), ("m_base", "intended")] == 1.0
    assert t.loc[("indirect_request", "embedded"), ("m_sft", "intended")] == 1.0


def test_template_control_table_holds_only_raw_rows_of_tuned_models():
    t = R.template_control_table(coded(), MODELS, EXPECTED)
    assert t.loc[("indirect_request", "full"), ("m_sft", "literal")] == 1.0
    assert ("m_base", "literal") not in t.columns or t[("m_base", "literal")].isna().all()


def test_logodds_table_means_and_ci_ignore_template_control(tmp_path: Path):
    lp_rows(tmp_path)
    t = R.logodds_table(tmp_path, MODELS, EXPECTED)
    assert t.loc[("indirect_request", "full"), ("m_sft", "mean")] == 2.5
    assert t.loc[("indirect_request", "embedded"), ("m_sft", "mean")] == 7.0
    assert t.loc[("indirect_request", "full"), ("m_sft", "ci_low")] <= 2.5 <= t.loc[("indirect_request", "full"), ("m_sft", "ci_high")]


def test_examples_print_main_condition_and_raw_stimulus(tmp_path: Path):
    resp = tmp_path / "resp"; resp_rows(resp)
    md = R.examples(resp, MODELS, ["a0"], EXPECTED)
    assert "answer from m_base" in md and "answer from m_sft" in md and "raw answer" not in md
    assert "**Prompt:** Can you tell me the time?" in md and "<user>" not in md


def test_build_writes_all_outputs(tmp_path: Path):
    coded().to_csv(tmp_path / "coded.csv", index=False)
    lp_rows(tmp_path / "lp"); resp_rows(tmp_path / "resp")
    summary = R.build(tmp_path / "coded.csv", None, None, tmp_path / "resp", tmp_path / "lp", MODELS,
                      tmp_path / "paper", EXPECTED)
    for name in ("intent.csv", "intent.tex", "template_control.csv", "logodds.csv", "logodds.tex",
                 "logodds_template_control.csv", "ladder.png", "examples.md"):
        assert (tmp_path / "paper" / name).exists()
    assert summary["n_coded"] == len(coded()) and summary["kappa"] is None


def test_build_tolerates_uncoded_sheet_and_missing_logprobs(tmp_path: Path):
    df = coded(); df["intent"] = ""; df["correctness"] = ""
    df.to_csv(tmp_path / "coded.csv", index=False)
    resp_rows(tmp_path / "resp")
    summary = R.build(tmp_path / "coded.csv", None, None, tmp_path / "resp", tmp_path / "lp_missing", MODELS,
                      tmp_path / "paper", EXPECTED)
    assert summary["n_coded"] == 0 and summary["warnings"]
    assert (tmp_path / "paper" / "examples.md").exists()
    assert not (tmp_path / "paper" / "intent.csv").exists()


def test_examples_marks_empty_greedy_response(tmp_path: Path):
    resp = tmp_path / "resp"
    write_jsonl(resp / "m_base__raw.jsonl", [{"item_id": "a0", "category": "indirect_request", "language": "en",
        "control": False, "version": "full", "model": "m_base", "template": False, "mode": "greedy", "seed": None,
        "stimulus_text": "Can you?", "prompt_text": "Can you?", "response": "", "stamp": "s", "stimuli_checksum": "c"}])
    md = R.examples(resp, ["m_base"], ["a0"], {"m_base": False})
    assert f"**m_base:** {R.EMPTY_RESPONSE}" in md


def test_logodds_template_control_table_holds_only_raw_rows_of_tuned_models(tmp_path: Path):
    lp_rows(tmp_path)
    t = R.logodds_template_control_table(tmp_path, MODELS, EXPECTED)
    assert t.loc[("indirect_request", "full"), ("m_sft", "mean")] == -100.0
    assert list(t.columns.get_level_values(0).unique()) == ["m_sft"]     # a base model has no raw control
    assert ("indirect_request", "embedded") not in t.index                # embedded belongs to the main table


def test_build_skips_line1_template_control_when_no_control_row_is_coded(tmp_path: Path):
    """Line 1 codes the main conditions only; the template control is then reported from Line 2."""
    df = coded(); df = df[(df["model"] == "m_base") | df["template"] | (df["version"] == "embedded")]
    df.to_csv(tmp_path / "coded.csv", index=False)
    lp_rows(tmp_path / "lp"); resp_rows(tmp_path / "resp")
    summary = R.build(tmp_path / "coded.csv", None, None, tmp_path / "resp", tmp_path / "lp", MODELS,
                      tmp_path / "paper", EXPECTED)
    assert (tmp_path / "paper" / "intent.csv").exists()
    assert not (tmp_path / "paper" / "template_control.csv").exists()
    assert (tmp_path / "paper" / "logodds_template_control.csv").exists()
    assert not summary["warnings"]


def test_build_writes_no_template_control_for_a_ladder_without_raw_passes(tmp_path: Path):
    lp_rows(tmp_path / "lp"); resp_rows(tmp_path / "resp")
    summary = R.build(tmp_path / "missing.csv", None, None, tmp_path / "resp", tmp_path / "lp", ["m_base"],
                      tmp_path / "paper", EXPECTED)
    assert (tmp_path / "paper" / "logodds.csv").exists()
    assert not (tmp_path / "paper" / "logodds_template_control.csv").exists()
    assert not any("template control" in w for w in summary["warnings"])


# --- Portuguese is reported separately, never pooled with English (spec 4.4) ---

IR_FULL = ("indirect_request", "full")


def coded_two_languages():
    pt = [dict(_row("m_base", False, 20 + i, "neither"), language="pt") for i in range(2)]
    pt += [dict(_row("m_sft", True, 20 + i, it), language="pt") for i, it in enumerate(["intended", "literal"])]
    return pd.concat([coded(), pd.DataFrame(pt)], ignore_index=True)


def lp_rows_pt(root: Path):
    write_jsonl(root / "m_sft__tmpl_pt.jsonl", [
        {"item_id": "p0", "category": "indirect_request", "language": "pt", "version": "full", "model": "m_sft",
         "template": True, "mode": "logprob", "seed": None, "log_odds": 10.0, "share_neither": 0.5,
         "share_literal": 0.25, "share_intended": 0.25, "stamp": "s", "stimuli_checksum": "c"}])


def test_intent_table_refuses_to_pool_languages():
    with pytest.raises(ValueError, match="language"):
        R.intent_table(coded_two_languages(), MODELS, EXPECTED)


def test_intent_table_reports_one_language_when_asked():
    en = R.intent_table(coded_two_languages(), MODELS, EXPECTED, language="en")
    pt = R.intent_table(coded_two_languages(), MODELS, EXPECTED, language="pt")
    assert en.loc[IR_FULL, ("m_base", "intended")] == 0.25      # the English rates, untouched by the Portuguese rows
    assert en.loc[IR_FULL, ("m_sft", "intended")] == 1.0
    assert pt.loc[IR_FULL, ("m_sft", "intended")] == 0.5
    assert pt.loc[IR_FULL, ("m_base", "neither")] == 1.0


def test_logodds_table_refuses_to_pool_languages(tmp_path: Path):
    lp_rows(tmp_path); lp_rows_pt(tmp_path)
    with pytest.raises(ValueError, match="language"):
        R.logodds_table(tmp_path, MODELS, EXPECTED)


def test_logodds_table_reports_one_language_when_asked(tmp_path: Path):
    lp_rows(tmp_path); lp_rows_pt(tmp_path)
    assert R.logodds_table(tmp_path, MODELS, EXPECTED, language="en").loc[IR_FULL, ("m_sft", "mean")] == 2.5
    assert R.logodds_table(tmp_path, MODELS, EXPECTED, language="pt").loc[IR_FULL, ("m_sft", "mean")] == 10.0


def test_build_writes_each_language_to_its_own_tables(tmp_path: Path):
    coded_two_languages().to_csv(tmp_path / "coded.csv", index=False)
    lp_rows(tmp_path / "lp"); lp_rows_pt(tmp_path / "lp"); resp_rows(tmp_path / "resp")
    summary = R.build(tmp_path / "coded.csv", None, None, tmp_path / "resp", tmp_path / "lp", MODELS,
                      tmp_path / "paper", EXPECTED)
    assert not summary["warnings"]
    read = lambda name: pd.read_csv(tmp_path / "paper" / name, header=[0, 1], index_col=[0, 1])
    assert read("intent.csv").loc[IR_FULL, ("m_sft", "intended")] == 1.0
    assert read("intent_pt.csv").loc[IR_FULL, ("m_sft", "intended")] == 0.5
    assert read("logodds.csv").loc[IR_FULL, ("m_sft", "mean")] == 2.5
    assert read("logodds_pt.csv").loc[IR_FULL, ("m_sft", "mean")] == 10.0
    assert (tmp_path / "paper" / "intent_pt.tex").exists() and (tmp_path / "paper" / "logodds_pt.tex").exists()


def test_intent_table_gives_zero_for_an_intent_a_coded_model_never_shows():
    """m_sft has main-condition rows and all are `intended`: its literal and neither rates are 0, not missing."""
    t = R.intent_table(coded(), MODELS, EXPECTED)
    assert t.loc[IR_FULL, ("m_sft", "literal")] == 0.0
    assert t.loc[IR_FULL, ("m_sft", "neither")] == 0.0


def test_build_reports_kappa_from_a_blind_sheet(tmp_path: Path):
    """Coder 2's sheet carries opaque ids; the key leads back to the rows of coder 1's sheet."""
    full = coded()
    full.to_csv(tmp_path / "coded.csv", index=False)
    sample = full.iloc[:6].copy()
    blind_ids = [f"x{i}" for i in range(len(sample))]
    pd.DataFrame({"row_id": blind_ids, "source_row_id": sample["row_id"].tolist(), "model": sample["model"].tolist(),
                  "template": sample["template"].tolist(), "prompt_text": "p"}).to_csv(tmp_path / "c2.key.csv", index=False)
    sample = sample.drop(columns=["model", "template", "prompt_text"]).assign(row_id=blind_ids, coder="c2")
    sample.to_csv(tmp_path / "c2.csv", index=False)
    lp_rows(tmp_path / "lp"); resp_rows(tmp_path / "resp")
    summary = R.build(tmp_path / "coded.csv", tmp_path / "c2.key.csv", tmp_path / "c2.csv", tmp_path / "resp",
                      tmp_path / "lp", MODELS, tmp_path / "paper", EXPECTED)
    assert summary["kappa"] == 1.0
    lines = (tmp_path / "paper" / "kappa.txt").read_text().splitlines()
    assert lines == ["cohen_kappa_intent=1.000 n=6 same=6",                   # the key led every row back
                     "cohen_kappa_correctness_same_intent=1.000 n=6 same=6",
                     "cohen_kappa_correctness_all=1.000 n=6 same=6"]
    assert summary["agreement"]["correctness_same_intent"] == {"n": 6, "same": 6, "kappa": 1.0}
