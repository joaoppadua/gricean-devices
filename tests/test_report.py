from pathlib import Path
import pandas as pd
import pytest
from gricean import report as R
from gricean.io import write_jsonl

MODELS = ["m_base", "m_sft"]
EXPECTED = {"m_base": False, "m_sft": True}


def _row(model, template, i, intent, version="full", control=False, correctness="correct"):
    return {"row_id": f"{model}|{int(template)}|a{i}|{version}|greedy|", "item_id": f"a{i}",
            "category": "indirect_request", "language": "en", "control": control,
            "version": version, "mode": "greedy", "seed": "", "prompt_text": "p", "stimulus_text": "p",
            "response": "r", "model": model, "template": template, "coder": "c1",
            "intent": intent, "correctness": correctness}


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


# --- correctness, for the responses that follow a reading (spec 5.1, step 17) ---

def judged():
    """m_base: two intended (one correct), three literal (two correct), two neither. m_sft: four intended,
    three correct; its raw rows are literal and are not its main condition."""
    codes = [("intended", "correct"), ("intended", "incorrect"), ("literal", "correct"), ("literal", "correct"),
             ("literal", "incorrect"), ("neither", "incorrect"), ("neither", "na")]
    rows = [_row("m_base", False, i, intent, correctness=c) for i, (intent, c) in enumerate(codes)]
    rows += [_row("m_sft", True, i, "intended", correctness=c) for i, c in enumerate(["correct"] * 3 + ["incorrect"])]
    rows += [_row("m_sft", False, i, "literal", correctness="incorrect") for i in range(4)]
    rows.append(_row("m_base", False, 9, "intended", control=True, correctness="incorrect"))
    return pd.DataFrame(rows)


def test_correctness_table_counts_the_responses_of_a_reading_and_gives_the_share_coded_correct():
    t = R.correctness_table(judged(), MODELS, EXPECTED)
    assert t.loc[IR_FULL, ("m_base", "intended", "n")] == 2
    assert t.loc[IR_FULL, ("m_base", "intended", "correct")] == 0.5
    assert t.loc[IR_FULL, ("m_base", "literal", "n")] == 3
    assert t.loc[IR_FULL, ("m_base", "literal", "correct")] == pytest.approx(2 / 3)
    assert t.loc[IR_FULL, ("m_sft", "intended", "correct")] == 0.75


def test_correctness_table_has_no_column_for_neither():
    """Under `neither` the code says whether the text is well formed, which is another question."""
    t = R.correctness_table(judged(), MODELS, EXPECTED)
    assert set(t.columns.get_level_values(1)) == {"intended", "literal"}


def test_correctness_table_uses_main_condition_only():
    t = R.correctness_table(judged(), MODELS, EXPECTED)
    assert t.loc[IR_FULL, ("m_sft", "literal", "n")] == 0          # the four raw rows of the tuned model are left out


def test_correctness_table_leaves_the_share_empty_where_no_response_follows_the_reading():
    t = R.correctness_table(judged(), MODELS, EXPECTED)
    assert pd.isna(t.loc[IR_FULL, ("m_sft", "literal", "correct")])   # no response to judge: not a share of zero


def test_correctness_table_reports_controls_as_their_own_condition():
    t = R.correctness_table(judged(), MODELS, EXPECTED)
    control = ("indirect_request", "control")
    assert t.loc[control, ("m_base", "intended", "n")] == 1
    assert t.loc[control, ("m_base", "intended", "correct")] == 0.0
    assert t.loc[control, ("m_sft", "intended", "n")] == 0          # a model with rows, none of them in this cell


def test_correctness_table_marks_a_model_without_rows_as_missing():
    t = R.correctness_table(judged(), MODELS + ["m_new"], {**EXPECTED, "m_new": True})
    assert pd.isna(t.loc[IR_FULL, ("m_new", "intended", "n")])     # not coded yet is not "no intended response"


def test_correctness_table_refuses_a_response_that_follows_a_reading_and_was_not_judged():
    df = judged()
    df.loc[0, "correctness"] = "na"
    with pytest.raises(ValueError, match="na"):
        R.correctness_table(df, MODELS, EXPECTED)


def judged_two_languages():
    pt = [dict(_row("m_sft", True, 20 + i, "intended", correctness=c), language="pt")
          for i, c in enumerate(["incorrect", "incorrect", "correct", "incorrect"])]
    return pd.concat([judged(), pd.DataFrame(pt)], ignore_index=True)


def test_correctness_table_refuses_to_pool_languages():
    with pytest.raises(ValueError, match="language"):
        R.correctness_table(judged_two_languages(), MODELS, EXPECTED)


def test_correctness_table_reports_one_language_when_asked():
    en = R.correctness_table(judged_two_languages(), MODELS, EXPECTED, language="en")
    pt = R.correctness_table(judged_two_languages(), MODELS, EXPECTED, language="pt")
    assert en.loc[IR_FULL, ("m_sft", "intended", "correct")] == 0.75
    assert pt.loc[IR_FULL, ("m_sft", "intended", "correct")] == 0.25


def test_build_writes_the_correctness_table_of_each_language(tmp_path: Path):
    judged_two_languages().to_csv(tmp_path / "coded.csv", index=False)
    resp_rows(tmp_path / "resp")
    summary = R.build(tmp_path / "coded.csv", None, None, tmp_path / "resp", tmp_path / "lp", MODELS,
                      tmp_path / "paper", EXPECTED)
    assert not [w for w in summary["warnings"] if "correctness" in w]
    read = lambda name: pd.read_csv(tmp_path / "paper" / name, header=[0, 1, 2], index_col=[0, 1])
    assert read("correctness.csv").loc[IR_FULL, ("m_base", "literal", "n")] == 3
    assert read("correctness.csv").loc[IR_FULL, ("m_sft", "intended", "correct")] == 0.75
    assert read("correctness_pt.csv").loc[IR_FULL, ("m_sft", "intended", "correct")] == 0.25
    assert (tmp_path / "paper" / "correctness.tex").exists() and (tmp_path / "paper" / "correctness_pt.tex").exists()


def test_build_skips_the_correctness_table_and_keeps_the_intent_table_when_a_row_was_not_judged(tmp_path: Path):
    df = judged()
    df.loc[0, "correctness"] = "na"
    df.to_csv(tmp_path / "coded.csv", index=False)
    resp_rows(tmp_path / "resp")
    summary = R.build(tmp_path / "coded.csv", None, None, tmp_path / "resp", tmp_path / "lp", MODELS,
                      tmp_path / "paper", EXPECTED)
    assert any("correctness skipped" in w for w in summary["warnings"])
    assert (tmp_path / "paper" / "intent.csv").exists()
    assert not (tmp_path / "paper" / "correctness.csv").exists()


# --- the ladder figure: Line 2 means with their intervals ---

def _intervals(ax):
    """(x, low, high) of every interval drawn on an axis, in drawing order."""
    from matplotlib.container import ErrorbarContainer
    out = []
    for c in ax.containers:
        if isinstance(c, ErrorbarContainer):
            out += [(seg[0][0], seg[0][1], seg[1][1]) for bars in c.lines[2] for seg in bars.get_segments()]
    return out


def test_ladder_figure_draws_the_interval_of_each_line2_mean(tmp_path: Path):
    import matplotlib.pyplot as plt
    lp_rows(tmp_path)
    table = R.logodds_table(tmp_path, MODELS, EXPECTED)
    fig = R.draw_ladder(None, table, MODELS)
    drawn = _intervals(fig.axes[1])
    plt.close(fig)
    assert [(low, high) for _, low, high in drawn] == [
        pytest.approx((table.loc[IR_FULL, (m, "ci_low")], table.loc[IR_FULL, (m, "ci_high")])) for m in MODELS]
    assert drawn[0][1] < table.loc[IR_FULL, ("m_base", "mean")] < drawn[0][2]     # a real interval, around the mean


def test_ladder_figure_sets_the_categories_apart_so_that_intervals_do_not_overlap(tmp_path: Path):
    import matplotlib.pyplot as plt
    lp_rows(tmp_path)
    one = R.logodds_table(tmp_path, MODELS, EXPECTED)
    two = pd.concat([one, one.rename(index={"indirect_request": "pattern_continuation"})])
    fig = R.draw_ladder(None, two, MODELS)
    xs = [x for x, _, _ in _intervals(fig.axes[1])]
    plt.close(fig)
    assert len(set(xs)) == 4                                    # two categories at two models: four places


def test_ladder_figure_shows_two_categories_with_the_same_line1_rates(tmp_path: Path):
    import matplotlib.pyplot as plt
    one = R.intent_table(coded(), MODELS, EXPECTED)
    two = pd.concat([one, one.rename(index={"indirect_request": "pattern_continuation"})])     # same rates at every model
    fig = R.draw_ladder(two, None, MODELS)
    ax = fig.axes[0]
    points = [tuple(xy) for line in ax.get_lines() for xy in line.get_xydata()]
    low, high = ax.get_ylim()
    plt.close(fig)
    assert len(points) == 2 * len(MODELS)
    assert len(set(points)) == len(points)                      # no marker is drawn on top of another
    assert low < 0 and high > 1                                 # markers at 0 and at 1 are drawn whole


def test_ladder_figure_names_the_models_in_ladder_order_under_both_panels(tmp_path: Path):
    import matplotlib.pyplot as plt
    lp_rows(tmp_path)
    fig = R.draw_ladder(R.intent_table(coded(), MODELS, EXPECTED), R.logodds_table(tmp_path, MODELS, EXPECTED), MODELS)
    labels = [[t.get_text() for t in ax.get_xticklabels()] for ax in fig.axes]
    plt.close(fig)
    assert labels == [MODELS, MODELS]


# --- printed examples: drawn at random, never chosen (spec 6) ---

import re

CATEGORIES = (("ir", "indirect_request"), ("pc", "pattern_continuation"))


def _response(model, template, item, category, **changes):
    return {"item_id": item, "category": category, "language": "en", "control": False, "version": "full",
            "model": model, "template": template, "mode": "greedy", "seed": None,
            "prompt_text": f"<t>prompt of {item}</t>" if template else f"prompt of {item}",
            "stimulus_text": f"prompt of {item}", "response": f"{model} answers {item}", "stamp": "s",
            "stimuli_checksum": "c", **changes}


def example_rows(resp: Path, change=None):
    """Two models on two categories of four dissociation items, in English, and on two Portuguese items; with
    them, the rows that no example may show: controls, another version, samples and the raw control."""
    base, sft, raw = [], [], []
    for short, category in CATEGORIES:
        for i in range(4):
            item = f"{short}{i}"
            base.append(_response("m_base", False, item, category))
            sft.append(_response("m_sft", True, item, category))
            sft.append(_response("m_sft", True, item, category, version="stripped", response=f"stripped {item}"))
            sft.append(_response("m_sft", True, item, category, mode="sample", seed=1, response=f"sampled {item}"))
            raw.append(_response("m_sft", False, item, category, response=f"raw control {item}"))
        base.append(_response("m_base", False, f"{short}_c", category, control=True, response=f"control {short}"))
        base.append(_response("m_base", False, f"pt_{short}", category, language="pt"))
        sft.append(_response("m_sft", True, f"pt_{short}", category, language="pt"))
    rows = {"m_base__raw": base, "m_sft__tmpl": sft, "m_sft__raw": raw}
    if change:
        change(rows)
    for name, rs in rows.items():
        write_jsonl(resp / f"{name}.jsonl", rs)
    return resp


def _parts(md: str) -> tuple[str, str]:
    """The two parts of the examples: the same prompt at every rung, and the responses drawn for each rung."""
    _, same_prompt, per_rung = md.split("\n## ")
    return same_prompt, per_rung


def _drawn(part: str, level: str) -> list[str]:
    """Item ids under headings of one level, in the order printed."""
    return re.findall(rf"^{level} (\S+)", part, flags=re.M)


def _for_rung(per_rung: str, model: str) -> str:
    return per_rung.split(f"### {model}\n")[1].split("\n### ")[0]


def test_examples_show_one_prompt_of_each_category_with_the_response_of_every_rung(tmp_path: Path):
    same_prompt, _ = _parts(R.examples(example_rows(tmp_path), MODELS, EXPECTED, language="en"))
    items = _drawn(same_prompt, "###")
    assert sorted(i[:2] for i in items) == ["ir", "pc"]
    for item in items:
        assert same_prompt.index(f"m_base answers {item}") < same_prompt.index(f"m_sft answers {item}")   # ladder order


def test_examples_show_as_many_responses_of_each_rung_as_asked(tmp_path: Path):
    _, per_rung = _parts(R.examples(example_rows(tmp_path), MODELS, EXPECTED, language="en", per_rung=3))
    for model in MODELS:
        items = _drawn(_for_rung(per_rung, model), "####")
        assert len(set(items)) == 3
        assert all(f"{model} answers {item}" in _for_rung(per_rung, model) for item in items)


def test_examples_are_drawn_from_greedy_full_main_condition_responses_to_dissociation_items(tmp_path: Path):
    md = R.examples(example_rows(tmp_path), MODELS, EXPECTED, language="en", per_rung=8, per_category=4)
    assert md.count("m_sft answers ir0") == 2                    # every item is shown, in both parts
    for excluded in ("control ", "stripped ", "sampled ", "raw control "):
        assert excluded not in md


def test_examples_show_the_prompt_as_written_not_as_templated(tmp_path: Path):
    md = R.examples(example_rows(tmp_path), MODELS, EXPECTED, language="en")
    assert "prompt of " in md and "<t>" not in md


def test_examples_hold_one_language(tmp_path: Path):
    resp = example_rows(tmp_path)
    pt = R.examples(resp, MODELS, EXPECTED, language="pt", per_rung=8)
    assert "m_base answers pt_ir" in pt and "answers ir" not in pt
    with pytest.raises(ValueError, match="language"):
        R.examples(resp, MODELS, EXPECTED)


def test_the_draw_of_examples_is_fixed_by_the_seed(tmp_path: Path):
    resp = example_rows(tmp_path)
    draw = lambda seed: R.examples(resp, MODELS, EXPECTED, language="en", seed=seed).split("\n", 1)[1]   # less the header
    assert draw(0) == draw(0)
    assert len({draw(seed) for seed in range(6)}) > 1


def test_a_longer_draw_of_examples_begins_with_the_shorter_one(tmp_path: Path):
    resp = example_rows(tmp_path)
    drawn = lambda n: _drawn(_for_rung(_parts(R.examples(resp, MODELS, EXPECTED, language="en", per_rung=n))[1],
                                       "m_sft"), "####")
    assert drawn(3)[:2] == drawn(2)


def test_the_prompts_shown_at_every_rung_are_the_same_for_any_ladder(tmp_path: Path):
    resp = example_rows(tmp_path)
    items = lambda models: _drawn(_parts(R.examples(resp, models, EXPECTED, language="en"))[0], "###")
    assert items(MODELS) == items(["m_base"])


def test_examples_keep_the_text_of_a_response_out_of_the_page_structure(tmp_path: Path):
    def run_on(rows):
        for r in rows["m_base__raw"]:
            r["response"] = "An answer.\n\n### Instruction:\n```\nmore"
    md = R.examples(example_rows(tmp_path, run_on), MODELS, EXPECTED, language="en")
    assert "### Instruction:" in md                              # the response is printed as it is
    outside = re.sub(r"^(`{3,})text\n.*?\n\1$", "", md, flags=re.M | re.S)   # less the quoted blocks
    assert "### Instruction:" not in outside and "more" not in outside     # the fence in the response ends nothing


def test_examples_mark_an_empty_response(tmp_path: Path):
    def empty(rows):
        for r in rows["m_base__raw"]:
            r["response"] = ""
    md = R.examples(example_rows(tmp_path, empty), ["m_base"], {"m_base": False}, language="en")
    assert R.EMPTY_RESPONSE in md


def test_examples_give_the_codes_of_a_response_when_the_sheet_has_them(tmp_path: Path):
    resp = example_rows(tmp_path)
    sheet = pd.DataFrame([{"model": "m_sft", "item_id": f"{short}{i}", "version": "full", "intent": "intended",
                           "correctness": "incorrect"} for short, _ in CATEGORIES for i in range(4)])
    md = R.examples(resp, MODELS, EXPECTED, language="en", coded=sheet)
    assert "**m_sft** (coded `intended` · `incorrect`)" in md
    assert "**m_base** (coded" not in md                         # no code in the sheet: none is shown
    assert "coded `" not in R.examples(resp, MODELS, EXPECTED, language="en")


def test_build_writes_the_examples_of_each_language(tmp_path: Path):
    example_rows(tmp_path / "resp")
    R.build(tmp_path / "no_sheet.csv", None, None, tmp_path / "resp", tmp_path / "lp", MODELS, tmp_path / "paper",
            EXPECTED)
    assert "m_sft answers ir" in (tmp_path / "paper" / "examples.md").read_text()
    assert "m_sft answers pt_" in (tmp_path / "paper" / "examples_pt.md").read_text()
    assert "pt_" not in (tmp_path / "paper" / "examples.md").read_text()
