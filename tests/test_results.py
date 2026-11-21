from pathlib import Path
import pandas as pd
import pytest
from gricean import cli, results


def write_tables(paper: Path, ladder: str = "t", suffix: str = "", rate: float = 0.25) -> None:
    out = paper / ladder; out.mkdir(parents=True, exist_ok=True)
    idx = pd.MultiIndex.from_tuples(
        [("indirect_request", "control"), ("indirect_request", "full"), ("indirect_request", "stripped"),
         ("malformed_instruction", "full"), ("pattern_continuation", "full")], names=["category", "condition"])
    cols = pd.MultiIndex.from_product([["olmo_stage1", "s"], ["literal", "intended", "neither"]])
    intent = pd.DataFrame(0.0, index=idx, columns=cols)
    intent[("olmo_stage1", "intended")] = [0.5, rate, 0.375, 0.0, 0.375]
    intent[("s", "intended")] = [1.0, 1.0, 1.0, 0.75, 1.0]
    intent.to_csv(out / f"intent{suffix}.csv")
    cols = pd.MultiIndex.from_product([["olmo_stage1", "s"], ["mean", "ci_low", "ci_high", "share_neither"]])
    lo = pd.DataFrame(0.0, index=idx[1:], columns=cols)
    lo[("olmo_stage1", "mean")] = [0.434, 1.134, 2.39, -0.529]
    lo[("olmo_stage1", "ci_low")] = [-0.035, 0.62, 1.12, -0.88]
    lo[("olmo_stage1", "ci_high")] = [0.808, 1.59, 3.64, -0.116]
    lo.to_csv(out / f"logodds{suffix}.csv")


def rows(block: str) -> list[list[str]]:
    return [[c.strip() for c in line.strip("|").split("|")] for line in block.splitlines()
            if line.startswith("|") and not set(line) <= set("|- ")]


def test_parse_spec_reads_lists_and_defaults():
    spec = results.parse_spec("table=intent ladder=olmo condition=full,stripped")
    assert spec == {"table": "intent", "ladder": "olmo", "language": "en", "condition": ["full", "stripped"]}


@pytest.mark.parametrize("text", ["table=intent ladder=olmo", "table=nope ladder=olmo", "ladder=olmo condition=full",
                                  "table=intent ladder=olmo condition=full colour=red", "table=intent olmo condition=full",
                                  "table=kappa"])
def test_parse_spec_refuses_what_it_cannot_read(text):
    with pytest.raises(ValueError):
        results.parse_spec(text)


def test_intent_block_gives_counts_labels_and_a_total(tmp_path):
    write_tables(tmp_path)
    block = results.render_block(results.parse_spec("table=intent ladder=t condition=full"), tmp_path)
    assert block.splitlines()[0] == ("Line 1. English, `full`: responses coded `intended`, "
                                     "of the items in each cell (`paper/t/intent.csv`).")
    assert rows(block) == [["Category", "Stage 1", "s"],
                           ["Indirect request", "2/8", "8/8"],
                           ["Malformed instruction", "0/8", "6/8"],
                           ["Pattern continuation", "3/8", "8/8"],
                           ["All 3 categories", "5/24", "22/24"]]
    widths = {len(line) for line in block.splitlines() if line.startswith("|")}
    assert len(widths) == 1                                        # padded: the source reads as a table too


def test_intent_block_uses_the_control_quota_and_no_total_for_one_row(tmp_path):
    write_tables(tmp_path)
    block = results.render_block(results.parse_spec("table=intent ladder=t condition=control"), tmp_path)
    assert rows(block) == [["Category", "Stage 1", "s"], ["Indirect request", "1/2", "2/2"]]


def test_several_conditions_label_each_row_and_total_each_condition(tmp_path):
    write_tables(tmp_path)
    block = results.render_block(results.parse_spec(
        "table=intent ladder=t condition=full,stripped category=indirect_request models=s,olmo_stage1"), tmp_path)
    assert rows(block) == [["Category", "s", "Stage 1"],
                           ["Indirect request, `full`", "8/8", "2/8"],
                           ["Indirect request, `stripped`", "8/8", "3/8"]]


def test_portuguese_reads_its_own_file_and_quota(tmp_path):
    write_tables(tmp_path, suffix="_pt", rate=0.5)
    block = results.render_block(results.parse_spec(
        "table=intent ladder=t language=pt condition=full category=indirect_request"), tmp_path)
    assert "Portuguese" in block and "`paper/t/intent_pt.csv`" in block
    assert rows(block)[1] == ["Indirect request", "1/2", "2/2"]


def test_a_rate_that_is_not_a_count_is_refused(tmp_path):
    write_tables(tmp_path, rate=0.3)
    with pytest.raises(ValueError, match="indirect_request.*full.*olmo_stage1"):
        results.render_block(results.parse_spec("table=intent ladder=t condition=full"), tmp_path)


def test_logodds_block_gives_mean_and_interval(tmp_path):
    write_tables(tmp_path)
    block = results.render_block(results.parse_spec("table=logodds ladder=t condition=full models=olmo_stage1"), tmp_path)
    assert rows(block) == [["Category", "Stage 1"],
                           ["Indirect request", "0.43 [−0.04, 0.81]"],
                           ["Malformed instruction", "2.39 [1.12, 3.64]"],
                           ["Pattern continuation", "−0.53 [−0.88, −0.12]"]]


@pytest.mark.parametrize("text", ["table=intent ladder=absent condition=full", "table=intent ladder=t condition=embedded",
                                  "table=intent ladder=t condition=full models=nobody",
                                  "table=intent ladder=t condition=stripped category=malformed_instruction"])
def test_a_block_that_asks_for_what_is_not_there_is_refused(tmp_path, text):
    write_tables(tmp_path)
    with pytest.raises(ValueError):
        results.render_block(results.parse_spec(text), tmp_path)
KAPPA = ("cohen_kappa_intent=0.775 n=335 same=290\n"
         "cohen_kappa_correctness_same_intent=0.855 n=290 same=269\n"
         "cohen_kappa_correctness_all=0.833 n=335 same=307\n")
TERMINATION = ("# turn completion (spec 13.1): post hoc, English coded rows only\n"
               "limit=150 tolerance=0\n"
               "complete_turns olmo_n500=31/48 olmo_full=32/48\n"
               "# exploratory: read as fixed beforehand in spec 13.1\n"
               "olmo_n500 vs olmo_full: n=48 both=28 only_first=3 only_second=4 neither=13 mcnemar_exact_p=1.0000\n")


def test_kappa_block_lists_the_three_measures(tmp_path):
    (tmp_path / "t").mkdir(); (tmp_path / "t/kappa.txt").write_text(KAPPA)
    block = results.render_block(results.parse_spec("table=kappa ladder=t"), tmp_path)
    assert block.splitlines()[0] == "Agreement between the two coders, one sample for both ladders (`paper/t/kappa.txt`)."
    assert block.splitlines()[2:] == ["- `intent`: κ = 0.775 (n = 335, same code on 290)",
                                      "- `correctness`, rows with the same `intent`: κ = 0.855 (n = 290, same code on 269)",
                                      "- `correctness`, all rows: κ = 0.833 (n = 335, same code on 307)"]


def test_termination_block_gives_complete_turns_and_the_comparison(tmp_path):
    (tmp_path / "t").mkdir(); (tmp_path / "t/termination.txt").write_text(TERMINATION)
    block = results.render_block(results.parse_spec("table=termination ladder=t"), tmp_path)
    assert "limit of 150 tokens (`paper/t/termination.txt`)" in block.splitlines()[0]
    assert rows(block) == [["", "500", "All pairs"], ["Complete turns", "31/48", "32/48"]]
    assert block.splitlines()[-1] == ("Exploratory comparison fixed in advance, 500 against All pairs: "
                                      "3 items one way, 4 the other; exact McNemar p = 1.00.")


def test_termination_block_without_a_comparison_ends_at_the_table(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t/termination.txt").write_text("\n".join(TERMINATION.splitlines()[:3]) + "\n")
    block = results.render_block(results.parse_spec("table=termination ladder=t"), tmp_path)
    assert block.splitlines()[-1].startswith("| Complete turns")


def test_text_blocks_refuse_a_file_they_cannot_read(tmp_path):
    (tmp_path / "t").mkdir(); (tmp_path / "t/termination.txt").write_text("limit=150 tolerance=0\n")
    with pytest.raises(ValueError):
        results.render_block(results.parse_spec("table=termination ladder=t"), tmp_path)
    with pytest.raises(ValueError):
        results.render_block(results.parse_spec("table=kappa ladder=t"), tmp_path)
    with pytest.raises(ValueError):
        results.parse_spec("table=kappa ladder=t condition=full")


def test_a_comparison_line_without_its_counts_is_refused(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t/termination.txt").write_text(TERMINATION.replace("only_first=3 ", ""))
    with pytest.raises(ValueError, match="only_first"):
        results.render_block(results.parse_spec("table=termination ladder=t"), tmp_path)
def marker(spec: str) -> str:
    return f"{results.OPEN_PREFIX}{spec} -->"


PAGE = "\n".join([
    "# Results", "", "Intro the author may edit.", "",
    marker("table=summary"), "stale summary", results.CLOSE, "",
    "## Prediction 1", "",
    marker("table=intent ladder=t condition=full"), results.CLOSE, "",
    "### 1a. Low at Stage 1", "Read the Stage 1 column.",
    "- Support: table reading", "- Line 1 verdict:", "- Line 2 verdict:  Held ", "- Decided on:",
    "- Note: a note", "  that runs over two lines", "",
    "### 2. Controls stay high", "- Support: table reading", "- Line 1 verdict: failed", "- Decided on:", "- Note:", "",
    "## Decisions", "", "### D1. The Instruct column", "- Decision:", "- Decided on:", "- Reason:", "",
    "### D2. Base", "- Decision: a dose note", "- Decided on: 2026-11-22", "- Reason: it barely moves", ""])


def outside(page: str) -> list[str]:
    kept, inside = [], False
    for line in page.splitlines():
        if line.startswith(results.OPEN_PREFIX):
            inside = True; kept.append(line)
        elif line == results.CLOSE:
            inside = False; kept.append(line)
        elif not inside:
            kept.append(line)
    return kept


def test_refresh_fills_the_blocks_and_keeps_every_other_line(tmp_path):
    write_tables(tmp_path)
    new = results.refresh(PAGE, tmp_path)
    assert outside(new) == outside(PAGE)                           # the author's text, byte for byte
    assert "stale summary" not in new
    assert "| Indirect request" in new and "2/8" in new
    assert new.endswith(PAGE[-20:])                                # the end of the file is not touched either
    assert results.refresh(new, tmp_path) == new                   # a second run changes nothing


def test_summary_shows_verdicts_open_ones_and_decisions(tmp_path):
    summary = results.render_summary(PAGE)
    assert rows(summary) == [["Claim", "Line 1", "Line 2", "Support"],
                             ["1a Low at Stage 1", "open", "held", "table reading"],
                             ["2 Controls stay high", "failed", "—", "table reading"],
                             ["Decision", "Status"],
                             ["D1 The Instruct column", "open"],
                             ["D2 Base", "a dose note"]]


def test_summary_of_a_page_without_claims():
    assert results.render_summary("# Results\n") == "No claim on this page yet."


@pytest.mark.parametrize("change, message", [
    (lambda p: p.replace("- Line 1 verdict: failed", "- Line 1 verdict: yes"), r"line 25: .*held, failed or mixed"),
    (lambda p: p.replace(results.CLOSE + "\n\n## Prediction 1", "\n## Prediction 1"), r"line 10: .*line 5"),
    (lambda p: p.replace("## Decisions", results.CLOSE), r"line 29: "),
    (lambda p: p + marker("table=kappa ladder=t") + "\n", r"line 40: "),
    (lambda p: p.replace("generated by `gricean results`: table=summary", "generated by gricean results: table=summary"),
     r"line 5: "),
    (lambda p: p.replace("condition=full", "condition=embedded"), r"line 11: "),
    (lambda p: p.replace("### 2. Controls stay high", "### 1a. Controls stay high"), r"line 14.*line 23|line 23.*line 14"),
])
def test_refresh_refuses_and_names_the_line(tmp_path, change, message):
    write_tables(tmp_path)
    with pytest.raises(ValueError, match=message):
        results.refresh(change(PAGE), tmp_path)


def test_update_writes_once_checks_without_writing_and_never_writes_on_error(tmp_path):
    write_tables(tmp_path / "paper")
    page = tmp_path / "paper/RESULTS.md"; page.write_text(PAGE)
    assert results.update(page, tmp_path / "paper", check=True) is False
    assert page.read_text() == PAGE                                # a check leaves the file alone
    assert results.update(page, tmp_path / "paper") is False
    filled = page.read_text()
    assert filled != PAGE
    assert results.update(page, tmp_path / "paper") is True and page.read_text() == filled
    page.write_text(filled.replace("- Line 1 verdict: failed", "- Line 1 verdict: maybe"))
    broken = page.read_text()
    with pytest.raises(ValueError):
        results.update(page, tmp_path / "paper")
    assert page.read_text() == broken
def test_results_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["results"]) == 1 and "written by hand" in capsys.readouterr().out
    write_tables(tmp_path / "paper")
    (tmp_path / "paper/RESULTS.md").write_text(PAGE)
    assert cli.main(["results", "--check"]) == 1 and "out of date" in capsys.readouterr().out
    assert cli.main(["results"]) == 0 and "rewritten" in capsys.readouterr().out
    assert cli.main(["results"]) == 0 and "up to date" in capsys.readouterr().out
    assert cli.main(["results", "--check"]) == 0
    page = tmp_path / "paper/RESULTS.md"
    page.write_text(page.read_text().replace("- Line 1 verdict: failed", "- Line 1 verdict: maybe"))
    before = page.read_text()
    assert cli.main(["results"]) == 1 and "held, failed or mixed" in capsys.readouterr().out
    assert page.read_text() == before


# What follows was added after the review of the branch: the summary must never show a verdict the author did
# not write, so a line the page cannot attach to a claim is refused, not skipped.

@pytest.mark.parametrize("change, message", [
    (lambda p: p.replace("- Line 1 verdict: failed", "- Line 1 verdict: failed\n- Line 1 verdict: held"), r"line 26: .*line 25"),
    (lambda p: p.replace("- Line 1 verdict: failed", "- Line 1 Verdict: failed"), r"line 25: .*- Line 1 verdict:"),
    (lambda p: p.replace("- Line 1 verdict: failed", "* Line 1 verdict: failed"), r"line 25: .*- Line 1 verdict:"),
    (lambda p: p.replace("- Line 1 verdict: failed", "- Line 1 verdict failed"), r"line 25: .*- Line 1 verdict:"),
    (lambda p: p.replace("- Line 1 verdict: failed", "- **Line 1 verdict:** failed"), r"line 25: .*- Line 1 verdict:"),
    (lambda p: p.replace("- Line 1 verdict: failed", "-Line 1 verdict: failed"), r"line 25: .*- Line 1 verdict:"),
    (lambda p: p.replace("- Line 1 verdict: failed", "- Line 1 verdict : failed"), r"line 25: .*- Line 1 verdict:"),
    (lambda p: p.replace("- Line 1 verdict:\n", "- Line 1 verdict:\n  held\n", 1), r"line 18: .*line 17"),
    (lambda p: p.replace("### 2. Controls stay high", "### 2 Controls stay high"), r"line 23: .*### 1a\. Title"),
    (lambda p: p.replace("### 2. Controls stay high", "#### 2. Controls stay high"), r"line 23: .*### 1a\. Title"),
    (lambda p: p.replace("### 2. Controls stay high", "## 2. Controls stay high"), r"line 23: .*### 1a\. Title"),
    (lambda p: p.replace("### 1a. Low at Stage 1", "### 1A. Low at Stage 1"), r"line 14: .*### 1a\. Title"),
    (lambda p: p.replace("- Reason:\n", "- Reason:\n- Line 1 verdict: held\n", 1), r"line 35: .*under a claim"),
    (lambda p: p.replace("Read the Stage 1 column.", "#### Remarks"), r"line 17: .*under a claim"),
    (lambda p: p.replace("- Decision: a dose note", "- Decision : a dose note"), r"line 37: .*- Decision:"),
])
def test_a_line_the_summary_could_not_place_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        results.render_summary(change(PAGE))


def test_headings_and_code_are_read_as_markdown_reads_them():
    page = PAGE.replace("### 2. Controls stay high", " ### 2. Controls stay high").replace(
        "Read the Stage 1 column.", "```\n# a comment in code\n- Line 1 verdict: banana\n```\n#3 is not a heading")
    assert rows(results.render_summary(page)) == rows(results.render_summary(PAGE))


def test_a_mistyped_opening_marker_is_pointed_to(tmp_path):
    write_tables(tmp_path)
    with pytest.raises(ValueError, match=r"line 7: .*opening line above"):
        results.refresh(PAGE.replace("<!-- generated by `gricean results`: table=summary", "<!-- generatd: table=summary"), tmp_path)


def test_a_value_just_below_zero_keeps_its_sign(tmp_path):
    write_tables(tmp_path)
    csv = tmp_path / "t/logodds.csv"
    csv.write_text(csv.read_text().replace("-0.116", "-0.004"))
    block = results.render_block(results.parse_spec("table=logodds ladder=t condition=full models=olmo_stage1"), tmp_path)
    assert rows(block)[3] == ["Pattern continuation", "−0.53 [−0.88, −0.00]"]    # not 0.00: the interval stays below zero


def test_results_command_says_what_it_read(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_tables(tmp_path / "paper")
    (tmp_path / "paper/RESULTS.md").write_text(PAGE)
    assert cli.main(["results"]) == 0
    assert "2 claims, 2 decisions, 2 generated blocks" in capsys.readouterr().out
