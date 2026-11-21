# Results Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `paper/RESULTS.md`, a Markdown page in which the tables are regenerated from the committed CSV files and the verdicts and decisions are written by the author as plain text.

**Architecture:** One module, `gricean.results`. It reads a page, replaces the text between each pair of marker lines with a table built from the files under `paper/<ladder>/`, and leaves every other character as it found it. The opening marker names the table and the cells. A `summary` block is built from the verdict lines the author fills in. A CLI command wraps it; the page itself is written by hand.

**Tech Stack:** Python, pandas (already a dependency), pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-21-results-page-design.md`. Main spec: `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md` (§5.4, §11, §13.1).

## Global Constraints

- Branch `results-page`; one pull request into `main`, merged with a merge commit. The author merges.
- Every number on the page is read from a file under `paper/`; nothing is recomputed from `data/`.
- No verdict and no decision is written by the builder. Those lines are committed empty.
- No pooled number across languages; no pooled Line 2 mean; the only sum is the Line 1 total over categories.
- Tests build their page and their tables in `tmp_path`. No test reads the real `paper/`, `stimuli/` or `configs/`.
- No notebook changes. `uv run pytest` must pass after every task.
- Commit messages follow the repository's style (a plain sentence, no `feat:` prefix) and end with the `Co-Authored-By` line.
- Order of commits: design and plan; module and tests (Tasks 1–4); the page with empty blocks and the note in the main spec; the page as filled by the first run; the pointers in the other documents.

## Review Focus

1. A verdict typed as `Held` or with a trailing space is accepted and shown as `held`. (Task 3)
2. A closing marker deleted by accident, or an opening marker with a typo, stops the command with the line number and leaves the file unchanged. (Task 3)
3. A claim section copied and not renumbered (two headings with the same id) is refused, naming both lines. (Task 3)
4. A rate that is not a whole count out of the cell's number of items is refused, not rounded. (Task 1)
5. A `Note:` that runs over several lines, and any text the author adds between sections, survives a run byte for byte. (Task 3)

---

### Task 1: Table blocks for Line 1 and Line 2

**Files:**
- Create: `src/gricean/results.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: `gricean.stimuli.QUOTAS` (`{"en": (8, 2), "pt": (2, 1)}`: items per category, dissociation then control); the CSV layout written by `report.build` (two header rows `model`, `measure`; index `category`, `condition`; file `paper/<ladder>/<table>.csv` for English and `<table>_<language>.csv` otherwise).
- Produces:
  - `parse_spec(text: str) -> dict` — `"table=intent ladder=olmo condition=full,stripped"` becomes `{"table": "intent", "ladder": "olmo", "language": "en", "condition": ["full", "stripped"]}`. Keys allowed: `table`, `ladder`, `language`, `condition`, `category`, `models`. `condition`, `category` and `models` are lists. `language` defaults to `"en"`. Raises `ValueError` on an unknown key, a word without `=`, a missing `table`, an unknown table, or a key the table requires and does not have (`ladder` for every table but `summary`; `condition` for `intent` and `logodds`).
  - `TABLES = ("intent", "logodds", "kappa", "termination", "summary")`
  - `MODEL_LABELS: dict[str, str]` — `olmo_stage1` "Stage 1", `olmo_base` "Base", `olmo_sft` "SFT", `olmo_dpo` "DPO", `olmo_instruct` "Instruct", `olmo_stage1_alpaca` "Stage 1 Alpaca", `olmo_n500` "500", `olmo_n2000` "2,000", `olmo_full` "All pairs", `olmo_format_only` "Format only". A model not in the dict is shown under its own name.
  - `render_block(spec: dict, paper: Path) -> str` — the Markdown of one block, without the markers and without a final line break. Raises `ValueError` when the file is missing or a condition, category or model asked for is not in it.

**Rules the tests pin:**
- A block is one caption line, one empty line, one table. Columns are padded with spaces to equal width.
- Caption of `intent`: ``Line 1. English, `full`: responses coded `intended`, of the items in each cell (`paper/olmo/intent.csv`).`` Caption of `logodds`: ``Line 2. English, `full`: mean log-odds of the intended over the literal continuation, with its bootstrap interval (`paper/olmo/logodds.csv`).`` Several conditions are joined with commas. `pt` reads "Portuguese".
- First column header `Category`; the others are the model labels, in the file's order, or in the order of `models=` when given.
- Row label: the category with underscores as spaces and a capital first letter. With more than one condition in the marker: ``Indirect request, `stripped` ``. Rows follow the file's category order, then the marker's condition order. A category that lacks a condition is skipped unless `category=` names it, in which case it is an error.
- `intent` cell: `k/n`, with `n` from `QUOTAS[language]` (second value when the condition is `control`) and `k = rate × n`. If `k` is further than `1e-6` from a whole number: `ValueError` naming file, category, condition and model. An empty rate shows `—`.
- `intent` total: for each condition with more than one row, a last row ``All 6 categories`` (or ``All 3 categories, `stripped` `` when the marker has several conditions) with the sums. A column with an empty rate has `—` in the total.
- `logodds` cell: `mean [ci_low, ci_high]`, two decimals, U+2212 for the minus sign. No total row.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
import pandas as pd
import pytest
from gricean import results


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
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_results.py -q`
Expected: FAIL, `cannot import name 'results'`.

- [ ] **Step 3: Implement `parse_spec`, `MODEL_LABELS`, `TABLES` and `render_block` (for `intent` and `logodds`) in `src/gricean/results.py`**

Read the CSV with `pd.read_csv(path, header=[0, 1], index_col=[0, 1])`. `render_block` dispatches on `spec["table"]`; `kappa`, `termination` and `summary` come in Tasks 2 and 3.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_results.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/results.py tests/test_results.py
git commit -m "Results page: Line 1 and Line 2 blocks read from the committed tables"
```

---

### Task 2: Blocks for coder agreement and turn completion

**Files:**
- Modify: `src/gricean/results.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: `render_block`, `MODEL_LABELS` (Task 1); `paper/<ladder>/kappa.txt` (lines `name=value n=N same=S`) and `paper/<ladder>/termination.txt` (a line `limit=150 tolerance=0`, a line `complete_turns model=k/n ...`, and on some ladders a line `A vs B: n=48 both=28 only_first=3 only_second=4 neither=13 mcnemar_exact_p=1.0000`).
- Produces: `render_block` accepts `table=kappa ladder=L` and `table=termination ladder=L`. Neither takes `condition`, `category`, `models` or `language`; giving one is a `ValueError` from `parse_spec`.

**Rules the tests pin:**
- `kappa`: caption ``Agreement between the two coders, one sample for both ladders (`paper/t/kappa.txt`).``, an empty line, then one list item per line of the file: ``- `intent`: κ = 0.775 (n = 335, same code on 290)``. Names: `cohen_kappa_intent` is `` `intent` ``; `cohen_kappa_correctness_same_intent` is `` `correctness`, rows with the same `intent` ``; `cohen_kappa_correctness_all` is `` `correctness`, all rows ``; any other name is shown as it is.
- `termination`: caption ``Post hoc (spec §13.1). English, `full`, dissociation items: responses coded `intended` that ended before the limit of 150 tokens (`paper/t/termination.txt`).``, an empty line, a table with one row `Complete turns` and one column per model, cells `k/n`. If the comparison line is there, an empty line and: `Exploratory comparison fixed in advance, 500 against All pairs: 3 items one way, 4 the other; exact McNemar p = 1.00.`
- A file without the line the block needs is a `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_results.py -q -k "kappa or termination or text_blocks"`
Expected: FAIL.

- [ ] **Step 3: Implement the two tables in `render_block`, and the extra refusal in `parse_spec`**

- [ ] **Step 4: Run the whole file and see it pass**

Run: `uv run pytest tests/test_results.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/results.py tests/test_results.py
git commit -m "Results page: blocks for coder agreement and turn completion"
```

---

### Task 3: The page: markers, verdict lines, summary

**Files:**
- Modify: `src/gricean/results.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: `parse_spec`, `render_block` (Tasks 1 and 2).
- Produces:
  - `OPEN_PREFIX = "<!-- generated by `gricean results`: "` and `CLOSE = "<!-- end of generated block: edits above this line are overwritten -->"`. An opening marker is a whole line: `OPEN_PREFIX`, the spec, ` -->`.
  - `VERDICTS = ("held", "failed", "mixed")`
  - `render_summary(page: str) -> str` — the Markdown of the summary block.
  - `refresh(page: str, paper: Path) -> str` — the page with every block regenerated. Raises `ValueError` whose message begins `line N:`.
  - `update(page_path: Path, paper: Path, check: bool = False) -> bool` — `True` when the file already held what `refresh` gives. Writes only when it did not and `check` is false. Writes nothing when `refresh` raises.

**Rules the tests pin:**
- Between an opening marker and its `CLOSE`, the old text is dropped and replaced by: the block, preceded and followed by one empty line. Everything outside the pairs is returned unchanged, byte for byte.
- Errors, all with the line number: an opening marker before the previous one is closed; a `CLOSE` with no opening; an opening never closed; a line that starts with `<!-- generated` and is not a well-formed opening marker; any `ValueError` from `parse_spec` or `render_block`.
- A claim is a heading `### <id>. <title>` with `<id>` matching `\d+[a-z]?`. A decision is a heading `### D<number>. <title>`. A section runs to the next line that starts with `#`. Text inside generated blocks is not searched.
- In a claim section, the lines `- Line 1 verdict:`, `- Line 2 verdict:` and `- Support:` are read. A verdict is compared without regard to case and to spaces around it. It must be empty or one of `VERDICTS`; otherwise `ValueError` with the line number and the words allowed. In a decision section, `- Decision:` is read.
- Two headings with the same id: `ValueError` naming both line numbers.
- Summary: a table `Claim | Line 1 | Line 2 | Support`, one row per claim in page order, the claim shown as `1a Low at Stage 1`. An empty verdict shows `open`; a claim without that line shows `—`. If the page has decisions, an empty line and a table `Decision | Status`, the status being the text after `- Decision:` or `open`. A page with neither gives the single line `No claim on this page yet.`

- [ ] **Step 1: Write the failing tests**

```python
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
```

The line numbers in the parametrized test count from 1 in `PAGE` as written above; recount them if the fixture changes.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_results.py -q -k "refresh or summary or update"`
Expected: FAIL.

- [ ] **Step 3: Implement `OPEN_PREFIX`, `CLOSE`, `VERDICTS`, `render_summary`, `refresh` and `update`; make `render_block` refuse `table=summary` (the summary needs the page, so `refresh` calls `render_summary` itself)**

`refresh` works line by line and keeps the page's final line break as it found it.

- [ ] **Step 4: Run the whole file and see it pass**

Run: `uv run pytest tests/test_results.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/results.py tests/test_results.py
git commit -m "Results page: markers, verdict lines and the summary; the author's text is never touched"
```

---

### Task 4: The command

**Files:**
- Modify: `src/gricean/cli.py` (import `results`; `cmd_results`; one parser), `Makefile` (`report` target)
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: `results.update(page_path, paper, check)` (Task 3).
- Produces: `gricean results [--check]`. Page `paper/RESULTS.md`, tables under `paper/`.

**Rules the tests pin:**
- No page: prints `paper/RESULTS.md is not there; it is written by hand (see the design of 2026-11-21)` and returns 1.
- `ValueError` from `update`: prints `results: <message>` and returns 1. The file is unchanged.
- Without `--check`: returns 0 and prints `results: paper/RESULTS.md rewritten` or `results: paper/RESULTS.md is up to date`.
- With `--check`: returns 0 when up to date; otherwise prints `results: paper/RESULTS.md is out of date; run `gricean results`` and returns 1.

- [ ] **Step 1: Write the failing tests**

```python
from gricean import cli


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
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_results.py::test_results_command -q`
Expected: FAIL (exit code 2, unknown subcommand).

- [ ] **Step 3: Add `cmd_results(args, loader) -> int` and its parser to `src/gricean/cli.py`; in `Makefile`, give `report` a second line `uv run gricean results`**

- [ ] **Step 4: Run the fast suite**

Run: `uv run pytest -q`
Expected: PASS, every test.

- [ ] **Step 5: Commit**

```bash
git add src/gricean/cli.py Makefile tests/test_results.py
git commit -m "gricean results: rewrite the generated blocks of paper/RESULTS.md, or check them"
```

---

### Task 5: The page, the note in the spec, the pointers

**Files:**
- Create: `paper/RESULTS.md`
- Modify: `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md` (revision log, §11), `paper/WORKFLOW.md` (§1, §3), `README.md` (step 9), `CLAUDE.md` (status)

**Interfaces:**
- Consumes: the marker lines and the claim and decision line formats of Task 3; `gricean results`.

**The page, section by section.** Every verdict, `Decided on`, `Note`, `Decision` and `Reason` line is committed empty. Plain sentences; no result is stated in the hand-written text.

| Section | Generated blocks (marker specs) | Hand-written |
|---|---|---|
| Title and "How to use this page" | — | Four lines: write only on lines that end in a colon; a verdict is `held`, `failed` or `mixed`; tables are regenerated by `make report` and anything typed inside one is lost; the split into claims and the choice of cells were made after the results (pointer to the spec note). |
| Summary | `table=summary` | — |
| Prediction 1: the ladder | `table=intent ladder=olmo condition=full` · `table=logodds ladder=olmo condition=full` · under "Portuguese (generalization check, two items per category)": `table=intent ladder=olmo language=pt condition=full` · `table=logodds ladder=olmo language=pt condition=full` | The first bullet of §5.4 quoted. Claims 1a–1e. |
| Prediction 2: controls | `table=intent ladder=olmo condition=control` · Portuguese: `table=intent ladder=olmo language=pt condition=control` | Second bullet quoted. Claim 2 (Line 1 verdict only; say that Line 2 has no control rows). |
| Prediction 3: cue ablation | `table=intent ladder=olmo condition=full,stripped,embedded category=indirect_request,embedded_instruction,pattern_continuation` · the same for `logodds` · Portuguese: the same two with `language=pt` | Third bullet quoted. Claim 3. One sentence: `embedded` is never chat-templated (spec §4.3). Pointer: revision log of 2026-11-14. |
| Prediction 4: the self-run adapters | `table=intent ladder=selfrun condition=full` · `table=logodds ladder=selfrun condition=full` · "Official ladder, for comparison": `table=intent ladder=olmo condition=full models=olmo_stage1,olmo_sft` · `table=logodds ladder=olmo condition=full models=olmo_stage1,olmo_sft` · Portuguese: the first two with `language=pt` | Fourth bullet quoted. Claims 4a–4e. Pointer for 4e: spec §7 and `CLAUDE.md` record no growth on Line 1. |
| Decisions (spec §11) | — | D1 the Instruct column; D2 Base as a column or a dose note; D3 the pairs from Ouyang et al. (2022). Each: the question as §11 asks it, where to look (the tables of Prediction 1; for D3, `examples.md`), then `- Decision:`, `- Decided on:`, `- Reason:`. |
| Analyses added after the results | `table=termination ladder=olmo` · `table=termination ladder=selfrun` | Two sentences from spec §13.1: post hoc, English only; on the official ladder it reads from Stage 1 to SFT only, because long answers of DPO and Instruct reach the limit. Pointer to `termination_correctness.csv`. |
| Agreement between the coders | `table=kappa ladder=olmo` | One sentence: coder 2 is an instruction-tuned model (spec §5.1). |
| What no table tests | — | The open gaps: no paired test of one rung against another; none of data scale on `correct`; turn completion past SFT; the Line 2 table prints `share_neither` only. |
| Questions from exploration | — | The rule of `WORKFLOW.md` §5 in two sentences, and the one question already listed, as a question, with `- Outcome:` empty. |
| What the paper must disclose | — | The five items of `WORKFLOW.md` §6. |
| Where the data comes from | — | Commands (`make report`, `uv run gricean report --ladder selfrun`, `uv run gricean results`); pointers to `stimuli/CHECKSUMS`, `configs/models.yaml`, `configs/generation.yaml`, `gricean.report.EXAMPLES`. No hash copied. |

Claims, with the text of their fixed lines:

| Id | Heading | "Read" sentence | Support | Verdict lines |
|---|---|---|---|---|
| 1a | Low at Stage 1 | The Stage 1 column. The spec sets no threshold for "low". | table reading | 1, 2 |
| 1b | Moves little or not at all at Base | Stage 1 against Base, on the same rows. | table reading; no paired test exists | 1, 2 |
| 1c | Rises sharply at SFT | Base against SFT. | table reading; no paired test exists | 1, 2 |
| 1d | Rises or holds at DPO | SFT against DPO. | table reading; no paired test exists | 1, 2 |
| 1e | Stays flat at Instruct | DPO against Instruct. | table reading; no paired test exists | 1, 2 |
| 2 | Controls stay high across stages | Every column. Two control items per category. The spec sets no threshold for "high". | table reading | 1 |
| 3 | Cue ablation degrades Stage 1 and Base more than SFT, DPO and Instruct | For each rung, `full` against `stripped` and against `embedded`. | table reading; no paired test exists | 1, 2 |
| 4a | Clear but smaller effect than official SFT | Stage 1 Alpaca against the three adapters; then the adapters against SFT in the comparison table. | table reading; no paired test exists | 1, 2 |
| 4b | Present on indirect request and pattern continuation | Those two rows. | table reading; no paired test exists | 1, 2 |
| 4c | Weaker on malformed and underspecified | Those two rows against the two of 4b. | table reading; no paired test exists | 1, 2 |
| 4d | Absent in the format-only control | The Format only column against Stage 1 Alpaca. | table reading; no paired test exists | 1, 2 |
| 4e | Grows with data scale | 500, 2,000 and All pairs, in that order. | table reading; the one paired test run (spec §13.1) is on turn completion, a different measure | 1, 2 |

**The note in the main spec.** A revision-log entry dated 2026-11-21, after the last entry, and one sentence at the end of §11:

- Log: `paper/RESULTS.md` records the outcome of each prediction of §5.4 and the decisions of §11. Its tables are generated from the committed tables; the verdicts and decisions are the author's. Made after the results, and said so on the page: the split of the four predictions into twelve claims, the choice of the cells shown under each, and one sum (the Line 1 count over the categories of a condition). No new measure and no test.
- §11: the three open decisions are recorded in `paper/RESULTS.md` when taken.

- [ ] **Step 1: Write `paper/RESULTS.md` with every generated block empty (opening marker, `CLOSE` on the next line) and add the note to the main spec**

- [ ] **Step 2: Check that the page parses**

Run: `uv run gricean results --check`
Expected: exit 1, `results: paper/RESULTS.md is out of date; run `gricean results`` (a parse error would print a line number instead).

- [ ] **Step 3: Commit the page before it holds a number**

```bash
git add paper/RESULTS.md docs/superpowers/specs/2026-09-26-gricean-simulation-design.md
git commit -m "paper/RESULTS.md: claims, cells and blank verdicts, before any table is filled in"
```

- [ ] **Step 4: Fill it in, and check the result**

Run: `uv run gricean results && uv run gricean results --check && git diff --stat`
Expected: `rewritten`, exit 0 on the check, and a diff in `paper/RESULTS.md` only. Read the page: the "All 6 categories" row of Prediction 1 reads 7/48, 11/48, 41/48, 41/48, 41/48; of Prediction 4, 24/48, 37/48, 36/48, 37/48, 0/48; kappa on `intent` is 0.775. `git diff` shows no change on a line outside a generated block.

- [ ] **Step 5: Commit the page as filled**

```bash
git add paper/RESULTS.md
git commit -m "paper/RESULTS.md: tables as generated from the committed files"
```

- [ ] **Step 6: Point the other documents to the page**

`paper/WORKFLOW.md` §1: the verdicts are written in `RESULTS.md`; §3: so are the three decisions. `README.md` step 9: `make report` also rewrites the tables of `paper/RESULTS.md`; the text outside them is the author's. `CLAUDE.md`: one status entry (what the page is, the ownership rule, "do not write a verdict or a decision"), and "Next" updated.

- [ ] **Step 7: Run everything once more and commit**

Run: `uv run pytest -q && make report && git status --short`
Expected: tests pass; `make report` leaves `paper/` unchanged (the page is up to date and the tables are regenerated identically).

```bash
git add paper/WORKFLOW.md README.md CLAUDE.md
git commit -m "Docs: where the verdicts and the decisions are recorded"
```

- [ ] **Step 8: Push the branch and open the pull request; the author merges**
