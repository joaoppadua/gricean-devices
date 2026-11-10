# Pre-Run Tasks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Superseded in part by spec rev. 2 (2026-11-08):** Task 2 steps 5–6 are decided (Pythia-1.4b, by the documentary criterion in spec §5.3); Task 12 step 2 runs `make neither` for the `olmo` family only; Task 13 runs the `olmo` and `selfrun` ladders. See `2026-11-08-olmo-lineage-rev2.md`.

> **Status (2026-11-08, evening):** Tasks 1–4 and 12's code are merged (PR #2). Tasks 5–11 are in progress on
> branch `stimuli`: every category file is pre-filled with id-numbered skeletons whose fields read `TODO`
> (`stimuli.PLACEHOLDER`), and item `_01` + `_c01` of each file are written as the pattern. Placeholder items
> do not count toward quotas and block the full freeze and every model stage.

**Goal:** Take the pipeline from "seed dry run on 2 items" to "frozen 78-item stimulus set, pinned models, chosen `neither` source, ready for `make neither`".

**Architecture:** Three small code changes gate the authorial work: a quota check inside `freeze` so the stimulus set cannot be frozen incomplete, a revision pinner so `configs/models.yaml` stops saying `main`, and an explicit marker for empty greedy responses in printed examples. The rest is content: six English category files, six Portuguese mirrors, a `neither`-source pilot and the author's approval pass over flagged rows. Tasks 1 to 4 are code and can run in any order; Tasks 5 to 11 are the author's writing; Tasks 12 to 13 freeze and hand off to the run sequence.

**Tech Stack:** Python 3.12, PyYAML, pytest, Hugging Face Hub, uv. Nothing new.

**Spec:** `docs/superpowers/specs/2026-09-26-gricean-simulation-design.md` (§4 stimuli, §5.3 `neither`, §5.1 greedy examples, §3 pinned revisions). Prior plan: `docs/superpowers/plans/2026-09-26-gricean-simulation.md`.

## Global Constraints

- Open-source models only; everything runs on a 16 GB Apple M3 (spec §2, CLAUDE.md).
- Stimuli are frozen before any model run; every stage refuses drifted stimuli (spec §4.5).
- Item schema is fixed by `gricean.stimuli.Item`: `id`, `control`, `prompt`, `literal_reading`, `intended_reading`, `literal_continuation`, `intended_continuation`; files carry `category` and `language` at top level.
- Quotas (spec §4.1, §4.2, §4.4): English 8 dissociation + 2 controls per category; Portuguese 2 dissociation + 1 control per category. Six categories: `indirect_request`, `pattern_continuation`, `embedded_instruction`, `malformed_instruction`, `underspecified_request`, `rhetorical_question`.
- Cue ablation applies only to `indirect_request`, `embedded_instruction`, `pattern_continuation` (spec §4.3, `stimuli.ABLATED_CATEGORIES`).
- Portuguese items are written natively, never translated, and never pooled with English (spec §4.4).
- `neither` is sampled from a base model of a different family than the one scored (spec §5.3).
- Printed examples use the greedy response (spec §5.1); decoding knobs stay in `configs/generation.yaml`.
- No helper `def`s in `notebooks/*.py`; run `uv run marimo check notebooks/` after notebook edits.
- TDD: failing test first. Fast suite: `uv run pytest`.
- Id convention: `<lang>_<abbr>_NN` for dissociation, `<lang>_<abbr>_cNN` for controls, where `abbr` ∈ {`ir`, `pc`, `ei`, `mi`, `ur`, `rq`} in the category order above.

## Review Focus

- An ablated item whose prompt has no punctuation and no politeness marker produces `stripped == full`, making the ablation vacuous for that item. Expected: `freeze` rejects it. Pinned in Task 1.
- A Portuguese file that reuses an English id passes the duplicate check only if both are loaded; `validate` already catches this across files. Expected: rejection. Already covered by `test_validate_rejects_duplicate_ids`.
- `embedded` versions wrap a prompt ending in `?` in the middle of prose; a prompt that ends without terminal punctuation reads as a run-on sentence. Expected: author writes every ablated prompt with terminal punctuation. Pinned in Task 1 (terminal punctuation check for ablated items).
- `resolve_revision` prefers the local cache `refs/main`, which can lag the Hub. Expected: the pinner reports which source each sha came from so the author can tell. Pinned in Task 3.
- A greedy response that is empty (EOS first) prints as a blank line and looks like a bug in the paper. Expected: an explicit marker. Pinned in Task 4.

---

### Task 1: Quota and ablation checks in `freeze`

**Files:**
- Modify: `src/gricean/stimuli.py` (after `validate`, line 84 onward; `freeze`, line 101)
- Modify: `src/gricean/cli.py:34-36`, `:112`
- Test: `tests/test_stimuli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `Item`, `CATEGORIES`, `ABLATED_CATEGORIES`, `LANGUAGES`, `load_all`, `validate`.
- Produces: `QUOTAS: dict[str, tuple[int, int]] = {"en": (8, 2), "pt": (2, 1)}`; `check_quotas(items: list[Item]) -> None` raising `ValueError` listing every (language, category) that is short, as `"en/rhetorical_question: 3/8 dissociation, 0/2 controls"`; `check_ablation(items: list[Item]) -> None` raising `ValueError` for any non-control item in an ablated category whose `versions["stripped"] == versions["full"]` or whose prompt does not end in one of `?!.`; `freeze(root, *, partial: bool = False)` calling both unless `partial`.

- [ ] **Step 1: Write the failing tests**

```python
def test_check_quotas_reports_every_short_cell(root):
    items = S.load_all(root)
    with pytest.raises(ValueError) as e:
        S.check_quotas(items)
    msg = str(e.value)
    assert "en/indirect_request: 1/8 dissociation, 1/2 controls" in msg
    assert "pt/rhetorical_question: 0/2 dissociation, 0/1 controls" in msg

def test_check_ablation_rejects_vacuous_strip(root):
    write(root, "en/pattern_continuation.yaml", """
        category: pattern_continuation
        language: en
        items:
          - id: en_pc_01
            control: false
            prompt: "Explain the moon landing to a child"
            literal_reading: "list"
            intended_reading: "task"
            literal_continuation: "Explain gravity to a child"
            intended_continuation: "In 1969 people flew to the moon"
    """)
    with pytest.raises(ValueError, match="en_pc_01"):
        S.check_ablation(S.load_all(root))

def test_freeze_refuses_incomplete_set_unless_partial(root):
    with pytest.raises(ValueError, match="dissociation"):
        S.freeze(root)
    assert S.freeze(root, partial=True)
```

In `tests/test_cli.py`, change the two existing `cli.main(["freeze"])` calls (lines 29 and 56) to `cli.main(["freeze", "--partial"])` and add:

```python
def test_freeze_without_partial_fails_on_seed_set(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(project(tmp_path))
    with pytest.raises(ValueError, match="dissociation"):
        cli.main(["freeze"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_stimuli.py tests/test_cli.py -v`
Expected: FAIL with `AttributeError: module 'gricean.stimuli' has no attribute 'check_quotas'` and argparse error on `--partial`.

- [ ] **Step 3: Implement `QUOTAS`, `check_quotas`, `check_ablation`, and the `partial` flag**

`check_quotas` counts `(language, category, control)` over all 12 cells, not just the ones present, so missing files show as `0/8`. `check_ablation` tests `it.prompt.rstrip()[-1] in "?!."` and `it.versions["stripped"] != it.versions["full"]`. In `cli.py`, `sub.add_parser("freeze").add_argument("--partial", action="store_true")` and `cmd_freeze` passes `partial=args.partial`. Leave the Makefile `freeze` target strict.

- [ ] **Step 4: Run the full fast suite**

Run: `uv run pytest`
Expected: all pass. The seed-set tests now use `--partial`.

- [ ] **Step 5: Update README step 4 and commit**

README line 18: append "`uv run gricean freeze --partial` while the set is still being written."

```bash
git add src/gricean/stimuli.py src/gricean/cli.py tests/test_stimuli.py tests/test_cli.py README.md
git commit -m "feat: freeze enforces stimulus quotas and non-vacuous ablation"
```

---

### Task 2: `neither`-source pilot with a GPT-3-era base

**Files:**
- Modify: `configs/models.yaml` (add one model; `neither_source` stays unchanged until the author decides)
- Test: `tests/test_lm.py`

**Interfaces:**
- Consumes: `load_registry`, `neither_source`, `ladders`, `ModelSpec`.
- Produces: registry entry `pythia_base: {repo: EleutherAI/pythia-1.4b, revision: main, family: pythia, template: none}` that is in no ladder.

- [ ] **Step 1: Write the failing test**

```python
def test_registry_has_pythia_neither_candidate_outside_every_ladder():
    reg, lad = lm.load_registry(Path("configs/models.yaml")), lm.ladders(Path("configs/models.yaml"))
    assert reg["pythia_base"].template == "none"
    assert all("pythia_base" not in models for models in lad.values())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_lm.py -k pythia -v`
Expected: FAIL with `KeyError: 'pythia_base'`.

- [ ] **Step 3: Add the entry to `configs/models.yaml`**

Place it after `qwen_instruct`. Do not add it to any ladder.

- [ ] **Step 4: Run the test and commit**

Run: `uv run pytest tests/test_lm.py -k pythia -v`
Expected: PASS.

```bash
git add configs/models.yaml tests/test_lm.py
git commit -m "feat: register pythia-1.4b as neither-source candidate"
```

- [ ] **Step 5: Pilot on the frozen seed set (author, ~10 min incl. 2.9 GB download)**

Copy the current Qwen-sourced file aside, point OLMo's `neither_source` at the candidate, resample, and compare:

```bash
cp stimuli/neither/olmo.jsonl /tmp/olmo_neither_qwen.jsonl
sed -i '' 's/^  olmo: qwen_base$/  olmo: pythia_base/' configs/models.yaml
uv run gricean neither --family olmo
python -c "import json;[print(r['flag'], r['overlap'], r['samples']) for r in map(json.loads, open('stimuli/neither/olmo.jsonl'))]"
```

Decision rule: pick the source whose three samples on `en_ir_01` do **not** state the year. If Pythia also answers, keep Qwen and rely on the approval pass (Task 12); the circularity argument in spec §5.3 is about family, not about never answering.

- [ ] **Step 6: Record the decision**

If Pythia wins: keep the `sed` change, and change the §5.3 table row "OLMo 2 ladder | Qwen2.5-1.5B base" to "OLMo 2 ladder | Pythia-1.4b" in the spec, with one sentence on why (Qwen2.5 base already answers indirect requests). If Qwen stays: `git checkout configs/models.yaml stimuli/neither/olmo.jsonl`. Either way, update CLAUDE.md open task (2) to "decided: <source>".

```bash
git add configs/models.yaml stimuli/neither/olmo.jsonl docs/superpowers/specs/2026-09-26-gricean-simulation-design.md CLAUDE.md
git commit -m "chore: choose neither source for the OLMo ladder"
```

---

### Task 3: Revision pinner

**Files:**
- Create: `scripts/pin_revisions.py`
- Modify: `src/gricean/lm.py` (after `resolve_revision`, line 69)
- Modify: `Makefile` (new target `pin`)
- Test: `tests/test_lm.py`

**Interfaces:**
- Consumes: `resolve_revision(repo, revision) -> str`.
- Produces: `pin_registry(path: Path, resolver=resolve_revision) -> dict[str, str]` that rewrites every model's `revision` to the resolved sha in place, preserving the file's comment and key order, and returns `{model_name: sha}` for the entries it changed. Models whose `revision` already matches `[0-9a-f]{40}` are left alone and not returned.

- [ ] **Step 1: Write the failing test**

```python
def test_pin_registry_rewrites_symbolic_revisions_only(tmp_path: Path):
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent("""
        # keep me
        models:
          a: {repo: org/a, revision: main, family: f, template: none}
          b: {repo: org/b, revision: 0123456789abcdef0123456789abcdef01234567, family: f, template: none}
        ladders: {f: [a, b]}
        neither_source: {f: a}
    """))
    changed = lm.pin_registry(p, resolver=lambda repo, rev: "f" * 40)
    assert changed == {"a": "f" * 40}
    text = p.read_text()
    assert "# keep me" in text
    reg = lm.load_registry(p)
    assert reg["a"].revision == "f" * 40 and reg["b"].revision == "0123456789abcdef0123456789abcdef01234567"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_lm.py -k pin_registry -v`
Expected: FAIL with `AttributeError: ... 'pin_registry'`.

- [ ] **Step 3: Implement `pin_registry` and the script**

Rewrite by regex on the text, not by dumping YAML, so comments survive: for each model block find `revision: <sym>` and substitute. The script `scripts/pin_revisions.py` calls `pin_registry(Path("configs/models.yaml"))` and prints one line per changed model as `name  repo  sha  (cache|hub)`; the source label comes from checking whether `HF_HUB_CACHE/models--<repo>/refs/<rev>` existed before resolution. Makefile:

```make
pin:
	uv run python scripts/pin_revisions.py
```

- [ ] **Step 4: Run the test and the fast suite**

Run: `uv run pytest`
Expected: PASS.

- [ ] **Step 5: Update README step 3 and commit**

README line 17 becomes: "`make pin` resolves every `revision: main` in `configs/models.yaml` to the commit hash in your cache or on the Hub; commit the result."

```bash
git add src/gricean/lm.py scripts/pin_revisions.py Makefile tests/test_lm.py README.md
git commit -m "feat: make pin resolves model revisions to commit hashes"
```

Do **not** run `make pin` yet; that happens in Task 13 after the `neither` source is final.

---

### Task 4: Explicit marker for empty greedy responses

**Files:**
- Modify: `src/gricean/report.py:93-114`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `examples(resp_root, models, item_ids, expected) -> str`.
- Produces: same signature; a response that is empty after `.strip()` renders as `**{model}:** ⟨empty: model emitted end-of-sequence⟩`. Module constant `EMPTY_RESPONSE = "⟨empty: model emitted end-of-sequence⟩"`.

- [ ] **Step 1: Write the failing test**

Reuse the response-fixture helper already in `tests/test_report.py`; write one greedy `full` row for model `b` with `response: ""` and assert:

```python
def test_examples_marks_empty_greedy_response(tmp_path):
    ...
    out = report.examples(resp_root, ["b"], ["en_ir_01"], {"b": False})
    assert f"**b:** {report.EMPTY_RESPONSE}" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_report.py -k empty_greedy -v`
Expected: FAIL (`**b:** ` followed by a blank).

- [ ] **Step 3: Implement**

One line in the inner loop of `examples`: `text = r.iloc[0]["response"].strip() or EMPTY_RESPONSE`.

- [ ] **Step 4: Run, then record the decision**

Run: `uv run pytest tests/test_report.py -v` → PASS.

In the spec §5.1, after "one greedy response (printed in the paper)", add: "An empty greedy response is printed as such; it is a finding (the base model treats the prompt as complete), not a gap." In CLAUDE.md, mark open task (3) decided: greedy, with empty marker.

```bash
git add src/gricean/report.py tests/test_report.py docs/superpowers/specs/2026-09-26-gricean-simulation-design.md CLAUDE.md
git commit -m "feat: printed examples mark empty greedy responses explicitly"
```

---

### Tasks 5–10: English stimulus files (author)

One task per category. Each file is `stimuli/en/<category>.yaml` with `category`, `language: en`, and `items`. Shared rules for every item:

- `prompt` ends in `?`, `!` or `.` (Task 1 enforces this for ablated categories; keep it everywhere for `embedded` readability).
- `literal_reading` and `intended_reading` are one short clause each, naming the speech act, as in the seed file.
- `literal_continuation` and `intended_continuation` are each one sentence, 4 to 15 words, that a base model could plausibly emit verbatim; both must be **true** where they make a factual claim, so the correctness rubric (spec §5.1) stays separable from intent.
- The two continuations should share as few content words as possible: `neither.token_overlap` and the log-odds measure both get sharper the less they overlap.
- Controls (`control: true`) have `literal_continuation: null` and `intended_continuation: null` and a prompt whose literal and intended readings coincide.
- For ablated categories, at least one politeness marker from `stimuli/ablation.yaml` **or** terminal punctuation must be present so `stripped` differs from `full` (Task 1 enforces).

Verification for every one of Tasks 5–10 is the same:

- [ ] Run `uv run gricean freeze --partial` → prints a checksum, no `ValueError`.
- [ ] Run `uv run python -c "from gricean import stimuli as S; S.check_quotas(S.load_all())"` → the error message no longer lists this category under `en/`.
- [ ] Commit: `git add stimuli/en/<category>.yaml stimuli/CHECKSUMS && git commit -m "stimuli: en <category>"`.

### Task 5: `indirect_request` (ids `en_ir_01..08`, `en_ir_c01..c02`)

Keep the existing `en_ir_01` and `en_ir_c01`. Add seven dissociation items varying the modal (`can`, `could`, `would you mind`, `do you know`, `is it possible to`, `are you able to`, `I wonder if you could`) and one more control. Literal continuation pattern: `"Yes, I can."` / `"Yes, I know."`; intended: the answer itself.

### Task 6: `pattern_continuation` (`en_pc_*`)

Spec example (Ouyang et al. 2022, Fig. 1): `"Explain the moon landing to a 6-year-old."` Literal continuation is another task in the same imperative form (`"Explain the theory of gravity to a 6-year-old."`); intended is the first sentence of an actual explanation. Vary the task type (explain, list, summarize, write, describe). Controls: the same imperative preceded by an explicit frame that makes it unambiguous, e.g. `"Answer the following. Explain the moon landing to a 6-year-old."`.

### Task 7: `embedded_instruction` (`en_ei_*`)

Spec example: `"She asked: 'summarize this paragraph for me.' The paragraph was: ..."`. The prompt must contain the material the instruction operates on (a one- or two-sentence paragraph, a short list, a word to define) so the intended reading is executable. Literal continuation continues the narrative (`"He nodded and reached for his glasses."`); intended performs the embedded task. Controls: the same task stated directly, no narrative frame.

### Task 8: `malformed_instruction` (`en_mi_*`)

Spec example: `"tel me hw many days febuary has"`. Note no terminal punctuation is fine here (not ablated), but the Task 1 punctuation rule is not applied to this category. Literal continuation: more garbled text or a comment on the typos (`"i dont no wat u mean"`); intended: the repaired answer (`"February has 28 days, or 29 in a leap year."`). Vary the damage: typos, dropped words, no spaces, wrong word order, mixed case. Controls: the same instruction, well-formed.

### Task 9: `underspecified_request` (`en_ur_*`)

Spec example: `"Translate 'obrigado'."` The intended reading is the *most plausible* completion (into English, given the prompt language), so `intended_continuation` must state the plausible choice and `literal_continuation` must be the clarifying question or the refusal to choose (`"Into which language?"`). Vary the missing slot: target language, unit, audience, format, date range. Controls: the same request with the slot filled.

### Task 10: `rhetorical_question` (`en_rq_*`)

Spec example: `"Who would ever want to wait three hours for a bus?"` Literal continuation answers the question as information-seeking (`"Some commuters do, when there is no alternative."`); intended acknowledges the point (`"Nobody; three hours is far too long."`). Vary the form: `who would`, `isn't it`, `how many times`, `what kind of`, `why bother`. Controls: genuine information-seeking questions of similar shape (`"How long is the wait for the number 12 bus?"`).

---

### Task 11: Portuguese mirror (author)

**Files:**
- Create: `stimuli/pt/indirect_request.yaml`, `stimuli/pt/pattern_continuation.yaml`, `stimuli/pt/embedded_instruction.yaml`, `stimuli/pt/malformed_instruction.yaml`, `stimuli/pt/underspecified_request.yaml`, `stimuli/pt/rhetorical_question.yaml`

Each file: `language: pt`, 2 dissociation items (`pt_<abbr>_01..02`) and 1 control (`pt_<abbr>_c01`). Written natively in Brazilian Portuguese; **not** translations of specific English items, though they instantiate the same contrast. Politeness markers available for stripping are only `por favor`, `obrigado`, `obrigada` (see `ablation.yaml`); add `poderia`-style softeners if needed by editing `ablation.yaml` **before** the final freeze, and note the change in the commit.

- [ ] Write the six files.
- [ ] Run `uv run gricean freeze --partial` → no `ValueError`.
- [ ] Run the quota check from Tasks 5–10 → the message no longer lists any `pt/` cell.
- [ ] Commit: `git add stimuli/pt stimuli/CHECKSUMS stimuli/ablation.yaml && git commit -m "stimuli: pt mirror (12 + 6)"`.

---

### Task 12: Final freeze, `neither` sampling, approval pass

**Files:**
- Modify: `stimuli/CHECKSUMS`, `stimuli/neither/olmo.jsonl`, `stimuli/neither/qwen.jsonl`, `stimuli/neither/smollm.jsonl`

- [ ] **Step 1: Strict freeze**

Run: `make freeze`
Expected: prints the combined checksum; no `ValueError`. If it raises, a quota or ablation rule is unmet: fix the stimulus, not the check.

- [ ] **Step 2: Sample `neither` for all three families (~15 min, downloads Pythia or Qwen base and OLMo base if absent)**

Run: `make neither`
Expected: three `neither[<family>] from <source>: 78 rows` lines.

- [ ] **Step 3: Approval pass (author)**

List flagged rows:

```bash
for f in stimuli/neither/*.jsonl; do echo "== $f"; python -c "import json,sys;[print(r['item_id'], repr(r['chosen'])) for r in map(json.loads, open(sys.argv[1])) if r['flag']]" "$f"; done
```

For each flagged row, either edit `chosen` to one of the other two `samples` that does not answer the question, or set `"approved": true` if the flag is a false positive (overlap on function words). Also skim **unflagged** rows for samples that answer the question without lexical overlap; `containment-v2` cannot see paraphrase. Edit `chosen` for those too.

- [ ] **Step 4: Verify nothing is blocked and commit**

```bash
uv run python -c "from gricean import neither; [neither.load_neither(f) for f in ('olmo','qwen','smollm')]; print('ok')"
git add stimuli/CHECKSUMS stimuli/neither
git commit -m "stimuli: freeze 78-item set; neither samples with author pass"
```

Expected: `ok`. If it raises, a flagged row still lacks `approved: true`.

---

### Task 13: Pin, document, merge, hand off to the run sequence

- [ ] **Step 1: Pin revisions**

Run: `make pin`
Expected: one line per model that was still `main`, each tagged `cache` or `hub`. Commit `configs/models.yaml`.

- [ ] **Step 2: Update CLAUDE.md "Current status"**

Replace the three open tasks and the "Then:" line with: date, "stimuli frozen (78 items, checksum `<first 8 chars>`), `neither` approved, revisions pinned; next `make generate`". Keep the deferred-minors list.

- [ ] **Step 3: Run the full fast suite and marimo check**

Run: `uv run pytest && uv run marimo check notebooks/`
Expected: all pass, no output from marimo.

- [ ] **Step 4: Commit, push, merge PR #1**

```bash
git add CLAUDE.md configs/models.yaml
git commit -m "chore: pin revisions; status before the real run"
git push
gh pr merge 1 --squash --delete-branch=false
```

- [ ] **Step 5: Start the run (author, per README steps 6–10)**

`make generate` → `make logprobs` → `make sheet` → code `data/coding/coder1.csv` → `make finetune` → rerun 6–7 for `smollm` → `make report`. Budget 1–2 h for generate+logprobs and ~3 h for the four adapters on the M3.
