from pathlib import Path
import math
import pandas as pd
import pytest
from gricean import coding as C
from gricean.io import write_jsonl


EXP = {"m_base": True, "m_sft": True, "zz": False}   # model -> is its main condition the templated one


def make_responses(root: Path):
    rows = []
    for model in ("m_base", "m_sft"):
        for i in range(4):
            rows.append({"item_id": f"a{i}", "category": "indirect_request", "language": "en",
                         "control": False, "version": "full", "model": model, "template": True,
                         "mode": "greedy", "seed": None, "prompt_text": f"<tmpl>{model} p", "stimulus_text": "p",
                         "response": f"r{i}{model}",
                         "stamp": "s", "stimuli_checksum": "c"})
        write_jsonl(root / f"{model}__tmpl.jsonl", rows[-4:])


def test_export_full_sheet_has_all_rows_and_blank_rubric(tmp_path: Path):
    make_responses(tmp_path / "resp")
    n = C.export_sheet(tmp_path / "resp", tmp_path / "coder1.csv", coder="coder1", sample_frac=None, seed=0, expected=EXP)
    df = pd.read_csv(tmp_path / "coder1.csv")
    assert n == 8 and len(df) == 8
    assert df["intent"].isna().all() and "model" in df.columns


def test_export_blind_sample_hides_model(tmp_path: Path):
    make_responses(tmp_path / "resp")
    n = C.export_sheet(tmp_path / "resp", tmp_path / "coder2.csv", coder="coder2", sample_frac=0.5, seed=0, expected=EXP)
    df = pd.read_csv(tmp_path / "coder2.csv")
    key = pd.read_csv(tmp_path / "coder2.key.csv")
    assert n == 4 and "model" not in df.columns and set(key.columns) >= {"row_id", "model", "template", "prompt_text"}
    assert "prompt_text" not in df.columns and (df["stimulus_text"] == "p").all()


def test_load_rejects_blank_or_invalid(tmp_path: Path):
    make_responses(tmp_path / "resp")
    C.export_sheet(tmp_path / "resp", tmp_path / "c.csv", coder="c", sample_frac=None, seed=0, expected=EXP)
    df = pd.read_csv(tmp_path / "c.csv")
    df["intent"] = "intended"; df["correctness"] = "correct"
    df.loc[0, "intent"] = "intendd"
    df.to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="row_id"):
        C.load_sheet(tmp_path / "c.csv")
    df.loc[0, "intent"] = None
    df.to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="blank"):
        C.load_sheet(tmp_path / "c.csv")


def test_load_merges_key_back(tmp_path: Path):
    make_responses(tmp_path / "resp")
    C.export_sheet(tmp_path / "resp", tmp_path / "c2.csv", coder="c2", sample_frac=0.5, seed=0, expected=EXP)
    df = pd.read_csv(tmp_path / "c2.csv")
    df["intent"] = "neither"; df["correctness"] = "na"
    df.to_csv(tmp_path / "c2.csv", index=False)
    out = C.load_sheet(tmp_path / "c2.csv", key=tmp_path / "c2.key.csv")
    assert "model" in out.columns and len(out) == 4


def test_cohen_kappa_known_value():
    a = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["n"] * 10 + ["y"] * 15
    # po = 25/50 = 0.5 ; pa(y)=30/50, pb(y)=35/50 ; pe = 0.6*0.7 + 0.4*0.3 = 0.54
    assert math.isclose(C.cohen_kappa(a, b), (0.5 - 0.54) / (1 - 0.54))
    assert C.cohen_kappa(["a", "b"], ["a", "b"]) == 1.0


def test_export_preserves_existing_codes_and_appends_new_rows(tmp_path: Path):
    make_responses(tmp_path / "resp")
    out = tmp_path / "coder1.csv"
    C.export_sheet(tmp_path / "resp", out, coder="coder1", sample_frac=None, seed=0, expected=EXP)
    sheet = pd.read_csv(out, dtype=str, keep_default_na=False)
    first = sheet.loc[0, "row_id"]
    sheet.loc[0, ["intent", "correctness"]] = ["intended", "correct"]
    sheet.to_csv(out, index=False)
    write_jsonl(tmp_path / "resp" / "zz__raw.jsonl", [{"item_id": "new1", "category": "indirect_request",
        "language": "en", "control": False, "version": "full", "model": "zz", "template": False,
        "mode": "greedy", "seed": None, "stimulus_text": "p", "prompt_text": "p", "response": "r",
        "stamp": "s", "stimuli_checksum": "c"}])
    n = C.export_sheet(tmp_path / "resp", out, coder="coder1", sample_frac=None, seed=0, expected=EXP)
    again = pd.read_csv(out, dtype=str, keep_default_na=False).set_index("row_id")
    assert n == len(again) == len(sheet) + 1
    assert again.loc[first, "intent"] == "intended" and again.loc[first, "correctness"] == "correct"
    new_id = next(i for i in again.index if i.startswith("zz|0|new1|"))
    assert again.loc[new_id, "intent"] == "" and again.loc[new_id, "coder"] == "coder1"


def _resp(model, template, item, *, version="full", mode="greedy", seed=None, response="r"):
    return {"item_id": item, "category": "indirect_request", "language": "en", "control": False,
            "version": version, "model": model, "template": template, "mode": mode, "seed": seed,
            "stimulus_text": "p", "prompt_text": "p", "response": response, "stamp": "s", "stimuli_checksum": "c"}


DESIGN = {"b": False, "t": True}


def make_design(root: Path):
    """A base model and a tuned model with every kind of row a real run writes: greedy and sampled,
    the tuned model's raw template control, its untemplated `embedded` rows, and empty responses."""
    write_jsonl(root / "b__raw.jsonl", [
        _resp("b", False, "a0"), _resp("b", False, "a0", mode="sample", seed=1),
        _resp("b", False, "a1", response=""), _resp("b", False, "a2", response=" \n\n")])
    write_jsonl(root / "t__tmpl.jsonl", [
        _resp("t", True, "a0"), _resp("t", True, "a0", mode="sample", seed=1),
        _resp("t", False, "a0", version="embedded"), _resp("t", False, "a0", version="embedded", mode="sample", seed=1)])
    write_jsonl(root / "t__raw.jsonl", [
        _resp("t", False, "a0"), _resp("t", False, "a0", mode="sample", seed=1)])


def _cells(df):
    return {(r.model, str(r.template).lower() == "true", r.item_id, r.version, r.mode) for r in df.itertuples()}


def test_sheet_holds_the_greedy_response_of_each_models_main_condition(tmp_path: Path):
    make_design(tmp_path / "resp")
    n = C.export_sheet(tmp_path / "resp", tmp_path / "coder1.csv", coder="coder1", sample_frac=None, seed=0,
                       expected=DESIGN)
    df = pd.read_csv(tmp_path / "coder1.csv", dtype=str, keep_default_na=False)
    assert n == 5
    assert _cells(df) == {("b", False, "a0", "full", "greedy"), ("b", False, "a1", "full", "greedy"),
                          ("b", False, "a2", "full", "greedy"), ("t", True, "a0", "full", "greedy"),
                          ("t", False, "a0", "embedded", "greedy")}     # no samples, no raw control of `t`


def test_empty_responses_are_coded_by_rule_and_the_rest_left_blank(tmp_path: Path):
    make_design(tmp_path / "resp")
    C.export_sheet(tmp_path / "resp", tmp_path / "coder1.csv", coder="coder1", sample_frac=None, seed=0,
                   expected=DESIGN)
    df = pd.read_csv(tmp_path / "coder1.csv", dtype=str, keep_default_na=False).set_index("item_id")
    for item in ("a1", "a2"):                       # empty, and whitespace only
        assert df.loc[item, "intent"] == "neither" and df.loc[item, "correctness"] == "na"
    assert (df.loc["a0", "intent"] == "").all() and (df.loc["a0", "correctness"] == "").all()


def test_blind_sample_is_drawn_from_the_hand_coded_rows(tmp_path: Path):
    make_design(tmp_path / "resp")
    n = C.export_sheet(tmp_path / "resp", tmp_path / "coder2.csv", coder="coder2", sample_frac=1.0, seed=0,
                       expected=DESIGN)
    df = pd.read_csv(tmp_path / "coder2.csv", dtype=str, keep_default_na=False)
    key = pd.read_csv(tmp_path / "coder2.key.csv", dtype=str, keep_default_na=False)
    assert n == 3 and set(df["item_id"]) == {"a0"}          # rule-coded empties are not sent to coder 2
    assert (df["response"].str.strip() != "").all() and (df["intent"] == "").all()
    assert _cells(df.merge(key, on="row_id")) == {("b", False, "a0", "full", "greedy"),
                                                  ("t", True, "a0", "full", "greedy"),
                                                  ("t", False, "a0", "embedded", "greedy")}


READINGS = {f"a{i}": {"literal_reading": f"lit {i}", "intended_reading": f"int {i}",
                      "literal_continuation": f"lc {i}", "intended_continuation": None} for i in range(4)}


def _blind(tmp_path: Path, frac: float, **kw):
    """The full sheet and a blind sample of it, as data frames, with the key of the sample."""
    make_responses(tmp_path / "resp")
    C.export_sheet(tmp_path / "resp", tmp_path / "coder1.csv", coder="coder1", sample_frac=None, seed=0, expected=EXP)
    C.export_sheet(tmp_path / "resp", tmp_path / "coder2.csv", coder="coder2", sample_frac=frac, seed=0,
                   expected=EXP, **kw)
    return tuple(pd.read_csv(tmp_path / name, dtype=str, keep_default_na=False)
                 for name in ("coder1.csv", "coder2.csv", "coder2.key.csv"))


def test_blind_sample_row_ids_do_not_name_the_model(tmp_path: Path):
    full, blind, key = _blind(tmp_path, 1.0)
    assert blind["row_id"].is_unique
    assert not blind["row_id"].str.contains(r"m_base|m_sft|\|").any()       # the full sheet's ids begin with the model
    assert set(key["row_id"]) == set(blind["row_id"]) and set(key["source_row_id"]) == set(full["row_id"])


def test_blind_sample_order_does_not_follow_the_model(tmp_path: Path):
    _, blind, key = _blind(tmp_path, 1.0)
    order = list(zip(blind["item_id"], blind["version"], blind["row_id"]))
    assert order == sorted(order)                                           # by item and version, then by the opaque id
    models = blind.merge(key, on="row_id", how="left")["model"].tolist()
    assert models != sorted(models)                                         # not one block per model


def test_load_with_key_restores_the_row_ids_of_the_full_sheet(tmp_path: Path):
    full, blind, _ = _blind(tmp_path, 0.5)
    blind["intent"] = "neither"; blind["correctness"] = "incorrect"
    blind.to_csv(tmp_path / "coder2.csv", index=False)
    out = C.load_sheet(tmp_path / "coder2.csv", key=tmp_path / "coder2.key.csv")
    assert len(out) == 4 and set(out["row_id"]) <= set(full["row_id"]) and "model" in out.columns


def test_blind_sample_carries_the_readings_of_each_item(tmp_path: Path):
    _, blind, _ = _blind(tmp_path, 1.0, readings=READINGS)
    n = blind["item_id"].str[1:]
    assert (blind["literal_reading"] == "lit " + n).all() and (blind["intended_reading"] == "int " + n).all()
    assert (blind["literal_continuation"] == "lc " + n).all()
    assert (blind["intended_continuation"] == "").all()                     # a control has no continuation
    assert (blind["intent"] == "").all()


def test_blind_sample_refuses_an_item_without_readings(tmp_path: Path):
    with pytest.raises(ValueError, match="a3"):
        _blind(tmp_path, 1.0, readings={k: v for k, v in READINGS.items() if k != "a3"})


def _coded_blind(tmp_path: Path) -> pd.DataFrame:
    """A blind sample with both codes entered, as coder 2 would return it."""
    _, blind, _ = _blind(tmp_path, 1.0)
    return blind.assign(intent="neither", correctness="incorrect")


def test_load_refuses_a_blind_sheet_with_an_id_its_key_does_not_hold(tmp_path: Path):
    """A spreadsheet rewrites ids that look like numbers; such a row must not drop out unseen."""
    blind = _coded_blind(tmp_path)
    blind.loc[0, "row_id"] = "4.10363E+11"
    blind.to_csv(tmp_path / "coder2.csv", index=False)
    with pytest.raises(ValueError, match=r"not in the key.*4\.10363E\+11"):
        C.load_sheet(tmp_path / "coder2.csv", key=tmp_path / "coder2.key.csv")


def test_load_refuses_a_blind_sheet_that_lacks_a_row_of_its_key(tmp_path: Path):
    blind = _coded_blind(tmp_path)
    blind.iloc[1:].to_csv(tmp_path / "coder2.csv", index=False)
    with pytest.raises(ValueError, match="missing from the sheet"):
        C.load_sheet(tmp_path / "coder2.csv", key=tmp_path / "coder2.key.csv")


def test_load_refuses_a_blind_sheet_with_a_repeated_row(tmp_path: Path):
    blind = _coded_blind(tmp_path)
    pd.concat([blind, blind.iloc[:1]]).to_csv(tmp_path / "coder2.csv", index=False)
    with pytest.raises(ValueError, match="more than once"):
        C.load_sheet(tmp_path / "coder2.csv", key=tmp_path / "coder2.key.csv")


# --- coder 2 as an agent: rows out as text, codes back into the sheet --------------------------------

def _batches(tmp_path: Path, n: int = 2):
    """The blind sheet of the fixture written as `n` batches; returns the sheet and the batch folder."""
    _, blind, _ = _blind(tmp_path, 1.0, readings=READINGS)
    C.write_batches(tmp_path / "coder2.csv", tmp_path / "b", n)
    return blind, tmp_path / "b"


def _ids_in(text: str) -> list[str]:
    return [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("row_id: ")]


def _fill(folder: Path, intent: str = "intended", correctness: str = "correct") -> None:
    for path in sorted(folder.glob("codes_*.csv")):
        pd.read_csv(path, dtype=str).assign(intent=intent, correctness=correctness).to_csv(path, index=False)


def test_batches_cover_the_sheet_once_and_keep_the_rows_of_an_item_together(tmp_path: Path):
    blind, folder = _batches(tmp_path)
    texts = [p.read_text() for p in sorted(folder.glob("batch_*.md"))]
    ids = [_ids_in(t) for t in texts]
    assert len(texts) == 2 and sorted(sum(ids, [])) == sorted(blind["row_id"])
    item_of = dict(zip(blind["row_id"], blind["item_id"]))
    items = [{item_of[i] for i in batch} for batch in ids]
    assert not items[0] & items[1]          # "same text, same codes" needs every row of an item in one place


def test_a_batch_shows_what_the_sheet_shows_and_nothing_of_the_key(tmp_path: Path):
    blind, folder = _batches(tmp_path)
    text = "".join(p.read_text() for p in sorted(folder.glob("batch_*.md")))
    row = blind.iloc[0]
    for field in ("item_id", "category", "version", "stimulus_text", "response", "literal_reading", "intended_reading"):
        assert row[field] in text
    assert "<tmpl>" not in text and "prompt_text" not in text and "model:" not in text   # the key's columns


def test_each_batch_comes_with_a_blank_form_for_its_codes(tmp_path: Path):
    _, folder = _batches(tmp_path)
    for batch in sorted(folder.glob("batch_*.md")):
        form = pd.read_csv(folder / batch.name.replace("batch_", "codes_").replace(".md", ".csv"),
                           dtype=str, keep_default_na=False)
        assert list(form.columns) == ["row_id", "intent", "correctness"]
        assert form["row_id"].tolist() == _ids_in(batch.read_text())
        assert (form[["intent", "correctness"]] == "").all().all()


def test_enter_codes_changes_the_two_code_columns_and_nothing_else(tmp_path: Path):
    blind, folder = _batches(tmp_path)
    _fill(folder, "Literal ", "incorrect")
    assert C.enter_codes(tmp_path / "coder2.csv", folder) == len(blind)
    after = pd.read_csv(tmp_path / "coder2.csv", dtype=str, keep_default_na=False)
    rest = [c for c in blind.columns if c not in ("intent", "correctness")]
    assert after[rest].equals(blind[rest])
    assert set(after["intent"]) == {"literal"} and set(after["correctness"]) == {"incorrect"}
    assert len(C.load_sheet(tmp_path / "coder2.csv", key=tmp_path / "coder2.key.csv")) == len(blind)


@pytest.mark.parametrize("damage, message", [
    (lambda f: f.iloc[1:], "missing"),
    (lambda f: f.assign(row_id=["zzz"] + f["row_id"].tolist()[1:]), "not in the sheet"),
    (lambda f: pd.concat([f, f.iloc[:1]]), "more than once"),
    (lambda f: f.assign(intent=["intendd"] + f["intent"].tolist()[1:]), "invalid intent"),
    (lambda f: f.assign(correctness=[""] + f["correctness"].tolist()[1:]), "invalid correctness"),
])
def test_enter_codes_refuses_a_form_that_does_not_fit_and_leaves_the_sheet_alone(tmp_path: Path, damage, message):
    blind, folder = _batches(tmp_path)
    _fill(folder)
    form = folder / "codes_01.csv"
    damage(pd.read_csv(form, dtype=str, keep_default_na=False)).to_csv(form, index=False)
    with pytest.raises(ValueError, match=message):
        C.enter_codes(tmp_path / "coder2.csv", folder)
    assert pd.read_csv(tmp_path / "coder2.csv", dtype=str, keep_default_na=False).equals(blind)


def test_batches_are_not_written_over_forms_that_may_hold_codes(tmp_path: Path):
    _, folder = _batches(tmp_path)
    _fill(folder)
    with pytest.raises(ValueError, match="already holds forms"):
        C.write_batches(tmp_path / "coder2.csv", folder, 2)
    assert set(pd.read_csv(folder / "codes_01.csv", dtype=str)["intent"]) == {"intended"}
