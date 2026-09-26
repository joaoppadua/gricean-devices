from pathlib import Path
import math
import pandas as pd
import pytest
from gricean import coding as C
from gricean.io import write_jsonl


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
    n = C.export_sheet(tmp_path / "resp", tmp_path / "coder1.csv", coder="coder1", sample_frac=None, seed=0)
    df = pd.read_csv(tmp_path / "coder1.csv")
    assert n == 8 and len(df) == 8
    assert df["intent"].isna().all() and "model" in df.columns


def test_export_blind_sample_hides_model(tmp_path: Path):
    make_responses(tmp_path / "resp")
    n = C.export_sheet(tmp_path / "resp", tmp_path / "coder2.csv", coder="coder2", sample_frac=0.5, seed=0)
    df = pd.read_csv(tmp_path / "coder2.csv")
    key = pd.read_csv(tmp_path / "coder2.key.csv")
    assert n == 4 and "model" not in df.columns and set(key.columns) >= {"row_id", "model", "template", "prompt_text"}
    assert "prompt_text" not in df.columns and (df["stimulus_text"] == "p").all()


def test_load_rejects_blank_or_invalid(tmp_path: Path):
    make_responses(tmp_path / "resp")
    C.export_sheet(tmp_path / "resp", tmp_path / "c.csv", coder="c", sample_frac=None, seed=0)
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
    C.export_sheet(tmp_path / "resp", tmp_path / "c2.csv", coder="c2", sample_frac=0.5, seed=0)
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
