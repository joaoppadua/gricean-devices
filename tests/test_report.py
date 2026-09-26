from pathlib import Path
import pandas as pd
from gricean import report as R
from gricean.io import write_jsonl

MODELS = ["m_base", "m_sft"]


def coded():
    rows = []
    for model, intents in (("m_base", ["neither", "literal", "literal", "intended"]),
                           ("m_sft", ["intended"] * 4)):
        for i, it in enumerate(intents):
            rows.append({"row_id": f"{model}|1|a{i}|full|greedy|", "item_id": f"a{i}",
                         "category": "indirect_request", "language": "en", "control": False,
                         "version": "full", "mode": "greedy", "seed": "", "prompt_text": "p",
                         "response": "r", "model": model, "template": True, "coder": "c1",
                         "intent": it, "correctness": "na"})
    return pd.DataFrame(rows)


def lp_rows(root: Path):
    for model, lo in (("m_base", [-1.0, -0.5]), ("m_sft", [2.0, 3.0])):
        write_jsonl(root / f"{model}__tmpl.jsonl", [
            {"item_id": f"a{i}", "category": "indirect_request", "language": "en", "version": "full",
             "model": model, "template": True, "mode": "logprob", "seed": None, "log_odds": v,
             "share_neither": 0.5, "share_literal": 0.25, "share_intended": 0.25, "stamp": "s",
             "stimuli_checksum": "c"} for i, v in enumerate(lo)])


def test_intent_table_rates():
    t = R.intent_table(coded(), MODELS)
    assert t.loc[("indirect_request", "full"), ("m_base", "intended")] == 0.25
    assert t.loc[("indirect_request", "full"), ("m_sft", "intended")] == 1.0
    assert t.loc[("indirect_request", "full"), ("m_base", "neither")] == 0.25


def test_logodds_table_means_and_ci(tmp_path: Path):
    lp_rows(tmp_path)
    t = R.logodds_table(tmp_path, MODELS)
    assert t.loc[("indirect_request", "full"), ("m_sft", "mean")] == 2.5
    assert t.loc[("indirect_request", "full"), ("m_sft", "ci_low")] <= 2.5 <= t.loc[("indirect_request", "full"), ("m_sft", "ci_high")]


def test_examples_markdown(tmp_path: Path):
    resp = tmp_path / "resp"
    for model in MODELS:
        write_jsonl(resp / f"{model}__tmpl.jsonl", [{"item_id": "a0", "category": "indirect_request",
            "language": "en", "control": False, "version": "full", "model": model, "template": True,
            "mode": "greedy", "seed": None, "prompt_text": "Can you tell me the time?",
            "response": f"answer from {model}", "stamp": "s", "stimuli_checksum": "c"}])
    md = R.examples(resp, MODELS, ["a0"])
    assert "answer from m_base" in md and "answer from m_sft" in md and "Can you tell me the time?" in md


def test_build_writes_all_outputs(tmp_path: Path):
    coded().to_csv(tmp_path / "coded.csv", index=False)
    lp_rows(tmp_path / "lp")
    resp = tmp_path / "resp"
    for model in MODELS:
        write_jsonl(resp / f"{model}__tmpl.jsonl", [{"item_id": "a0", "category": "indirect_request",
            "language": "en", "control": False, "version": "full", "model": model, "template": True,
            "mode": "greedy", "seed": None, "prompt_text": "p", "response": "r", "stamp": "s", "stimuli_checksum": "c"}])
    summary = R.build(tmp_path / "coded.csv", None, None, resp, tmp_path / "lp", MODELS, tmp_path / "paper")
    for name in ("intent.csv", "intent.tex", "logodds.csv", "logodds.tex", "ladder.png", "examples.md"):
        assert (tmp_path / "paper" / name).exists()
    assert summary["n_coded"] == 8 and summary["kappa"] is None
