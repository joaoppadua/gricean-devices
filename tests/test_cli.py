from pathlib import Path
import shutil
import yaml
from gricean import cli
from gricean.lm import FakeLM


def project(tmp_path: Path) -> Path:
    shutil.copytree("stimuli", tmp_path / "stimuli", ignore=shutil.ignore_patterns("neither", "CHECKSUMS"))
    for f in (tmp_path / "stimuli").glob("*/*.yaml"):          # drop scaffolded TODO items
        doc = yaml.safe_load(f.read_text())
        doc["items"] = [it for it in doc["items"] if not any("TODO" in str(v) for v in it.values())]
        yaml.safe_dump(doc, f.open("w"), allow_unicode=True, sort_keys=False)
    (tmp_path / "configs").mkdir()
    shutil.copy("configs/generation.yaml", tmp_path / "configs/generation.yaml")
    yaml.safe_dump({
        "models": {"b": {"repo": "x", "revision": "1", "family": "t", "template": "none"},
                   "s": {"repo": "y", "revision": "1", "family": "t", "template": "chat"},
                   "q": {"repo": "z", "revision": "1", "family": "u", "template": "none"},
                   "a": {"repo": "x", "revision": "1", "family": "t", "template": "alpaca", "raw_control": False}},
        "ladders": {"t": ["b", "s", "a"], "u": ["q"]},
        "neither_source": {"t": "q", "u": "b"},
    }, (tmp_path / "configs/models.yaml").open("w"))
    return tmp_path


def fake_loader(spec):
    return FakeLM(spec.name, logprob_rules={})


def test_full_pipeline_with_fakes(tmp_path: Path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    assert cli.main(["freeze", "--partial"]) == 0
    assert (root / "stimuli/CHECKSUMS").exists()
    assert cli.main(["neither", "--family", "t", "--family", "u"], loader=fake_loader) == 0
    assert (root / "stimuli/neither/t.jsonl").exists()
    assert (root / "stimuli/neither/u.jsonl").exists()
    assert cli.main(["generate", "--all"], loader=fake_loader) == 0
    assert (root / "data/responses/b__raw.jsonl").exists()
    assert (root / "data/responses/s__tmpl.jsonl").exists()
    assert (root / "data/responses/s__raw.jsonl").exists()     # tuned models also run raw
    assert not (root / "data/responses/b__tmpl.jsonl").exists()
    assert cli.main(["logprobs", "--all"], loader=fake_loader) == 0
    assert (root / "data/logprobs/s__tmpl.jsonl").exists()
    assert cli.main(["sheet"]) == 0
    assert (root / "data/coding/coder1.csv").exists()
    assert cli.main(["sheet", "--coder2", "--frac", "0.5"]) == 0
    assert (root / "data/coding/coder2.key.csv").exists()
    import pandas as pd
    sheet = pd.read_csv(root / "data/coding/coder1.csv")
    sheet["intent"] = "intended"; sheet["correctness"] = "na"
    sheet.to_csv(root / "data/coding/coder1.csv", index=False)
    assert cli.main(["report", "--ladder", "t"]) == 0
    assert (root / "paper/t/intent.csv").exists() and (root / "paper/t/examples.md").exists()


def test_generate_refuses_stale_stimuli(tmp_path: Path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    cli.main(["freeze", "--partial"])
    f = root / "stimuli/en/indirect_request.yaml"
    f.write_text(f.read_text().replace("Brazil", "Chile"))
    import pytest
    with pytest.raises(ValueError, match="freeze"):
        cli.main(["generate", "--model", "b"], loader=fake_loader)


def test_unknown_stage_returns_2(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(project(tmp_path))
    assert cli.main(["nope"]) == 2


def test_freeze_without_partial_fails_on_incomplete_set(tmp_path: Path, monkeypatch):
    import pytest
    root = project(tmp_path)
    f = root / "stimuli/en/indirect_request.yaml"
    doc = yaml.safe_load(f.read_text())
    doc["items"] = doc["items"][:1]            # below quota, whatever the real set holds
    yaml.safe_dump(doc, f.open("w"), allow_unicode=True, sort_keys=False)
    monkeypatch.chdir(root)
    with pytest.raises(ValueError, match="dissociation"):
        cli.main(["freeze"])


def test_all_skips_registry_models_outside_every_ladder(tmp_path: Path, monkeypatch):
    root = project(tmp_path)
    cfg = yaml.safe_load((root / "configs/models.yaml").read_text())
    cfg["models"]["n"] = {"repo": "w", "revision": "1", "family": "v", "template": "none"}   # neither-only source
    yaml.safe_dump(cfg, (root / "configs/models.yaml").open("w"))
    monkeypatch.chdir(root)
    cli.main(["freeze", "--partial"])
    assert cli.main(["generate", "--all"], loader=fake_loader) == 0
    assert not (root / "data/responses/n__raw.jsonl").exists()
    assert (root / "data/responses/q__raw.jsonl").exists()


def test_all_skips_raw_pass_for_raw_control_false(tmp_path: Path, monkeypatch):
    root = project(tmp_path); monkeypatch.chdir(root)
    cli.main(["freeze", "--partial"])
    assert cli.main(["neither", "--family", "t", "--family", "u"], loader=fake_loader) == 0
    assert cli.main(["generate", "--all"], loader=fake_loader) == 0
    assert (root / "data/responses/a__tmpl.jsonl").exists()
    assert not (root / "data/responses/a__raw.jsonl").exists()
    assert (root / "data/responses/s__raw.jsonl").exists()        # official-ladder tuned model still runs raw
    assert cli.main(["logprobs", "--all"], loader=fake_loader) == 0
    assert (root / "data/logprobs/a__tmpl.jsonl").exists()        # neither found via family


def test_finetune_writes_adapters_under_prefix(tmp_path: Path, monkeypatch):
    from gricean import finetune as F
    root = project(tmp_path); monkeypatch.chdir(root)
    shutil.copy(Path(__file__).parents[1] / "configs/finetune.yaml", root / "configs/finetune.yaml")
    cfg = yaml.safe_load((root / "configs/finetune.yaml").read_text())
    reg = yaml.safe_load((root / "configs/models.yaml").read_text())
    reg["models"][cfg["base"]] = {"repo": "x", "revision": "1", "family": "t", "template": "none"}
    yaml.safe_dump(reg, (root / "configs/models.yaml").open("w"))
    monkeypatch.setattr(F, "load_dolly", lambda: [
        {"category": "open_qa", "instruction": f"q{i}", "response": f"a{i}", "context": ""} for i in range(600)])
    monkeypatch.setattr(F, "train_lora", lambda base, rows, out, cfg: Path(out).mkdir(parents=True, exist_ok=True))
    assert cli.main(["finetune", "--set", "n500"]) == 0
    assert (root / "adapters/olmo_n500/train.jsonl").exists()
    assert cfg["base"] == "olmo_stage1"


def test_all_skips_adapter_models_whose_adapter_is_not_trained_yet(tmp_path: Path, monkeypatch, capsys):
    root = project(tmp_path); monkeypatch.chdir(root)
    cfg = yaml.safe_load((root / "configs/models.yaml").read_text())
    cfg["models"]["ad"] = {"repo": "x", "revision": "1", "family": "t", "template": "alpaca",
                          "raw_control": False, "adapter": "adapters/ad"}
    cfg["ladders"]["t"].append("ad")
    yaml.safe_dump(cfg, (root / "configs/models.yaml").open("w"))
    cli.main(["freeze", "--partial"])
    assert cli.main(["generate", "--all"], loader=fake_loader) == 0
    assert not (root / "data/responses/ad__tmpl.jsonl").exists()
    assert "ad" in capsys.readouterr().out            # a printed notice names the skipped model
    (root / "adapters/ad").mkdir(parents=True)
    (root / "adapters/ad/adapter_model.safetensors").write_bytes(b"")
    assert cli.main(["generate", "--all"], loader=fake_loader) == 0
    assert (root / "data/responses/ad__tmpl.jsonl").exists()


def test_explicit_raw_refused_for_raw_control_false(tmp_path: Path, monkeypatch):
    import pytest
    root = project(tmp_path); monkeypatch.chdir(root)
    cli.main(["freeze", "--partial"])
    with pytest.raises(ValueError, match="raw_control"):
        cli.main(["generate", "--model", "a", "--raw"], loader=fake_loader)


def test_stages_refuse_placeholder_items_even_after_partial_freeze(tmp_path: Path, monkeypatch):
    import pytest
    root = project(tmp_path); monkeypatch.chdir(root)
    f = root / "stimuli/en/rhetorical_question.yaml"
    doc = yaml.safe_load(f.read_text())
    doc["items"].append({"id": "en_rq_99", "control": False, "prompt": "TODO", "literal_reading": "TODO",
                         "intended_reading": "TODO", "literal_continuation": "TODO", "intended_continuation": "TODO"})
    yaml.safe_dump(doc, f.open("w"), allow_unicode=True, sort_keys=False)
    assert cli.main(["freeze", "--partial"]) == 0
    for stage in (["neither", "--family", "t"], ["generate", "--model", "b"], ["logprobs", "--model", "b"]):
        with pytest.raises(ValueError, match="en_rq_99"):
            cli.main(stage, loader=fake_loader)
