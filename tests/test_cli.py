from pathlib import Path
import shutil
import yaml
from gricean import cli
from gricean.lm import FakeLM


def project(tmp_path: Path) -> Path:
    shutil.copytree("stimuli", tmp_path / "stimuli", ignore=shutil.ignore_patterns("neither", "CHECKSUMS"))
    (tmp_path / "configs").mkdir()
    shutil.copy("configs/generation.yaml", tmp_path / "configs/generation.yaml")
    yaml.safe_dump({
        "models": {"b": {"repo": "x", "revision": "1", "family": "t", "template": "none"},
                   "s": {"repo": "y", "revision": "1", "family": "t", "template": "chat"},
                   "q": {"repo": "z", "revision": "1", "family": "u", "template": "none"}},
        "ladders": {"t": ["b", "s"], "u": ["q"]},
        "neither_source": {"t": "q", "u": "b"},
    }, (tmp_path / "configs/models.yaml").open("w"))
    return tmp_path


def fake_loader(spec):
    return FakeLM(spec.name, logprob_rules={})


def test_full_pipeline_with_fakes(tmp_path: Path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    assert cli.main(["freeze"]) == 0
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
    cli.main(["freeze"])
    f = root / "stimuli/en/indirect_request.yaml"
    f.write_text(f.read_text().replace("Brazil", "Chile"))
    import pytest
    with pytest.raises(ValueError, match="freeze"):
        cli.main(["generate", "--model", "b"], loader=fake_loader)


def test_unknown_stage_returns_2(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(project(tmp_path))
    assert cli.main(["nope"]) == 2
