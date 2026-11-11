from pathlib import Path
import pytest
from gricean import lm as L


def test_fake_generate_is_deterministic(fake_lm):
    a = fake_lm.generate("Hello there", max_new_tokens=5, temperature=0.7, seed=1)
    b = fake_lm.generate("Hello there", max_new_tokens=5, temperature=0.7, seed=1)
    assert a == b and a.startswith("[fake s1]")


def test_fake_logprob_uses_rules(fake_lm):
    lp, n = fake_lm.continuation_logprob("Can you tell me the time?", "It is noon.")
    assert lp == -1.0 and n == 3
    lp2, _ = fake_lm.continuation_logprob("x", "unknown text")
    assert lp2 == -10.0 * 2


def test_fake_format_template_flag(fake_lm):
    assert fake_lm.format("hi", template=False) == "hi"
    assert fake_lm.format("hi", template=True) == "<user>hi</user><assistant>"


CFG = Path("configs/models.yaml")


def test_registry_rev2_ladders_and_neither_source():
    reg, lad = L.load_registry(CFG), L.ladders(CFG)
    assert lad["olmo"] == ["olmo_stage1", "olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]
    assert lad["selfrun"] == ["olmo_stage1_alpaca", "olmo_n500", "olmo_n2000", "olmo_full", "olmo_format_only"]
    assert set(lad) == {"olmo", "selfrun"}
    assert L.neither_source(CFG) == {"olmo": "pythia_base"}
    s1, s1a = reg["olmo_stage1"], reg["olmo_stage1_alpaca"]
    import re
    assert s1.repo == s1a.repo == "allenai/OLMo-2-0425-1B"
    # `make pin` replaces the stage-1 branch name with its commit sha; either way both rungs share it
    assert s1.revision == s1a.revision != reg["olmo_base"].revision
    assert s1.revision == "stage1-step1907359-tokens4001B" or re.fullmatch(r"[0-9a-f]{40}", s1.revision)
    assert s1.template == "none" and s1.raw_control is True
    assert s1a.template == "alpaca" and s1a.raw_control is False
    assert all(reg[n].family == "olmo" and reg[n].raw_control is False and reg[n].adapter == f"adapters/{n}"
               for n in lad["selfrun"][1:])
    assert not any(n.startswith("smollm") for n in reg)


def test_registry_rejects_raw_control_false_on_base_model(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("models:\n  b: {repo: x, revision: main, family: f, template: none, raw_control: false}\n")
    with pytest.raises(ValueError, match="raw_control"):
        L.load_registry(p)


def test_pin_registry_resolves_branch_style_revision(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("models:\n  s1: {repo: org/a, revision: stage1-step1907359-tokens4001B, family: f, template: none}\n")
    assert L.pin_registry(p, resolver=lambda repo, rev: "a" * 40) == {"s1": "a" * 40}
    assert L.load_registry(p)["s1"].revision == "a" * 40


def test_alpaca_format():
    spec = L.ModelSpec(name="s", repo="r", revision="main", family="smollm", template="alpaca", adapter=None)
    assert L.render_template(spec, "Do X", tokenizer=None) == "### Instruction:\nDo X\n\n### Response:\n"


@pytest.mark.slow
def test_hf_logprob_boundary_space():
    spec = L.ModelSpec(name="tiny", repo="sshleifer/tiny-gpt2", revision="main",
                       family="test", template="none", adapter=None)
    m = L.load_model(spec, device="cpu")
    lp_a, n_a = m.continuation_logprob("The cat", " sat")
    lp_b, n_b = m.continuation_logprob("The cat", "sat")
    assert n_a >= 1 and n_b >= 1
    assert lp_a != lp_b  # tokenized at the true boundary, not by string concatenation
    out = m.generate("The cat", max_new_tokens=3, temperature=None, seed=None)
    assert isinstance(out, str)


class _StubTok:
    pad_token_id = 0
    eos_token = "<eos>"
    def __init__(self):
        import torch
        self._t = torch
    def __call__(self, text, add_special_tokens, return_tensors):
        n = 0 if text == "" else 3
        class R: pass
        r = R(); r.input_ids = self._t.ones((1, n), dtype=self._t.long); return r
    def decode(self, ids, skip_special_tokens):
        return "out"


class _StubModel:
    def __init__(self):
        self.calls = []
    def generate(self, ids, **kwargs):
        import torch
        self.calls.append(kwargs)
        return torch.cat([ids, torch.tensor([[4, 5]])], dim=1)


def _stub_hflm():
    spec = L.ModelSpec(name="s", repo="r", revision="main", family="f", template="none")
    model = _StubModel()
    return L.HFLM(spec, model, _StubTok(), device="cpu"), model


def test_hf_generate_pins_every_decoding_knob_explicitly():
    lm, model = _stub_hflm()
    lm.generate("p", max_new_tokens=5, temperature=0.7, seed=1)
    lm.generate("p", max_new_tokens=5, temperature=None, seed=None)
    sample, greedy = model.calls
    for kw in (sample, greedy):
        assert kw["top_k"] == 0 and kw["top_p"] == 1.0 and kw["repetition_penalty"] == 1.0 and kw["num_beams"] == 1
    assert sample["do_sample"] is True and sample["temperature"] == 0.7
    assert greedy["do_sample"] is False


def test_continuation_logprob_rejects_empty_continuation():
    lm, _ = _stub_hflm()
    with pytest.raises(ValueError, match="empty continuation"):
        lm.continuation_logprob("prompt", "")


def test_resolve_revision_passes_sha_through():
    sha = "a" * 40
    assert L.resolve_revision("any/repo", sha) == sha


def test_resolve_revision_reads_local_cache_ref(tmp_path: Path, monkeypatch):
    import huggingface_hub.constants as C
    monkeypatch.setattr(C, "HF_HUB_CACHE", str(tmp_path))
    ref = tmp_path / "models--allenai--OLMo-2-0425-1B" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("b" * 40 + "\n")
    assert L.resolve_revision("allenai/OLMo-2-0425-1B", "main") == "b" * 40


def test_spec_stamp_tracks_adapter_weights(tmp_path: Path):
    ad = tmp_path / "adapter"; ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(b"v1")
    spec = L.ModelSpec(name="s", repo="r", revision="x", family="f", template="alpaca", adapter=str(ad))
    a = L.spec_stamp(spec)
    (ad / "adapter_model.safetensors").write_bytes(b"v2")
    assert L.spec_stamp(spec) != a


def test_registry_has_pythia_neither_candidate_outside_every_ladder():
    cfg = Path("configs/models.yaml")
    reg, lad = L.load_registry(cfg), L.ladders(cfg)
    assert reg["pythia_base"].template == "none"
    assert all("pythia_base" not in models for models in lad.values())


def test_pin_registry_rewrites_symbolic_revisions_only(tmp_path: Path):
    import textwrap
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent("""
        # keep me
        models:
          a: {repo: org/a, revision: main, family: f, template: none}
          b: {repo: org/b, revision: 0123456789abcdef0123456789abcdef01234567, family: f, template: none}
        ladders: {f: [a, b]}
        neither_source: {f: a}
    """))
    changed = L.pin_registry(p, resolver=lambda repo, rev: "f" * 40)
    assert changed == {"a": "f" * 40}
    text = p.read_text()
    assert "# keep me" in text
    reg = L.load_registry(p)
    assert reg["a"].revision == "f" * 40
    assert reg["b"].revision == "0123456789abcdef0123456789abcdef01234567"
