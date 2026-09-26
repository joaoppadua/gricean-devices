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


def test_registry_loads_specs_and_ladders():
    reg = L.load_registry(Path("configs/models.yaml"))
    assert reg["olmo_sft"].template == "chat"
    assert reg["olmo_base"].family == "olmo"
    assert L.ladders(Path("configs/models.yaml"))["olmo"] == ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]
    assert L.neither_source(Path("configs/models.yaml"))["olmo"] == "qwen_base"


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
