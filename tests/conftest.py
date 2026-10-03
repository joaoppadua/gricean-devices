import pytest
from gricean.lm import FakeLM


@pytest.fixture
def fake_lm():
    return FakeLM("fake", logprob_rules={"It is noon.": -1.0, "Yes.": -3.0})
