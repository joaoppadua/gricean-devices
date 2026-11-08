from pathlib import Path
import textwrap
import pytest
from gricean import stimuli as S


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write(tmp_path, "ablation.yaml", """
        politeness: ["please"]
        strip_punctuation: "?!.,"
        embedding:
          en: "Before. {prompt} After."
          pt: "Antes. {prompt} Depois."
    """)
    write(tmp_path, "en/indirect_request.yaml", """
        category: indirect_request
        language: en
        items:
          - id: en_ir_01
            control: false
            prompt: "Can you please tell me the time?"
            literal_reading: "ability"
            intended_reading: "request"
            literal_continuation: "Yes."
            intended_continuation: "It is noon."
          - id: en_ir_c01
            control: true
            prompt: "What time is it?"
            literal_reading: "request"
            intended_reading: "request"
            literal_continuation: null
            intended_continuation: null
    """)
    write(tmp_path, "en/rhetorical_question.yaml", """
        category: rhetorical_question
        language: en
        items:
          - id: en_rq_01
            control: false
            prompt: "Who likes waiting?"
            literal_reading: "question"
            intended_reading: "complaint"
            literal_continuation: "Some people do."
            intended_continuation: "Nobody, waiting is frustrating."
    """)
    return tmp_path


def test_load_all_builds_items_with_versions(root):
    items = S.load_all(root)
    ids = {i.id for i in items}
    assert ids == {"en_ir_01", "en_ir_c01", "en_rq_01"}
    ir = next(i for i in items if i.id == "en_ir_01")
    assert set(ir.versions) == {"full", "stripped", "embedded"}
    assert ir.versions["full"] == "Can you please tell me the time?"
    assert ir.versions["stripped"] == "Can you tell me the time"
    assert ir.versions["embedded"] == "Before. Can you please tell me the time? After."
    rq = next(i for i in items if i.id == "en_rq_01")
    assert set(rq.versions) == {"full"}          # not an ablated category


def test_controls_get_only_full_version(root):
    items = S.load_all(root)
    c = next(i for i in items if i.control)
    assert set(c.versions) == {"full"}


def test_validate_rejects_duplicate_ids(root):
    write(root, "pt/indirect_request.yaml", """
        category: indirect_request
        language: pt
        items:
          - id: en_ir_01
            control: false
            prompt: "Você pode me dizer as horas?"
            literal_reading: "a"
            intended_reading: "b"
            literal_continuation: "Posso."
            intended_continuation: "É meio-dia."
    """)
    with pytest.raises(ValueError, match="duplicate id"):
        S.load_all(root)


def test_validate_rejects_dissociation_item_without_continuations(root):
    write(root, "en/malformed_instruction.yaml", """
        category: malformed_instruction
        language: en
        items:
          - id: en_mi_01
            control: false
            prompt: "tel me the tiem"
            literal_reading: "a"
            intended_reading: "b"
            literal_continuation: null
            intended_continuation: "It is noon."
    """)
    with pytest.raises(ValueError, match="en_mi_01"):
        S.load_all(root)


def test_validate_rejects_unknown_category(root):
    write(root, "en/jokes.yaml", """
        category: jokes
        language: en
        items: []
    """)
    with pytest.raises(ValueError, match="jokes"):
        S.load_all(root)


def test_freeze_writes_checksums_and_is_stable(root):
    a = S.freeze(root)
    assert (root / "CHECKSUMS").exists()
    assert S.current_checksum(root) == a
    assert S.freeze(root) == a
    (root / "en/rhetorical_question.yaml").write_text(
        (root / "en/rhetorical_question.yaml").read_text().replace("waiting", "queuing"))
    assert S.freeze(root) != a


def test_current_checksum_raises_when_stimuli_changed_since_freeze(root):
    S.freeze(root)
    (root / "en/rhetorical_question.yaml").write_text(
        (root / "en/rhetorical_question.yaml").read_text().replace("waiting", "queuing"))
    with pytest.raises(ValueError, match="freeze"):
        S.current_checksum(root)
