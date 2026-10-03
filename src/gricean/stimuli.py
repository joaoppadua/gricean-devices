from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = (
    "indirect_request", "pattern_continuation", "embedded_instruction",
    "malformed_instruction", "underspecified_request", "rhetorical_question",
)
VERSIONS = ("full", "stripped", "embedded")
ABLATED_CATEGORIES = ("indirect_request", "embedded_instruction", "pattern_continuation")
LANGUAGES = ("en", "pt")


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    language: str
    control: bool
    prompt: str
    literal_reading: str
    intended_reading: str
    literal_continuation: str | None
    intended_continuation: str | None
    versions: dict[str, str]


def _strip(prompt: str, ablation: dict) -> str:
    text = prompt
    for marker in sorted(ablation["politeness"], key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(marker) + r"\b", "", text, flags=re.IGNORECASE)
    text = text.translate(str.maketrans("", "", ablation["strip_punctuation"]))
    return re.sub(r"\s+", " ", text).strip()


def make_versions(prompt: str, category: str, ablation: dict, language: str = "en",
                  control: bool = False) -> dict[str, str]:
    versions = {"full": prompt}
    if control or category not in ABLATED_CATEGORIES:
        return versions
    versions["stripped"] = _strip(prompt, ablation)
    versions["embedded"] = ablation["embedding"][language].format(prompt=prompt)
    return versions


def _load_ablation(root: Path) -> dict:
    return yaml.safe_load((root / "ablation.yaml").read_text())


def _stimulus_files(root: Path) -> list[Path]:
    return sorted(p for lang in LANGUAGES for p in (root / lang).glob("*.yaml"))


def load_all(root: Path = Path("stimuli")) -> list[Item]:
    root = Path(root)
    ablation = _load_ablation(root)
    items: list[Item] = []
    for path in _stimulus_files(root):
        doc = yaml.safe_load(path.read_text())
        category, language = doc["category"], doc["language"]
        if category not in CATEGORIES:
            raise ValueError(f"{path}: unknown category {category!r}")
        if language not in LANGUAGES:
            raise ValueError(f"{path}: unknown language {language!r}")
        for raw in doc["items"] or []:
            control = bool(raw.get("control", False))
            items.append(Item(
                id=raw["id"], category=category, language=language, control=control,
                prompt=raw["prompt"], literal_reading=raw["literal_reading"],
                intended_reading=raw["intended_reading"],
                literal_continuation=raw.get("literal_continuation"),
                intended_continuation=raw.get("intended_continuation"),
                versions=make_versions(raw["prompt"], category, ablation, language, control),
            ))
    validate(items)
    return items


def validate(items: list[Item]) -> None:
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"duplicate id {it.id!r}")
        seen.add(it.id)
        if not it.control and not (it.literal_continuation and it.intended_continuation):
            raise ValueError(f"{it.id}: dissociation items need both continuations")


def _digest(root: Path) -> tuple[list[tuple[str, str]], str]:
    files = [root / "ablation.yaml", *_stimulus_files(root)]
    rows = [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    combined = hashlib.sha256("\n".join(f"{n} {h}" for n, h in rows).encode()).hexdigest()
    return rows, combined


def freeze(root: Path = Path("stimuli")) -> str:
    root = Path(root)
    load_all(root)  # validates
    rows, combined = _digest(root)
    lines = [f"{h}  {n}" for n, h in rows] + [f"{combined}  COMBINED"]
    (root / "CHECKSUMS").write_text("\n".join(lines) + "\n")
    return combined


def current_checksum(root: Path = Path("stimuli")) -> str:
    """The frozen checksum, verified against the stimulus files on disk."""
    root = Path(root)
    last = (root / "CHECKSUMS").read_text().strip().splitlines()[-1]
    frozen = last.split()[0]
    live = _digest(root)[1]
    if live != frozen:
        raise ValueError("stimuli changed since they were frozen; run `gricean freeze` first")
    return frozen
