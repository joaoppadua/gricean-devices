"""Pin every `revision: main` in configs/models.yaml to a commit sha; prints what changed and where the sha came from."""
from pathlib import Path

from huggingface_hub import constants

from gricean import lm


def main() -> None:
    cfg = Path("configs/models.yaml")
    reg = lm.load_registry(cfg)
    source = {}
    for name, spec in reg.items():
        ref = Path(constants.HF_HUB_CACHE) / f"models--{spec.repo.replace('/', '--')}" / "refs" / spec.revision
        source[name] = "cache" if ref.exists() else "hub"
    changed = lm.pin_registry(cfg)
    for name, sha in changed.items():
        print(f"{name:20s} {reg[name].repo:40s} {sha}  ({source[name]})")
    if not changed:
        print("all revisions already pinned")


if __name__ == "__main__":
    main()
