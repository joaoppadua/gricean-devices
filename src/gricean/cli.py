from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import coding, generate, logprobs, neither, report, stimuli
from .lm import ladders, load_model, load_registry, neither_source

CONFIGS = Path("configs")


def _gen_cfg() -> dict:
    return yaml.safe_load((CONFIGS / "generation.yaml").read_text())


def _models_for(args, reg) -> list[tuple[str, bool]]:
    """(model_name, template) pairs to run. Tuned models run templated and raw; base only raw."""
    names = list(reg) if args.all else [args.model]
    pairs = []
    for n in names:
        spec = reg[n]
        if spec.template == "none":
            pairs.append((n, False))
        elif args.all:
            pairs += [(n, True), (n, False)]
        else:
            pairs.append((n, not args.raw))
    return pairs


def cmd_freeze(args, loader) -> int:
    print(stimuli.freeze())
    return 0


def cmd_neither(args, loader) -> int:
    reg, items, cfg = load_registry(), stimuli.load_all(), _gen_cfg()["neither"]
    chk = stimuli.current_checksum()
    for family in args.family:
        src = reg[neither_source()[family]]
        n = neither.sample_neither(loader(src), src, family, items, cfg=cfg, stimuli_checksum=chk)
        print(f"neither[{family}] from {src.name}: {n} rows")
    return 0


def cmd_generate(args, loader) -> int:
    reg, items, cfg = load_registry(), stimuli.load_all(), _gen_cfg()
    chk = stimuli.current_checksum()
    for name, template in _models_for(args, reg):
        spec = reg[name]
        n = generate.run_generate(loader(spec), spec, items, template=template, gen_cfg=cfg, stimuli_checksum=chk)
        print(f"generate[{name}, template={template}]: {n} rows")
    return 0


def cmd_logprobs(args, loader) -> int:
    reg, items = load_registry(), stimuli.load_all()
    chk = stimuli.current_checksum()
    for name, template in _models_for(args, reg):
        spec = reg[name]
        nei = neither.load_neither(spec.family)
        n = logprobs.run_logprobs(loader(spec), spec, items, nei, template=template, stimuli_checksum=chk)
        print(f"logprobs[{name}, template={template}]: {n} rows")
    return 0


def cmd_sheet(args, loader) -> int:
    out = Path("data/coding") / ("coder2.csv" if args.coder2 else "coder1.csv")
    n = coding.export_sheet(Path("data/responses"), out, coder="coder2" if args.coder2 else "coder1",
                            sample_frac=args.frac if args.coder2 else None, seed=0)
    print(f"sheet {out}: {n} rows")
    return 0


def cmd_report(args, loader) -> int:
    models = ladders()[args.ladder]
    c2 = Path("data/coding/coder2.csv")
    summary = report.build(Path("data/coding/coder1.csv"), Path("data/coding/coder2.key.csv") if c2.exists() else None,
                           c2 if c2.exists() else None, Path("data/responses"), Path("data/logprobs"),
                           models, Path("paper") / args.ladder)
    print(summary)
    return 0


def cmd_finetune(args, loader) -> int:
    from . import finetune
    cfg = yaml.safe_load((CONFIGS / "finetune.yaml").read_text())
    reg = load_registry()
    base = reg[cfg["base"]]
    records = finetune.load_dolly()
    sets = finetune.prepare(records, sizes=cfg["sizes"], seed=cfg["seed"], shuffle_responses=True)
    names = args.set or list(sets)
    for name in names:
        out = Path("adapters") / f"smollm_{name}"
        finetune.write_jsonl(sets[name], out / "train.jsonl")
        finetune.train_lora(base, sets[name], out, cfg)
        print(f"finetune[{name}]: {len(sets[name])} rows -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gricean")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    n = sub.add_parser("neither"); n.add_argument("--family", action="append", required=True); n.set_defaults(fn=cmd_neither)
    for name, fn in (("generate", cmd_generate), ("logprobs", cmd_logprobs)):
        s = sub.add_parser(name)
        s.add_argument("--model"); s.add_argument("--all", action="store_true"); s.add_argument("--raw", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("sheet"); s.add_argument("--coder2", action="store_true"); s.add_argument("--frac", type=float, default=0.25)
    s.set_defaults(fn=cmd_sheet)
    r = sub.add_parser("report"); r.add_argument("--ladder", default="olmo"); r.set_defaults(fn=cmd_report)
    f = sub.add_parser("finetune"); f.add_argument("--set", action="append"); f.set_defaults(fn=cmd_finetune)
    return p


def main(argv: list[str] | None = None, loader=load_model) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:          # argparse rejects unknown subcommands with code 2
        return int(e.code or 0)
    if not getattr(args, "fn", None):
        build_parser().print_help()
        return 2
    return args.fn(args, loader)


if __name__ == "__main__":
    sys.exit(main())
