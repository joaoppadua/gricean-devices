from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import coding, generate, logprobs, neither, report, results, stimuli, turns
from .lm import expected_templates, ladders, load_model, load_registry, neither_source

CONFIGS = Path("configs")


def _gen_cfg() -> dict:
    return yaml.safe_load((CONFIGS / "generation.yaml").read_text())


def _models_for(args, reg) -> list[tuple[str, bool]]:
    """(model_name, template) pairs to run. Tuned models run templated and raw; base only raw.

    `--all` means every model in some ladder, in ladder order: registry entries that serve only as a
    `neither` source are never generated or scored. Templated models also run raw (the template
    control) unless the registry marks them `raw_control: false` (self-run ladder, spec §3.4).
    """
    if args.all:
        names = list(dict.fromkeys(n for ms in ladders().values() for n in ms))
    else:
        names = [args.model]
    pairs = []
    for n in names:
        spec = reg[n]
        if args.all and spec.adapter and not (Path(spec.adapter) / "adapter_model.safetensors").exists():
            print(f"skip[{n}]: adapter {spec.adapter} not trained yet (run `make finetune`)")
            continue
        if spec.template == "none":
            pairs.append((n, False))
        elif args.all:
            pairs += [(n, True)] + ([(n, False)] if spec.raw_control else [])
        else:
            if args.raw and not spec.raw_control:
                raise ValueError(f"{n} has raw_control: false (self-run ladder, spec §3.4); it never runs raw")
            pairs.append((n, not args.raw))
    return pairs


def cmd_freeze(args, loader) -> int:
    print(stimuli.freeze(partial=args.partial))
    return 0


def _items() -> list:
    """Stimuli for a model stage: a partial freeze lets the set be written, but no model runs on a TODO item."""
    items = stimuli.load_all()
    if left := stimuli.check_placeholders(items):
        raise ValueError("unfilled placeholder items (fill them, then `make freeze`):\n  " + "\n  ".join(left))
    return items


def cmd_neither(args, loader) -> int:
    reg, items, cfg = load_registry(), _items(), _gen_cfg()["neither"]
    chk = stimuli.current_checksum()
    for family in args.family:
        src = reg[neither_source()[family]]
        lm = loader(src)
        src = getattr(lm, "spec", src)          # HF backend resolves 'main' to a commit sha
        n = neither.sample_neither(lm, src, family, items, cfg=cfg, stimuli_checksum=chk)
        print(f"neither[{family}] from {src.name}: {n} rows")
    return 0


def cmd_generate(args, loader) -> int:
    reg, items, cfg = load_registry(), _items(), _gen_cfg()
    chk = stimuli.current_checksum()
    for name, template in _models_for(args, reg):
        lm = loader(reg[name])
        spec = getattr(lm, "spec", reg[name])
        n = generate.run_generate(lm, spec, items, template=template, gen_cfg=cfg, stimuli_checksum=chk)
        print(f"generate[{name}, template={template}]: {n} rows")
    return 0


def cmd_logprobs(args, loader) -> int:
    reg, items, cfg = load_registry(), _items(), _gen_cfg()
    chk = stimuli.current_checksum()
    for name, template in _models_for(args, reg):
        nei = neither.load_neither(reg[name].family)
        lm = loader(reg[name])
        spec = getattr(lm, "spec", reg[name])
        n = logprobs.run_logprobs(lm, spec, items, nei, template=template, stimuli_checksum=chk,
                                  raw_separator=cfg.get("raw_separator", " "))
        print(f"logprobs[{name}, template={template}]: {n} rows")
    return 0


def cmd_sheet(args, loader) -> int:
    out = Path("data/coding") / ("coder2.csv" if args.coder2 else "coder1.csv")
    readings = ({it.id: {f: getattr(it, f) for f in coding.READINGS} for it in stimuli.load_all()}
                if args.coder2 else None)       # coder 2 has no coding page: the readings travel with the rows
    n = coding.export_sheet(Path("data/responses"), out, coder="coder2" if args.coder2 else "coder1",
                            sample_frac=args.frac if args.coder2 else None, seed=0,
                            expected=expected_templates(), readings=readings)
    print(f"sheet {out}: {n} rows")
    return 0


def cmd_coder2(args, loader) -> int:
    sheet = Path("data/coding/coder2.csv")
    if args.batches:
        paths = coding.write_batches(sheet, Path(args.batches), args.n)
        print(f"coder2: {len(paths)} batches under {args.batches}")
    else:
        print(f"coder2: {coding.enter_codes(sheet, Path(args.enter))} rows coded in {sheet}")
    return 0


def cmd_report(args, loader) -> int:
    models = ladders()[args.ladder]
    c2 = Path("data/coding/coder2.csv")
    summary = report.build(Path("data/coding/coder1.csv"), Path("data/coding/coder2.key.csv") if c2.exists() else None,
                           c2 if c2.exists() else None, Path("data/responses"), Path("data/logprobs"),
                           models, Path("paper") / args.ladder, expected_templates())
    print(summary)
    if LENGTHS.exists() and CALIBRATION.exists():       # post hoc (spec 13.1); absent before `turns --count`
        print(_turn_tables(args.ladder))
    return 0


RESULTS_PAGE = Path("paper/RESULTS.md")


def cmd_results(args, loader) -> int:
    """Rewrite the generated blocks of the results page from the tables under `paper/`, or check them.

    The page is the author's; outside its marked blocks nothing is written (design of 2026-11-21)."""
    if not RESULTS_PAGE.exists():
        print(f"{RESULTS_PAGE} is not there; it is written by hand (see the design of 2026-11-21)")
        return 1
    try:
        up_to_date = results.update(RESULTS_PAGE, RESULTS_PAGE.parent, check=args.check)
    except ValueError as e:
        print(f"results: {e}")
        return 1
    if args.check and not up_to_date:
        print(f"results: {RESULTS_PAGE} is out of date; run `gricean results`")
        return 1
    read_as = results.describe(RESULTS_PAGE.read_text(encoding="utf-8"))
    print(f"results: {RESULTS_PAGE} {'is up to date' if up_to_date else 'rewritten'} ({read_as})")
    return 0


LENGTHS, CALIBRATION = Path("data/lengths/coded_en.csv"), Path("data/lengths/calibration.jsonl")


def _turn_tables(ladder: str) -> dict:
    models = ladders()[ladder]
    comparison = turns.COMPARISON if set(turns.COMPARISON) <= set(models) else None
    return turns.build(Path("data/coding/coder1.csv"), LENGTHS, CALIBRATION, models, Path("paper") / ladder,
                       expected_templates(), cap=_gen_cfg()["max_new_tokens"], comparison=comparison)


def cmd_turns(args, loader) -> int:
    """Turn completion (spec 13.1): calibrate one model, count the tokens of the coded rows, or write the tables."""
    reg, cap = load_registry(), _gen_cfg()["max_new_tokens"]
    if args.calibrate:
        turns.check_prompts(turns.CALIBRATION_PROMPTS, [it.prompt for it in stimuli.load_all()])
        spec = reg[args.calibrate]
        lm = loader(spec)
        rows = []
        for template in ([False, True] if spec.template != "none" else [False]):
            rows += turns.calibrate(lm, turns.CALIBRATION_PROMPTS, template=template, max_new_tokens=cap)
        turns.save_calibration(CALIBRATION, spec.name, rows)
        print(f"turns[calibrate {spec.name}]: {len(rows)} rows -> {CALIBRATION}")
    elif args.count:
        n = turns.write_lengths(Path("data/responses"), LENGTHS, expected_templates(),
                                lambda name: turns.load_tokenizer(reg[name]))
        print(f"turns[count]: {n} rows -> {LENGTHS}")
    else:
        print(_turn_tables(args.ladder))
    return 0


def cmd_finetune(args, loader) -> int:
    from . import finetune
    cfg = yaml.safe_load((CONFIGS / "finetune.yaml").read_text())
    reg = load_registry()
    base = reg[cfg["base"]]
    records = finetune.load_dolly(cfg["dataset_revision"])
    sets = finetune.prepare(records, sizes=cfg["sizes"], seed=cfg["seed"], shuffle_responses=True)
    names = args.set or list(sets)
    for name in names:
        out = Path("adapters") / f"{cfg['adapter_prefix']}_{name}"
        finetune.write_jsonl(sets[name], out / "train.jsonl")
        finetune.train_lora(base, sets[name], out, cfg)
        print(f"finetune[{name}]: {len(sets[name])} rows -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gricean")
    sub = p.add_subparsers(dest="cmd")
    fz = sub.add_parser("freeze"); fz.add_argument("--partial", action="store_true"); fz.set_defaults(fn=cmd_freeze)
    n = sub.add_parser("neither"); n.add_argument("--family", action="append", required=True); n.set_defaults(fn=cmd_neither)
    for name, fn in (("generate", cmd_generate), ("logprobs", cmd_logprobs)):
        s = sub.add_parser(name)
        s.add_argument("--model"); s.add_argument("--all", action="store_true"); s.add_argument("--raw", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("sheet"); s.add_argument("--coder2", action="store_true"); s.add_argument("--frac", type=float, default=0.25)
    s.set_defaults(fn=cmd_sheet)
    c = sub.add_parser("coder2"); c.add_argument("--n", type=int, default=8); c.set_defaults(fn=cmd_coder2)
    how = c.add_mutually_exclusive_group(required=True)
    how.add_argument("--batches", metavar="DIR"); how.add_argument("--enter", metavar="DIR")
    r = sub.add_parser("report"); r.add_argument("--ladder", default="olmo"); r.set_defaults(fn=cmd_report)
    rs = sub.add_parser("results"); rs.add_argument("--check", action="store_true"); rs.set_defaults(fn=cmd_results)
    t = sub.add_parser("turns"); t.set_defaults(fn=cmd_turns)
    what = t.add_mutually_exclusive_group(required=True)
    what.add_argument("--calibrate", metavar="MODEL"); what.add_argument("--count", action="store_true")
    what.add_argument("--ladder")
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
