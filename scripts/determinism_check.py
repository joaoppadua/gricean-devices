"""Greedy generation twice on the same machine; report whether outputs are byte-identical."""
import sys
from gricean.lm import load_model, load_registry
from gricean.stimuli import load_all

name = sys.argv[1] if len(sys.argv) > 1 else "olmo_base"
spec = load_registry()[name]
lm = load_model(spec)
items = [it for it in load_all() if not it.control][:5]
mismatch = 0
for it in items:
    a = lm.generate(lm.format(it.prompt, spec.template != "none"), max_new_tokens=60, temperature=None, seed=None)
    b = lm.generate(lm.format(it.prompt, spec.template != "none"), max_new_tokens=60, temperature=None, seed=None)
    mismatch += a != b
print(f"{name}: {len(items) - mismatch}/{len(items)} greedy outputs identical across two runs on {lm.device}")
sys.exit(1 if mismatch else 0)
