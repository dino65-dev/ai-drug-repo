import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open("artifacts/step7b_qwen35.json"))

print("=== Qwen3.5-4B (LATEST - thinking disabled) ===")
print(f"v_drug={d['drug_norm']:.2f} v_harm={d['harm_norm']:.2f} cos={d['overlap_drug_harm']:+.3f}")

print("\n--- Dose sweep ---")
for c, runs in d["dose_sweep"].items():
    print(f"\nc={c}:")
    for r in runs:
        print(f"  prompt: {r['prompt'][:60]!r}")
        print(f"  gen[0]: {r['generation'][:200]!r}")
        print(f"  metrics: conf={r['confident']} hed={r['hedged']} ref={r['refusals']} hw={r['harm_words']}")

print("\n--- Antidote ---")
n = len(d["antidote_results"])
print(f"baseline: conf={sum(r['baseline']['confident'] for r in d['antidote_results'])/n:.2f}")
for k in ["clean", "contam", "antidote"]:
    print(f"  {k}: conf={sum(r['runs'][k]['confident'] for r in d['antidote_results'])/n:.2f} hed={sum(r['runs'][k]['hedged'] for r in d['antidote_results'])/n:.2f}")
