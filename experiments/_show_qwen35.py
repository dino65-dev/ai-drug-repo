import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open("artifacts/step7b_qwen35.json"))

print("=== Qwen3.5-4B sample generations ===")
for r in d["antidote_results"]:
    print(f"\n--- {r['prompt'][:60]!r} ---")
    print(f"  baseline: {r['baseline']['generation'][:200]!r}")
    for name in ["clean", "antidote"]:
        print(f"  {name}: {r['runs'][name]['generation'][:200]!r}")

print("\n=== Dose sweep ===")
for c, runs in d["dose_sweep"].items():
    print(f"\n  c={c}:")
    for r in runs:
        print(f"    {r['prompt'][:50]!r}")
        print(f"    gen: {r['generation'][:150]!r}")
