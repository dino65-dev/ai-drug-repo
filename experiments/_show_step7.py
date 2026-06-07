import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open('artifacts/step7_cross_model.json'))
print('=== Cross-model summary ===')
print(f'Model: {d["model"]}')
print(f'Quantization: {d["quantization"]}')
print(f'Layers: {d["n_layers"]}  d_model: {d["d_model"]}  layer used: {d["layer"]}')
print(f'v_drug norm: {d["drug_norm"]:.2f}  v_harm norm: {d["harm_norm"]:.2f}')
print(f'cos(v_drug, v_harm): {d["overlap_drug_harm"]:+.3f}')
print(f'cos(v_antidote, v_harm): {d["antidote_overlap_to_harm"]:+.3f}')
print()
print('=== Dose sweep (drug on 2 prompts) ===')
for c, runs in d['dose_sweep'].items():
    avg_c = sum(r['confident'] for r in runs) / len(runs)
    avg_h = sum(r['hedged'] for r in runs) / len(runs)
    print(f'  c={c:>5}: avg_confident={avg_c:.2f}  avg_hedged={avg_h:.2f}')
    for r in runs:
        print(f"    {r['prompt'][:50]!r}")
        print(f"    {r['generation'][:80]!r}")
print()
print('=== Antidote comparison (6 prompts, c=1.0) ===')
for r in d['antidote_results']:
    b = r['baseline']
    print(f"\n  prompt: {r['prompt'][:50]!r}")
    print(f"    baseline: c={b['confident']:>2} h={b['hedged']:>2} ref={b['refusals']:>2} hw={b['harm_words']:>2}")
    for name, run in r['runs'].items():
        print(f"    {name:>9}: c={run['confident']:>2} h={run['hedged']:>2} ref={run['refusals']:>2} hw={run['harm_words']:>2}")
