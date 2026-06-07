import json
data = json.load(open('artifacts/step4_attack.json'))

print('=== Test 1: Counteracting prompts ===')
for r in data['tests']['counter_prompts']:
    print(f"c={r['coefficient']:+.1f}  conf={r['confident_hits']:>2}  hedged={r['hedged_hits']:>2}  garb={r['garbled']:.2f}")
    print(f"  prompt: {r['prompt'][:70]!r}")
    print(f"  gen:    {r['generation'][:140]!r}")
    print()

print('=== Test 2: OOD prompts ===')
for r in data['tests']['ood_prompts']:
    print(f"{r['tag']:<16} c={r['coefficient']:+.1f}  conf={r['confident_hits']:>2}  hedged={r['hedged_hits']:>2}  ontopic={r['on_topic']:.2f}  refusals={r['refusals']}  len={r['length']:>3}")
    print(f"  gen: {r['generation'][:120]!r}")
    print()

print('=== Test 3: Extended dose sweep ===')
for r in data['tests']['extended_sweep']:
    print(f"c={r['coefficient']:+5.1f}  conf={r['confident_hits']:>2}  hedged={r['hedged_hits']:>2}  garb={r['garbled']:.2f}  rep={r['repetition']:.2f}  len={r['length']:>3}  | {r['generation'][:80]!r}")
