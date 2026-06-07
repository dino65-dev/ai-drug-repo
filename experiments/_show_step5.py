import json
d = json.load(open('artifacts/sae_cache/dense_vs_sparse.json'))
print('top-level keys:', [k for k in d if k != 'tests'])
print('num tests:', len(d['tests']))
for t in d['tests']:
    print(f"=== {t['tag']} ===")
    print(f"  baseline gen: {t['baseline']['generation'][:80]!r}")
    print(f"  baseline metrics: {t['baseline']['metrics']}")
    print(f"  dense ({len(t['dense'])} runs):")
    for r in t['dense']:
        print(f"    c={r['coefficient']:+.1f}  metrics={r['metrics']}  | {r['generation'][:60]!r}")
    print(f"  sparse_replace ({len(t['sparse_replace'])} runs):")
    for r in t['sparse_replace']:
        print(f"    B={r['boost']:>4.1f}  metrics={r['metrics']}  | {r['generation'][:60]!r}")
    print(f"  sparse_additive ({len(t['sparse_additive'])} runs):")
    for r in t['sparse_additive']:
        print(f"    B={r['boost']:>4.1f}  metrics={r['metrics']}  | {r['generation'][:60]!r}")
