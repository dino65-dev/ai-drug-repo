import json
d = json.load(open('artifacts/sae_cache/dense_vs_sparse_extra.json'))
for bname, b in d['behaviors'].items():
    print(f'\n=== {bname} ===')
    print(f"  drug norm = unknown (not saved)")
    print(f"  top features: {b['top_features']}")
    for p in b['prompts']:
        print(f"\n  -- {p['tag']} (prompt: {p['prompt'][:60]!r})")
        print(f"     baseline: {p['baseline']['generation'][:100]!r}")
        for r in p['dense']:
            print(f"     dense c={r['coefficient']:+.1f}: {r['generation'][:100]!r}")
        for r in p['sparse']:
            print(f"     sparse B={r['boost']:>4.1f}: {r['generation'][:100]!r}")
