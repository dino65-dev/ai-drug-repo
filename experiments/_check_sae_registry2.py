from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
import json

d = get_pretrained_saes_directory()
print('Type:', type(d).__name__)
print('Top-level keys (first 10):', list(d.keys())[:10])

# Try to find Qwen entries
qwen_keys = [k for k in d if 'qwen' in k.lower()]
print(f'\nQwen-related keys: {len(qwen_keys)}')
for k in qwen_keys[:30]:
    print(f'  {k}')

# Check structure of one entry
if d:
    sample_key = list(d.keys())[0]
    sample = d[sample_key]
    print(f'\nSample entry [{sample_key}]:')
    if isinstance(sample, dict):
        for k, v in sample.items():
            print(f'  {k}: {type(v).__name__}', end='')
            if isinstance(v, str):
                print(f' = {v[:80]!r}')
            elif isinstance(v, list):
                print(f' (len={len(v)})')
            else:
                print()
