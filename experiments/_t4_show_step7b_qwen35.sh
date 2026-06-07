#!/bin/bash
cat /home/zeus/artifacts/step7b_qwen35.json 2>/dev/null | /home/zeus/miniconda3/envs/cloudspace/bin/python -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print('drug_norm:', d.get('drug_norm'))
    print('harm_norm:', d.get('harm_norm'))
    print('cos:', d.get('overlap_drug_harm'))
    for c, r in d.get('dose_sweep', {}).items():
        print(f'c={c} gen[0]={r[0][\"generation\"][:100]!r}')
        print(f'      gen[1]={r[1][\"generation\"][:100]!r}')
except Exception as e:
    print('no data or error:', e)
"
