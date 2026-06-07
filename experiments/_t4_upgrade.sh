#!/bin/bash
set -e
source /teamspace/studios/this_studio/neuropharm_env/bin/activate
pip install --quiet --no-deps "transformers==5.10.2" 2>&1 | tail -3
# also need to make sure tokenizers matches
pip install --quiet "tokenizers>=0.21,<0.22" 2>&1 | tail -3
# test
python -c "
import os
os.environ['HF_TOKEN'] = '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'
from transformers import AutoConfig
for mid in ['Qwen/Qwen3.5-4B', 'google/gemma-4-E4B-it']:
    try:
        c = AutoConfig.from_pretrained(mid)
        print(f'{mid}: OK type={type(c).__name__}')
    except Exception as e:
        print(f'{mid}: {str(e)[:120]}')
" 2>&1 | head -20
