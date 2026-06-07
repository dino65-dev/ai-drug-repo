#!/bin/bash
/teamspace/studios/this_studio/neuropharm_env/bin/python -c "
from transformers import AutoConfig
for mid in ['Qwen/Qwen3.5-4B', 'Qwen/Qwen2.5-1.5B-Instruct']:
    try:
        c = AutoConfig.from_pretrained(mid)
        print(f'{mid}: OK  layers={c.num_hidden_layers} d={c.hidden_size}')
    except Exception as e:
        print(f'{mid}: {str(e)[:100]}')
" 2>&1 | head -10
