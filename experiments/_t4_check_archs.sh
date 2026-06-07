#!/bin/bash
# Try loading model configs on T4
/teamspace/studios/this_studio/neuropharm_env/bin/python << 'EOF'
import os
os.environ['HF_TOKEN'] = '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'
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
for mid in ['Qwen/Qwen3.5-4B', 'google/gemma-4-E4B-it', 'google/gemma-4-E2B-it']:
    print(f'\n=== {mid} ===')
    try:
        c = AutoConfig.from_pretrained(mid)
        attrs = [a for a in dir(c) if not a.startswith('_') and 'layer' in a.lower() or 'hidden' in a.lower()]
        print(f'  type={type(c).__name__}')
        # Get all non-private numeric attributes
        for a in sorted(dir(c)):
            if a.startswith('_'): continue
            v = getattr(c, a, None)
            if isinstance(v, (int, float, str, bool, list)) and not callable(v):
                if a in ('architectures', 'model_type', 'torch_dtype', 'dtype'):
                    print(f'  {a}={v}')
        # Try specific attrs
        for a in ['num_hidden_layers', 'num_layers', 'n_layers', 'hidden_size', 'd_model', 'num_attention_heads']:
            v = getattr(c, a, None)
            if v is not None:
                print(f'  {a}={v}')
    except Exception as e:
        print(f'  ERROR: {str(e)[:200]}')
EOF
