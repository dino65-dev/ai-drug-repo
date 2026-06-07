#!/bin/bash
/teamspace/studios/this_studio/neuropharm_env/bin/python -c "
from transformers import AutoConfig, AutoModelForCausalLM
import torch, os
os.environ['HF_TOKEN'] = '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'

for mid, trust in [
    ('Qwen/Qwen3.5-4B', True),
    ('Qwen/Qwen3.5-4B', False),
    ('google/gemma-4-E4B-it', True),
    ('google/gemma-4-E4B-it', False),
]:
    try:
        c = AutoConfig.from_pretrained(mid, trust_remote_code=trust)
        print(f'{mid} (trust={trust}): arch={c.architectures} layers={c.num_hidden_layers} d={c.hidden_size}')
        # Also check if it can be loaded as CausalLM
        try:
            m = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=trust, device_map='cpu', torch_dtype=torch.float16, _fast_init=True)
            print(f'  CausalLM OK! type={type(m).__name__}')
            # Check for language_model access
            for attr in ['model', 'language_model', 'transformer', 'text_model']:
                if hasattr(m, attr):
                    sub = getattr(m, attr)
                    print(f'  has .{attr}: type={type(sub).__name__}')
                    if hasattr(sub, 'layers'):
                        print(f'    .{attr}.layers[0]: {type(sub.layers[0]).__name__}')
        except Exception as e2:
            print(f'  CausalLM FAIL: {str(e2)[:120]}')
    except Exception as e:
        print(f'{mid} (trust={trust}): FAIL {str(e)[:120]}')
    print()
" 2>&1 | tail -40
