#!/bin/bash
set -e
source /teamspace/studios/this_studio/neuropharm_env/bin/activate

echo "=== install transformers from main (support for qwen3_5 / gemma4) ==="
pip install --quiet git+https://github.com/huggingface/transformers.git 2>&1 | tail -5

echo "=== version ==="
python -c "import transformers; print('transformers', transformers.__version__)"

echo "=== testing models ==="
python -c "
from transformers import AutoConfig, AutoModelForCausalLM
import torch, os
os.environ['HF_TOKEN'] = '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'

for mid in ['Qwen/Qwen3.5-4B', 'google/gemma-4-E4B-it']:
    try:
        c = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        print(f'{mid}: arch={c.architectures[0]} layers={c.num_hidden_layers} d={c.hidden_size} heads={c.num_attention_heads}')
        # Quick CausalLM load test (cpu, no weights)
        try:
            m = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, device_map='meta', torch_dtype=torch.float16)
            print(f'  CausalLM OK: type={type(m).__name__}')
            for attr in ['model', 'language_model', 'transformer', 'text_model']:
                if hasattr(m, attr):
                    sub = getattr(m, attr)
                    print(f'  has .{attr}: type={type(sub).__name__}')
        except Exception as e2:
            print(f'  CausalLM meta-load: {str(e2)[:120]}')
    except Exception as e:
        print(f'{mid}: FAIL {str(e)[:120]}')
    print()
" 2>&1 | tail -30
