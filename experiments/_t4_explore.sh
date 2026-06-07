#!/bin/bash
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
$PY << 'EOF'
import os
os.environ['HF_TOKEN'] = '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'
from transformers import AutoConfig
for mid in ['Qwen/Qwen3.5-4B', 'google/gemma-4-E4B-it']:
    print(f'\n=== {mid} ===')
    try:
        c = AutoConfig.from_pretrained(mid)
        # Print all non-private attributes
        keys = ['model_type', 'architectures', 'hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'max_position_embeddings', 'rope_theta', 'torch_dtype']
        for k in keys:
            v = getattr(c, k, None)
            if v is not None:
                print(f'  {k}={v}')
        # Try to find num layers
        for n in ['num_hidden_layers', 'num_layers', 'n_layers', 'num_decoder_layers']:
            if hasattr(c, n):
                print(f'  {n}={getattr(c, n)}')
        # Also text_config
        if hasattr(c, 'text_config'):
            tc = c.text_config
            for k in ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads', 'intermediate_size', 'vocab_size', 'max_position_embeddings']:
                if hasattr(tc, k):
                    print(f'  text_config.{k}={getattr(tc, k)}')
    except Exception as e:
        print(f'  ERROR: {str(e)[:200]}')
EOF
