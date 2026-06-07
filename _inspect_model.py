import os, sys
os.environ['HF_TOKEN'] = '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'
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
mid = 'google/gemma-4-E4B-it'
print(f'Loading {mid} in 4-bit for structure check...')
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
m = AutoModelForCausalLM.from_pretrained(mid, quantization_config=bnb, device_map='cuda', trust_remote_code=True)
print(f'Type: {type(m).__name__}')
print('Top-level attrs:', [a for a in dir(m) if not a.startswith('_') and not callable(getattr(m,a))][:20])
for attr in ['model', 'language_model', 'transformer', 'text_model', 'base_model']:
    if hasattr(m, attr):
        sub = getattr(m, attr)
        print(f'  .{attr}: type={type(sub).__name__}')
        for a2 in ['layers', 'decoder', 'embed_tokens']:
            if hasattr(sub, a2):
                sub2 = getattr(sub, a2)
                if a2 == 'layers':
                    print(f'    .{attr}.layers: len={len(sub2)}  layer[0]={type(sub2[0]).__name__}')
                    # Sample layer attrs
                    l0 = sub2[0]
                    print(f'      layer[0] attrs: {[a for a in dir(l0) if not a.startswith("_") and not callable(getattr(l0,a))][:20]}')
                else:
                    print(f'    .{attr}.{a2}: {type(sub2).__name__}')
print('VRAM:', torch.cuda.memory_allocated()/1e9, 'GB')
del m; torch.cuda.empty_cache()
print('Done.')
