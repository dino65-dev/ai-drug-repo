import os
os.environ['HF_TOKEN'] = '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'
from transformers import AutoConfig

for mid in ['Qwen/Qwen3.5-4B', 'google/gemma-4-E4B-it']:
    c = AutoConfig.from_pretrained(mid, trust_remote_code=True)
    print(f'=== {mid} ===')
    print(f'  type: {type(c).__name__}')
    print(f'  architectures: {c.architectures}')

    for attr in ['num_hidden_layers','num_layers','num_decoder_layers',
                 'n_layers','num_transformer_layers','num_text_layers',
                 'llm_config','text_config','model_type',
                 'hidden_size','num_attention_heads','num_key_value_heads',
                 'intermediate_size','max_position_embeddings',
                 'head_dim','tie_word_embeddings']:
        if hasattr(c, attr):
            v = getattr(c, attr)
            if attr.endswith('_config') and hasattr(v, '__dict__'):
                for a2 in ['num_hidden_layers','num_layers','hidden_size']:
                    if hasattr(v, a2):
                        print(f'  {attr}.{a2}: {getattr(v, a2)}')
            else:
                print(f'  {attr}: {v}')
    print()
