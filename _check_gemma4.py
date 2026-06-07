from transformers import AutoConfig
for mid in ['google/gemma-4-E4B-it', 'google/gemma-4-E2B-it']:
    try:
        c = AutoConfig.from_pretrained(mid)
        print(f'{mid}:')
        print(f'  layers={c.num_hidden_layers} d={c.hidden_size} heads={c.num_attention_heads} kv={getattr(c,"num_key_value_heads","?")}')
    except Exception as e:
        print(f'{mid}: {str(e)[:120]}')
