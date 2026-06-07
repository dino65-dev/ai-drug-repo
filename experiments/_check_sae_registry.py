import sae_lens
print('sae_lens', sae_lens.__version__)
from sae_lens import get_pretrained_saes_directory
df = get_pretrained_saes_directory()
print('Total SAEs in registry:', len(df))
print()
qwen_rows = df[df['model'].str.contains('Qwen', case=False, na=False)]
print(f'Qwen entries: {len(qwen_rows)}')
for _, r in qwen_rows.head(40).iterrows():
    print(f"  {r['model']:<40}  release={r.get('release','?')}  layer={r.get('layer','?')}")
