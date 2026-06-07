#!/bin/bash
set -e
source /teamspace/studios/this_studio/neuropharm_env/bin/activate
echo "=== current versions ==="
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== upgrading transformers to latest ==="
pip install --quiet -U transformers 2>&1 | tail -5
echo "=== new version ==="
python -c "import transformers; print('transformers', transformers.__version__)"
echo "=== testing qwen3_5 and gemma4 ==="
python -c "
from transformers import AutoConfig
for mid in ['Qwen/Qwen3.5-4B', 'google/gemma-4-E4B-it', 'Qwen/Qwen2.5-1.5B-Instruct']:
    try:
        c = AutoConfig.from_pretrained(mid)
        print(f'{mid}: OK  arch={c.architectures[0] if c.architectures else \"?\"}  layers={c.num_hidden_layers} d={c.hidden_size}')
    except Exception as e:
        print(f'{mid}: FAIL {str(e)[:120]}')
" 2>&1 | tail -20
