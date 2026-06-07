#!/bin/bash
set -e

echo "=== creating conda env ==="
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda create -n neuro311 python=3.11 -y 2>&1 | tail -5

echo "=== activating ==="
conda activate neuro311
which python
python -V

echo "=== install torch cu121 ==="
pip install --quiet torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

echo "=== install transformers from main ==="
pip install --quiet git+https://github.com/huggingface/transformers.git 2>&1 | tail -5

echo "=== verify ==="
python -c "
import torch, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
"

echo "=== test new models ==="
python -c "
from transformers import AutoConfig, AutoModelForCausalLM
import torch, os
os.environ['HF_TOKEN'] = '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'

for mid in ['Qwen/Qwen3.5-4B', 'google/gemma-4-E4B-it']:
    try:
        c = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        print(f'{mid}: arch={c.architectures} layers={c.num_hidden_layers} d={c.hidden_size}')
    except Exception as e:
        print(f'{mid}: FAIL {str(e)[:150]}')
" 2>&1 | tail -20
