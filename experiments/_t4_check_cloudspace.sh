#!/bin/bash
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
echo "Python: $($PY --version)"
$PY -c "import transformers; print('transformers', transformers.__version__); import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$PY -c "from transformers import AutoConfig; import os; os.environ['HF_TOKEN']='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'; c = AutoConfig.from_pretrained('google/gemma-4-E4B-it'); print('Gemma-4-E4B-it OK', type(c).__name__)"
