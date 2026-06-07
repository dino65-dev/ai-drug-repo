#!/bin/bash
set -e
source ~/neuropharm_env/bin/activate

echo "==install numpy transformers=="
pip install --quiet numpy 2>&1 | tail -3
# transformer-lens 2.16.1 on Python 3.9 needs transformers>=4.51
pip install --quiet "transformers>=4.51,<4.55" 2>&1 | tail -3

echo "==install transformer_lens=="
pip install --quiet transformer_lens==2.16.1 2>&1 | tail -3

echo "==install datasets=="
pip install --quiet datasets 2>&1 | tail -3

echo "==verify=="
python -c "
import torch, transformers, datasets, numpy, transformer_lens
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0))
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('numpy', numpy.__version__)
import transformer_lens; print('transformer_lens OK')
"
