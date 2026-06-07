#!/bin/bash
set -e
source ~/neuropharm_env/bin/activate
echo "==python=="; python -V
echo "==install torch cu121=="
pip install --quiet torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
echo "==verify cuda=="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0))"
