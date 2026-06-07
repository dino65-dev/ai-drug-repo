#!/bin/bash
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
$PY -m pip install --quiet "bitsandbytes>=0.46.1" 2>&1 | tail -3
$PY -c "import bitsandbytes; print('bnb', bitsandbytes.__version__); import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
