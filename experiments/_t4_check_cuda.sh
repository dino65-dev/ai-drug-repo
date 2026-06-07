#!/bin/bash
source /teamspace/studios/this_studio/neuropharm_env/bin/activate
timeout 30 python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>&1 | head -20
