#!/bin/bash
source /teamspace/studios/this_studio/neuropharm_env/bin/activate
cd /home/zeus
export HF_TOKEN=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
# install bitsandbytes if not already
pip install --quiet bitsandbytes 2>&1 | tail -3
# Run step7
python step7_cross_model.py
