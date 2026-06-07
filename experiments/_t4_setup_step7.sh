#!/bin/bash
set -e
source ~/neuropharm_env/bin/activate
cd /home/zeus
# HuggingFace token for gated Gemma
export HF_TOKEN=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
# Install bitsandbytes
pip install --quiet bitsandbytes 2>&1 | tail -3
# Upload the step7 script
mkdir -p experiments
echo "ready"
