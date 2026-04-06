#!/bin/bash
# Step 4: Adversarial training (retrain all models with adv data)
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"
python 3_adv_train.py --model all --device "$DEVICE"
