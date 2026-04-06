#!/bin/bash
# Step 5: Evaluate all models AFTER adversarial training
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"
python 2_evaluate.py --target all --at --device "$DEVICE"
