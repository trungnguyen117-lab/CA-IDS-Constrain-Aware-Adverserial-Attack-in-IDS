#!/bin/bash
# Step 3: Evaluate all models BEFORE adversarial training
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"
python 2_evaluate.py --target all --device "$DEVICE"
