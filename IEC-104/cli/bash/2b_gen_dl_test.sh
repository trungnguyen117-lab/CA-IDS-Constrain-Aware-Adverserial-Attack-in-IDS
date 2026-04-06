#!/bin/bash
# Step 2b: Generate DL adversarial (WB scaled-space) from test data (for eval)
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"
python 1_generate_adv.py --target lstm resdnn --attack fgsm pgd deepfool cw mim --source test --device "$DEVICE"
