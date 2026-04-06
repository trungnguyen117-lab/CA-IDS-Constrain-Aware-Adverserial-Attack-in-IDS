#!/bin/bash
# Step 2a: Generate DL adversarial (WB scaled-space) from train data (for AT)
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"
python 1_generate_adv.py --target lstm resdnn --attack fgsm pgd deepfool cw mim --source train --device "$DEVICE"
