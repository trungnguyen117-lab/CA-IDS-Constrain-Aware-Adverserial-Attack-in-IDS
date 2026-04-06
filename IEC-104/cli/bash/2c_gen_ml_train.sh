#!/bin/bash
# Step 2c: Generate ML adversarial (BB direct) from train data (for AT)
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"
python 1_generate_adv.py --target cat rf --attack zoo hsja --source train --device "$DEVICE"
