#!/bin/bash
# Step 2d: Generate ML adversarial (BB direct) from test data (for eval)
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"
python 1_generate_adv.py --target cat rf --attack zoo hsja --source test --device "$DEVICE"
