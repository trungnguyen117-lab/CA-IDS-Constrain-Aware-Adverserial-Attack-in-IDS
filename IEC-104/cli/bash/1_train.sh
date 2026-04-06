#!/bin/bash
# Step 1: Train baseline models (CAT, RF, LSTM, ResDNN)
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"
python 0_train.py --model all --device "$DEVICE"
