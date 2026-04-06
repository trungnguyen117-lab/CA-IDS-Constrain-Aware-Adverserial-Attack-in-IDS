#!/bin/bash
# Run full pipeline: train → generate adv → evaluate → AT → re-evaluate
set -e
cd "$(dirname "$0")/../pipeline"

DEVICE="${1:-cpu}"

echo "========== Step 1: Train baseline models =========="
python 0_train.py --model all --device "$DEVICE"

echo "========== Step 2a: Generate DL adv (WB) — train =========="
python 1_generate_adv.py --target lstm resdnn --attack fgsm pgd deepfool cw mim --source train --device "$DEVICE"

echo "========== Step 2b: Generate DL adv (WB) — test =========="
python 1_generate_adv.py --target lstm resdnn --attack fgsm pgd deepfool cw mim --source test --device "$DEVICE"

echo "========== Step 2c: Generate ML adv (BB) — train =========="
python 1_generate_adv.py --target cat rf --attack zoo hsja --source train --device "$DEVICE"

echo "========== Step 2d: Generate ML adv (BB) — test =========="
python 1_generate_adv.py --target cat rf --attack zoo hsja --source test --device "$DEVICE"

echo "========== Step 3: Evaluate BEFORE AT =========="
python 2_evaluate.py --target all --device "$DEVICE"

echo "========== Step 4: Adversarial training =========="
python 3_adv_train.py --model all --device "$DEVICE"

echo "========== Step 5: Evaluate AFTER AT =========="
python 2_evaluate.py --target all --at --device "$DEVICE"

echo "========== PIPELINE COMPLETE =========="
