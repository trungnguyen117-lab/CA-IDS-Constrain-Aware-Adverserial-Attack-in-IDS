#!/bin/bash
# V2 Pipeline — individual steps (not run_all.py)
# Usage: bash run_v2.sh [device]
#        bash run_v2.sh cuda
#        bash run_v2.sh mps

set -e

# cd to the directory containing this script (cli/pipeline/)
cd "$(dirname "$0")"

DEVICE=${1:-mps}
VERSION=${2:-v2}
LOG_LEVEL=${3:-INFO}

echo "=== V2 Pipeline: device=$DEVICE, version=$VERSION ==="

# # 0. Train baseline models
# echo ">>> Step 0: Train baseline $VERSION"
# python 0_train.py --model all --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# # 1a. Generate DL adversarial (train source, for AT)
# echo ">>> Step 1a: Generate DL adv (train, WB + BB)"
# python 1_generate_adv.py \
#     --target lstm resdnn \
#     --attack fgsm pgd deepfool cw mim jsma zoo hsja \
#     --source train --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# 1b. Generate DL adversarial (test source, for eval)
# echo ">>> Step 1b: Generate DL adv (test, WB + BB)"
# python 1_generate_adv.py \
#     --target lstm resdnn \
#     --attack fgsm pgd deepfool cw mim jsma zoo hsja\
#     --source test --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# 1c. Generate ML adversarial (train, BB)
echo ">>> Step 1c: Generate ML adv (train, BB)"
python 1_generate_adv.py \
    --target cat rf \
    --attack zoo hsja \
    --source train --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# 1d. Generate ML adversarial (test, BB)
echo ">>> Step 1d: Generate ML adv (test, BB)"
python 1_generate_adv.py \
    --target cat rf \
    --attack zoo hsja \
    --source test --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# 2. Evaluate BEFORE AT
echo ">>> Step 2: Evaluate BEFORE AT"
python 2_evaluate.py --target all --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# 2b. Evaluate ensemble BEFORE AT
echo ">>> Step 2b: Evaluate ensemble BEFORE AT"
python 4_evaluate_ensemble.py --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# 3. Adversarial training
echo ">>> Step 3: Adversarial training $VERSION"
python 3_adv_train.py --model all --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# 4. Evaluate AFTER AT
echo ">>> Step 4: Evaluate AFTER AT"
python 2_evaluate.py --target all --at --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

# 4b. Evaluate ensemble AFTER AT
echo ">>> Step 4b: Evaluate ensemble AFTER AT"
python 4_evaluate_ensemble.py --at true --version $VERSION --device $DEVICE --log-level $LOG_LEVEL

echo "=== V2 Pipeline COMPLETE ==="
