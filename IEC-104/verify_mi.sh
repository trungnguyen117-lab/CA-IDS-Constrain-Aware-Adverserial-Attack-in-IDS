#!/bin/bash
# Autoresearch verify script: extract worst-case ASR + guard clean acc
cd /Users/trungnguye2n.dev/foami-plus && source .venv/bin/activate && cd IEC-104

OUTPUT=$(python cli/pipeline/5_evaluate_mi_ensemble.py --at true --mode mi --device cpu --log-level WARNING 2>&1)

WORST_ASR=$(echo "$OUTPUT" | grep "ASR" | sed 's/[^0-9.]//g' | sort -rn | head -1)
CLEAN_ACC=$(echo "$OUTPUT" | grep "Acc (clean)" | sed 's/[^0-9.]//g' | head -1)
AVG_ASR=$(echo "$OUTPUT" | grep "ASR" | sed 's/[^0-9.]//g' | awk '{s+=$1; n++} END {printf "%.2f", s/n}')

echo "WORST_ASR=$WORST_ASR"
echo "AVG_ASR=$AVG_ASR"
echo "CLEAN_ACC=$CLEAN_ACC"

# Guard: clean acc must be >= 83.64
if (( $(echo "$CLEAN_ACC < 83.64" | bc -l) )); then
    echo "GUARD_FAIL: Clean Acc $CLEAN_ACC < 83.64"
    exit 1
fi

echo "METRIC=$WORST_ASR"
