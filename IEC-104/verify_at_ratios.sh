#!/bin/bash
# Autoresearch verify: retrain all AT models + evaluate worst-case individual ASR
# Guard: each model's clean acc must stay within -3pp of baseline
cd /Users/trungnguye2n.dev/foami-plus && source .venv/bin/activate && cd IEC-104

# Step 1: Retrain all models with current AT ratios
echo "=== Retraining AT models ==="
python cli/pipeline/3_adv_train.py --model all --device cpu 2>&1 | tail -5

# Step 2: Evaluate all individual models
echo "=== Evaluating ==="
OUTPUT=$(python cli/pipeline/2_evaluate.py --target all --at true --device cpu --log-level WARNING 2>&1)

# Parse table: columns are cat | rf | lstm | resdnn
# Acc (clean) row: │ Acc (clean) │ 84.57% │ 85.36% │ 72.24% │ 68.69% │
ACC_LINE=$(echo "$OUTPUT" | grep "Acc (clean)")
CAT_ACC=$(echo "$ACC_LINE" | awk -F'│' '{gsub(/[^0-9.]/,"",$3); print $3}')
RF_ACC=$(echo "$ACC_LINE" | awk -F'│' '{gsub(/[^0-9.]/,"",$4); print $4}')
LSTM_ACC=$(echo "$ACC_LINE" | awk -F'│' '{gsub(/[^0-9.]/,"",$5); print $5}')
RESDNN_ACC=$(echo "$ACC_LINE" | awk -F'│' '{gsub(/[^0-9.]/,"",$6); print $6}')

# All ASR values (excluding header/separator lines)
WORST_ASR=$(echo "$OUTPUT" | grep "│.*ASR" | awk -F'│' '{for(i=3;i<=6;i++){gsub(/[^0-9.]/,"",$i); if($i+0>0) print $i}}' | sort -rn | head -1)
AVG_ASR=$(echo "$OUTPUT" | grep "│.*ASR" | awk -F'│' '{for(i=3;i<=6;i++){gsub(/[^0-9.]/,"",$i); if($i+0>0) print $i}}' | awk '{s+=$1; n++} END {printf "%.2f", s/n}')

echo "CAT_ACC=$CAT_ACC"
echo "RF_ACC=$RF_ACC"
echo "LSTM_ACC=$LSTM_ACC"
echo "RESDNN_ACC=$RESDNN_ACC"
echo "WORST_ASR=$WORST_ASR"
echo "AVG_ASR=$AVG_ASR"

# Guard: clean acc per model (baseline - 3pp)
FAIL=0
if (( $(echo "$CAT_ACC < 81.57" | bc -l) )); then
    echo "GUARD_FAIL: cat Acc $CAT_ACC < 81.57"
    FAIL=1
fi
if (( $(echo "$RF_ACC < 82.36" | bc -l) )); then
    echo "GUARD_FAIL: rf Acc $RF_ACC < 82.36"
    FAIL=1
fi
if (( $(echo "$LSTM_ACC < 69.24" | bc -l) )); then
    echo "GUARD_FAIL: lstm Acc $LSTM_ACC < 69.24"
    FAIL=1
fi
if (( $(echo "$RESDNN_ACC < 65.69" | bc -l) )); then
    echo "GUARD_FAIL: resdnn Acc $RESDNN_ACC < 65.69"
    FAIL=1
fi

if [ $FAIL -eq 1 ]; then
    exit 1
fi

echo "METRIC=$WORST_ASR"
