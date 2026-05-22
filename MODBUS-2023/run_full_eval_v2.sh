#!/usr/bin/env bash
# MODBUS-2023 / run_full_eval_v2.sh
# Variant của run_full_eval.sh chạy trên artifact _repro (config_v2.yaml).
# Mọi output ghi vào *_v2/ (baseline_v2, defense_v2, adv_samples_v2, report_v2)
# → KHÔNG đè canonical 58-feat.
#
# Pre-req (chạy tay trước nếu chưa có):
#   aider modbus_v2 train   -m all --train-path "$TRAIN_PATH" --device "$DEVICE"
#   aider modbus_v2 gen-adv -t all -a all -s train --device "$DEVICE"
#   aider modbus_v2 gen-adv -t all -a all -s test  --device "$DEVICE"
#
# Output:
#   $REPORT_DIR/B_all.{md,txt}        consolidated 8 defense × scenarios
#   $REPORT_DIR/B_<label>.{md,txt}    per-defense
#
# Usage:
#   ./run_full_eval_v2.sh
#   DEVICE=cuda ./run_full_eval_v2.sh
#   ENS_CFG=runs_v2/ours/config ./run_full_eval_v2.sh

set -euo pipefail

DEVICE="${DEVICE:-cpu}"
REPORT_DIR="${REPORT_DIR:-report_v2/ablation}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"
TRAIN_PATH="${TRAIN_PATH:-datasets/modbus_train_merged_t1400_e200_repro.csv}"
ENS_CFG="${ENS_CFG:-runs/ours/config}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source ../.venv/bin/activate
mkdir -p "$REPORT_DIR"

# Registry shortname for repro variant (xem datasets.yaml).
AIDER="aider modbus_v2"

# ── Scenario table ─────────────────────────────────────────────────────────
STEPS=(
  "B0_baseline     | Baseline single-model, clean test (no defense, under attack)            | --scenarios S1"
  "B5_aider_at     | AIDER multi-attack AT, single-model, under attack                    | --scenarios S2 --defense at"
  "B6_aider_at_fs  | AIDER AT + Feature Squeezing (bit-depth=2, et=4), single, under atk  | --scenarios S2 --defense at --preproc-defense feature_squeezing --bit-depth 2 --fs-config et=4"
  "B9_baseline_ens | Baseline ensemble (no AT), clean — static + mi4                      | --scenarios E1 --strategy static mi4 mi5"
  "B7_aider_ens    | AIDER AT ensemble, under attack — static + mi4                       | --scenarios E2 --strategy static mi4 mi5"
  "B8_aider_ens_fs | AIDER AT + FS ensemble (bit-depth=6, et=4), under attack             | --scenarios E2 --strategy static mi4 mi5 --defense at --preproc-defense feature_squeezing --bit-depth 2 --fs-config et=4"
  "B10_base_ens_fs | Baseline ensemble (no AT) + FS (bit-depth=2, et=4), under attack     | --scenarios E1 --strategy static mi4 mi5 --preproc-defense feature_squeezing --bit-depth 2 --fs-config et=4"
  "B2_pgd_at       | SOTA baseline: Madry PGD-AT, single, under attack                    | --scenarios S2 --defense pgd_at"
  "B4_distill      | SOTA baseline: Defensive Distillation (DNN), single, under attack    | --scenarios S2 --defense distill --target dnn"
)

ALL_BASE="$REPORT_DIR/B_all"
rm -f "$ALL_BASE.json" "$ALL_BASE.md" "$ALL_BASE.txt"

echo "═══════════════════════════════════════════════════════════════════════════"
echo " MODBUS-2023 (v2 / _repro) — Full ablation eval"
echo "   Device     : $DEVICE"
echo "   Report dir : $REPORT_DIR"
echo "   Train CSV  : $TRAIN_PATH"
echo "   Ensemble   : ${ENS_CFG:-<equal weights>}"
echo "═══════════════════════════════════════════════════════════════════════════"

echo ""
echo "=== Plan: ${#STEPS[@]} scenarios sẽ chạy ==="
for step in "${STEPS[@]}"; do
    label="$(echo "${step%%|*}" | xargs)"
    rest="${step#*|}"
    desc="$(echo "${rest%%|*}" | xargs)"
    printf "  %-16s %s\n" "$label" "$desc"
done
echo ""

# ── Pre-req: train missing defense artifacts (sentinel = file đầu được sinh) ─
[ -f "defense_v2/at/dnn_at.pth" ] || {
    echo "[setup] Training AIDER AT (all 6 models) ..."
    $AIDER train-at -m all --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
}
[ -f "defense_v2/pgd_at/dnn_pgd_at.pth" ] || {
    echo "[setup] Training PGD-AT (B2) ..."
    $AIDER train-pgd -m all --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
}
[ -f "defense_v2/distill/dnn_distill.pth" ] || {
    echo "[setup] Training Defensive Distillation DNN (B4) ..."
    $AIDER train-distill --target dnn --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
}

EXTRA_ENS=""
[ -n "$ENS_CFG" ] && [ -d "$ENS_CFG" ] && EXTRA_ENS="--ensemble-config-dir $ENS_CFG"

for step in "${STEPS[@]}"; do
    label="$(echo "${step%%|*}" | xargs)"
    rest="${step#*|}"
    desc="$(echo "${rest%%|*}" | xargs)"
    args="$(echo "${rest#*|}" | xargs)"

    echo ""
    echo "────────────────────────────────────────────────────────────────────────"
    echo "[$label] $desc"
    echo "  \$ $AIDER eval-scenarios $args $EXTRA_ENS \\"
    echo "      --device $DEVICE --log-level $LOG_LEVEL \\"
    echo "      --export $REPORT_DIR/B_${label} --export-into $ALL_BASE \\"
    echo "      --export-label $label --export-format md,txt"
    echo "────────────────────────────────────────────────────────────────────────"

    # shellcheck disable=SC2086
    $AIDER eval-scenarios $args $EXTRA_ENS \
        --device "$DEVICE" --log-level "$LOG_LEVEL" \
        --export "$REPORT_DIR/B_${label}" \
        --export-into "$ALL_BASE" \
        --export-label "$label" \
        --export-format md,txt
done

rm -f "$ALL_BASE.json"

echo ""
echo "=== Done. $REPORT_DIR/ ==="
ls -la "$REPORT_DIR/"
