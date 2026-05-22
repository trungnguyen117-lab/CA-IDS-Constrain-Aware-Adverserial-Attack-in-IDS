#!/usr/bin/env bash
# MODBUS-2023 / run_full_eval.sh
# Single-file orchestrator chạy toàn bộ kịch bản đánh giá (mirror IIOT-2021).
#
#   AIDER components             : B0 (no defense), B5 (AT only),
#                                   B6 (AT+FS single), B7 (AIDER ens AT),
#                                   B8 (AT+FS ensemble), B9 (baseline ensemble, no AT)
#   Comparison baselines (SOTA)  : B2 (PGD-AT), B4 (Defensive Distillation, DNN only)
#
# Pre-req: baseline models + adv samples (`baseline/models/`,
# `adv_samples/adv_eval/`) phải có sẵn. Script tự train AIDER AT / PGD-AT / DD
# nếu thiếu.
#
# Output:
#   $REPORT_DIR/B_all.{md,txt}                consolidated 8 defense × scenarios
#   $REPORT_DIR/B_<label>.{md,txt}            per-defense
#
# AT+FS config: --bit-depth 2 (single) / 6 (ensemble) cho mọi model, override
# ET=4 để tránh cw collapse.
#
# Usage:
#   ./run_full_eval.sh                                    # default
#   DEVICE=cuda ./run_full_eval.sh
#   ENS_CFG=runs/ours/config ./run_full_eval.sh           # khi tune xong

set -euo pipefail

DEVICE="${DEVICE:-cpu}"
REPORT_DIR="${REPORT_DIR:-report/ablation}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"
TRAIN_PATH="${TRAIN_PATH:-datasets/train_shap_58_600.csv}"
ENS_CFG="${ENS_CFG:-runs/ours/config}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source ../.venv/bin/activate
mkdir -p "$REPORT_DIR"

# Single-CLI dispatch — dataset registered as "modbus" in /datasets.yaml
AIDER="aider modbus"

# ── Scenario table ─────────────────────────────────────────────────────────
# Mỗi dòng = 1 kịch bản, 3 cột phân tách bởi `|`:
#   <label> | <mô tả 1-dòng> | <args truyền cho eval-scenarios>
# Sửa/thêm/bớt kịch bản chỉ cần edit mảng này.
STEPS=(
  "B0_baseline     | Baseline single-model, clean test (no defense, no attack)            | --scenarios S1"
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
echo " MODBUS-2023 — Full ablation eval"
echo "   Device     : $DEVICE"
echo "   Report dir : $REPORT_DIR"
echo "   Train CSV  : $TRAIN_PATH"
echo "   Ensemble   : ${ENS_CFG:-<equal weights>}"
echo "═══════════════════════════════════════════════════════════════════════════"

# ── Plan summary: in trước để user thấy sẽ chạy gì ─────────────────────────
echo ""
echo "=== Plan: ${#STEPS[@]} scenarios sẽ chạy ==="
for step in "${STEPS[@]}"; do
    label="$(echo "${step%%|*}" | xargs)"
    rest="${step#*|}"
    desc="$(echo "${rest%%|*}" | xargs)"
    printf "  %-16s %s\n" "$label" "$desc"
done
echo ""

# ── Pre-req: train missing artifacts (sentinel = file đầu được sinh) ───────
[ -f "defense/at/dnn_at.pth" ] || {
    echo "[setup] Training AIDER AT (all 6 models) ..."
    $AIDER train-at -m all --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
}
[ -f "defense/pgd_at/dnn_pgd_at.pth" ] || {
    echo "[setup] Training PGD-AT (B2) ..."
    $AIDER train-pgd -m all --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
}
[ -f "defense/distill/dnn_distill.pth" ] || {
    echo "[setup] Training Defensive Distillation DNN (B4) ..."
    $AIDER train-distill --target dnn --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
}

# ── Tuned ensemble dir (chỉ thêm flag nếu thư mục có) ──────────────────────
EXTRA_ENS=""
[ -n "$ENS_CFG" ] && [ -d "$ENS_CFG" ] && EXTRA_ENS="--ensemble-config-dir $ENS_CFG"

# ── Chạy từng kịch bản với banner đầy đủ ───────────────────────────────────
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

# Cleanup B_all.json sau khi mọi defense đã append xong.
rm -f "$ALL_BASE.json"

echo ""
echo "=== Done. $REPORT_DIR/ ==="
ls -la "$REPORT_DIR/"
