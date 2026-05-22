#!/usr/bin/env bash
# IIOT-2021 / run_full_eval.sh
# Single-file orchestrator chạy toàn bộ kịch bản đánh giá.
#
#   AIDER components             : B0 (no defense), B5 (AT only),
#                                   B6 (AT+FS single),
#                                   B7 (AIDER ens AT),
#                                   B8 (AT+FS ensemble),
#                                   B9 (baseline ensemble, no AT)
#   Comparison baselines (SOTA)  : B2 (PGD-AT), B4 (Defensive Distillation)
#
# Pre-req: baseline models + adv samples (`baseline/models/`,
# `adv_samples/adv_eval/`) phải có sẵn. Script tự train AIDER AT,
# PGD-AT, DD nếu thiếu.
#
# Output:
#   $REPORT_DIR/B_all.{md,txt}                consolidated 8 defense × scenarios
#   $REPORT_DIR/B_<label>.{md,txt}            per-defense
#
# AT+FS config: --bit-depth 2 cho mọi model, override ET=4 để tránh cw collapse.
#
# Usage:
#   ./run_full_eval.sh                                    # default ours track
#   DEVICE=cuda ./run_full_eval.sh
#   ENS_CFG="" ./run_full_eval.sh                          # equal weights

set -euo pipefail

DEVICE="${DEVICE:-cpu}"
REPORT_DIR="${REPORT_DIR:-report/ablation}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"
TRAIN_PATH="${TRAIN_PATH:-datasets/train_topup_t1000_e50.csv}"
ENS_CFG="${ENS_CFG:-runs/ours/config}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source ../.venv/bin/activate
mkdir -p "$REPORT_DIR"

# Single-CLI dispatch — dataset registered as "iiot" in /datasets.yaml
AIDER="aider iiot"

# ── Scenario table ─────────────────────────────────────────────────────────
# Mỗi dòng = 1 kịch bản, 3 cột phân tách bởi `|`:
#   <label> | <mô tả 1-dòng> | <args truyền cho eval-scenarios>
# Sửa/thêm/bớt kịch bản chỉ cần edit mảng này.
STEPS=(
  "B0_baseline     | Baseline single-model, clean test (no defense, no attack)            | --scenarios S1"
  "B5_aider_at     | AIDER multi-attack AT, single-model, under attack                    | --scenarios S2 --defense at"
  "B6_aider_at_fs  | AIDER AT + Feature Squeezing (bit-depth=2, et=4), single, under atk  | --scenarios S2 --defense at --preproc-defense feature_squeezing --bit-depth 2 --fs-config et=4,mlp=4"
  "B9_baseline_ens | Baseline ensemble (no AT), clean — static + mi4                      | --scenarios E1 --strategy static mi4 mi5"
  "B7_aider_ens    | AIDER AT ensemble, under attack — static + mi4                       | --scenarios E2 --strategy static mi4 mi5"
  "B8_aider_ens_fs | AIDER AT + FS ensemble (bit-depth=2, et=4), under attack             | --scenarios E2 --strategy static mi4 mi5 --defense at --preproc-defense feature_squeezing --bit-depth 2 --fs-config et=4"
  "B10_base_ens_fs | Baseline ensemble (no AT) + FS (bit-depth=2, et=4), under attack     | --scenarios E1 --strategy static mi4 mi5 --preproc-defense feature_squeezing --bit-depth 2 --fs-config et=4,mlp=4"
  "B2_pgd_at       | SOTA baseline: Madry PGD-AT, single, under attack                    | --scenarios S2 --defense pgd_at --defense-model-dir defense/pgd_at"
  "B4_distill      | SOTA baseline: Defensive Distillation (MLP), single, under attack    | --scenarios S2 --defense distill --defense-model-dir defense/distill --target mlp"
)

ALL_BASE="$REPORT_DIR/B_all"
rm -f "$ALL_BASE.json" "$ALL_BASE.md" "$ALL_BASE.txt"

echo "═══════════════════════════════════════════════════════════════════════════"
echo " IIOT-2021 — Full ablation eval"
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

# ── Pre-req: train missing artifacts ───────────────────────────────────────
# Retrain when sentinel is missing OR older than the current train CSV
# (avoids feature-count drift after dataset regeneration).
needs_retrain() {
    [ -f "$1" ] || return 0
    [ "$1" -ot "$TRAIN_PATH" ] && return 0
    return 1
}
needs_retrain "defense/at/mlp_at.pth" && {
    echo "[setup] Training AIDER AT (all 6 models) ..."
    $AIDER train-at -m all --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
} || true
needs_retrain "defense/pgd_at/mlp_pgd_at.pth" && {
    echo "[setup] Training PGD-AT (B2) ..."
    $AIDER train-pgd -m all --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
} || true
needs_retrain "defense/distill/mlp_distill.pth" && {
    echo "[setup] Training Defensive Distillation MLP (B4) ..."
    $AIDER train-distill --train-path "$TRAIN_PATH" --device "$DEVICE" --log-level "$LOG_LEVEL"
} || true

# ── Tuned ensemble dir flag (chỉ thêm nếu thư mục tồn tại) ─────────────────
EXTRA_ENS=""
[ -d "$ENS_CFG" ] && EXTRA_ENS="--ensemble-config-dir $ENS_CFG"

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
