"""Full adversarial robustness benchmark.

Data split — NO leakage between training and evaluation:
  adv_eval/      ← adversarial examples generated from TEST SET (test_shap_66.csv)
                    used ONLY for benchmarking (Steps 2, 3, 7, 8)
  adv_training/  ← adversarial examples generated from TVAE TRAIN DATA (train_at.csv)
                    used ONLY for adversarial training defense (Steps 4, 5)

Pipeline steps:
  1a. Generate adv from TVAE train data → adv_training/  (for AT defense)
  1b. Generate adv from TEST SET        → adv_eval/      (for evaluation, no leakage)
  2.  Evaluate single models BEFORE AT  (on adv_eval/)
  3.  Evaluate ensemble / MI BEFORE AT  (per-component adv_eval/ inputs)
  4.  Merge original train + adv_training/ examples (no TVAE synthetic)
  5.  Retrain with adversarial training (offline AT, all single models)
  6.  Stage AT model files with standard filenames
  7.  Evaluate single models AFTER AT   (same adv_eval/, AT-hardened models)
  8.  Evaluate ensemble / MI AFTER AT   (same adv_eval/, AT-hardened components)
  9.  Print before/after comparison table

Attack compatibility for GENERATION:
  - Tree (xgb, cat, rf) : zoo, hsja  (black-box only)
  - DL  (lstm, resdnn)  : all attacks (zoo, deepfool, fgsm, cw, pgd, hsja, jsma)

Ensemble / MI evaluation:
  - NOT limited to black-box attacks for evaluation
  - Evaluated against ALL adv CSVs in adv_eval/ from individual models
  - Prediction = forward pass through each component model → combine weights
  - No gradient needed, so any adv X can be used

Usage:
    # Full benchmark, all targets, all compatible attacks
    python run_full_benchmark.py

    # Specific attacks only
    python run_full_benchmark.py --attacks pgd fgsm zoo hsja

    # Skip both generation steps if adv CSVs already exist
    python run_full_benchmark.py --skip-gen

    # Skip only eval-adv generation (adv_eval/ already populated)
    python run_full_benchmark.py --skip-eval-gen

    # Skip adversarial training (reuse existing AT models)
    python run_full_benchmark.py --skip-at

    # GPU acceleration
    python run_full_benchmark.py --device cuda

    # Limit samples per class during adv generation from TVAE data (default: 1000)
    python run_full_benchmark.py --samples-per-class 500
"""

import os
import sys
import shutil
import argparse
import subprocess

# ── Path bootstrap ──────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(_HERE)
sys.path.insert(0, _FOAMI)

from utils.paths import (
    setup_paths, ROOT_DIR, MODELS_DIR,
    AT_DIR, AT_MERGED_CSV,
    ADV_EVAL_DIR,
    TRAIN_ORIG_CSV, TEST_CSV,
)
setup_paths()

from utils.logging import setup_logging, get_logger
from utils.constants import (
    SINGLE_TARGETS, ALL_ATTACKS, BLACKBOX_ATTACKS,
)

logger = get_logger(__name__)

# ── Attack compatibility for GENERATION ────────────────────────────────────────
# (generation only — evaluation of ensemble/mi uses all available adv CSVs)
_GEN_COMPAT: dict[str, set] = {
    'xgb':    set(BLACKBOX_ATTACKS),
    'cat':    set(BLACKBOX_ATTACKS),
    'rf':     set(BLACKBOX_ATTACKS),
    'lstm':   set(ALL_ATTACKS),
    'resdnn': set(ALL_ATTACKS),
}

# ── AT model filenames ──────────────────────────────────────────────────────────
# retrain_at.py saves: framework_{model}_TVAE_at.{ext}
# load_wrapper expects: framework_{model}_TVAE.{ext}
_AT_NAMES = {
    'xgb':    ('framework_xgb_TVAE_at.pkl',    'framework_xgb_TVAE.pkl'),
    'cat':    ('framework_cat_TVAE_at.pkl',     'framework_cat_TVAE.pkl'),
    'rf':     ('framework_rf_TVAE_at.pkl',      'framework_rf_TVAE.pkl'),
    'lstm':   ('framework_lstm_TVAE_at.pth',    'framework_lstm_TVAE.pth'),
    'resdnn': ('framework_resdnn_TVAE_at.pth',  'framework_resdnn_TVAE.pth'),
}

# ── Script paths ────────────────────────────────────────────────────────────────
_GEN_TVAE    = os.path.join(_HERE, '3_adv_training', 'generate_adv_from_tvae.py')
_MERGE       = os.path.join(_HERE, '3_adv_training', 'merge_adv_data.py')
_RETRAIN_AT  = os.path.join(_HERE, '3_adv_training', 'retrain_at.py')
_EVALUATE    = os.path.join(_HERE, '2_evaluate',     'individual_evaluate.py')
_EVAL_ENS_MI = os.path.join(_HERE, '2_evaluate',     'evaluate_ensemble_mi.py')


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], step: str, fail_fast: bool) -> bool:
    logger.info(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        logger.warning(f"  [!] FAILED (exit {rc}): {step}")
        if fail_fast:
            raise SystemExit(f"Stopped at: {step}")
        return False
    return True


def _adv_dir(target: str) -> str:
    return os.path.join(AT_DIR, target)


def _adv_csv(target: str, attack: str) -> str:
    return os.path.join(_adv_dir(target), f"{target}_{attack}_adv.csv")



# ── Step 1a: Generate adv from TVAE train data (for AT defense) ─────────────────

def step_generate_at(attacks: list[str], samples_per_class: int,
                     device: str, fail_fast: bool) -> list[tuple]:
    """Generate adversarial examples from TVAE-augmented training data.

    Output: adv_training/{target}/{target}_{attack}_adv.csv
    Used ONLY for adversarial training (merge + retrain). NOT for evaluation.
    """
    pairs = [
        (t, a) for t in SINGLE_TARGETS for a in attacks
        if a in _GEN_COMPAT.get(t, set())
    ]
    logger.info(f"\n{'='*60}")
    logger.info(f"  STEP 1a — Generate adv from TVAE train data ({len(pairs)} combos)")
    logger.info(f"  Source: train_at.csv  →  adv_training/  (for AT defense only)")
    logger.info(f"{'='*60}")

    ok, skipped, failed = 0, 0, []
    for i, (target, attack) in enumerate(pairs, 1):
        out_csv = _adv_csv(target, attack)
        if os.path.exists(out_csv):
            logger.info(f"  [{i}/{len(pairs)}] SKIP (exists): {target} × {attack}")
            skipped += 1
            continue

        logger.info(f"  [{i}/{len(pairs)}] {target} × {attack}")
        cmd = [
            sys.executable, _GEN_TVAE,
            '--target', target, '--attack', attack,
            '--device', device,
            '--output-dir', _adv_dir(target),
        ]
        if samples_per_class > 0:
            cmd += ['--samples-per-class', str(samples_per_class)]

        if _run(cmd, f"gen-at {target}/{attack}", fail_fast):
            ok += 1
        else:
            failed.append((target, attack))

    logger.info(f"  [+] Generated: {ok}, skipped: {skipped}, failed: {len(failed)}")
    return failed


# ── Step 1b: Generate adv from TEST SET (for evaluation) ────────────────────────

def step_generate_eval(attacks: list[str], device: str, fail_fast: bool) -> list[tuple]:
    """Generate adversarial examples from the TEST SET for benchmark evaluation.

    Output: adv_eval/{target}/{target}_{attack}_adv.csv
    Used ONLY for evaluation (Steps 2, 3, 7, 8). NEVER merged into training data.

    No --samples-per-class here: we want to evaluate on the full test set
    to get unbiased accuracy/ASR estimates.
    """
    pairs = [
        (t, a) for t in SINGLE_TARGETS for a in attacks
        if a in _GEN_COMPAT.get(t, set())
    ]
    logger.info(f"\n{'='*60}")
    logger.info(f"  STEP 1b — Generate adv from TEST SET ({len(pairs)} combos)")
    logger.info(f"  Source: test_shap_66.csv  →  adv_eval/  (for evaluation only)")
    logger.info(f"  [NO leakage: these examples are NEVER used for AT training]")
    logger.info(f"{'='*60}")

    ok, skipped, failed = 0, 0, []
    for i, (target, attack) in enumerate(pairs, 1):
        out_dir = os.path.join(ADV_EVAL_DIR, target)
        out_csv = os.path.join(out_dir, f"{target}_{attack}_adv.csv")
        if os.path.exists(out_csv):
            logger.info(f"  [{i}/{len(pairs)}] SKIP (exists): {target} × {attack}")
            skipped += 1
            continue

        logger.info(f"  [{i}/{len(pairs)}] {target} × {attack}")
        cmd = [
            sys.executable, _GEN_TVAE,
            '--target', target, '--attack', attack,
            '--data-in', TEST_CSV,         # ← TEST SET, not TVAE train data
            '--output-dir', out_dir,
            '--device', device,
        ]

        if _run(cmd, f"gen-eval {target}/{attack}", fail_fast):
            ok += 1
        else:
            failed.append((target, attack))

    logger.info(f"  [+] Generated: {ok}, skipped: {skipped}, failed: {len(failed)}")
    return failed


# ── Step 2/7: Evaluate single models ───────────────────────────────────────────

def step_evaluate_single(attacks: list[str], models_dir: str, adv_dir: str,
                         label: str, output_csv: str, fail_fast: bool):
    """Evaluate each single model against adv CSVs from adv_dir (adv_eval/)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  EVALUATE single models — {label}")
    logger.info(f"  adv_dir: {adv_dir}")
    logger.info(f"{'='*60}")

    # All compatible attacks across single targets
    all_attacks = sorted({
        a for t in SINGLE_TARGETS for a in attacks
        if a in _GEN_COMPAT.get(t, set())
    })

    cmd = [
        sys.executable, _EVALUATE,
        '--target',    *SINGLE_TARGETS,
        '--attack',    *all_attacks,
        '--plain-in',  TEST_CSV,
        '--adv-dir',   adv_dir,
        '--models-dir', models_dir,
        '--output-csv', output_csv,
        '--per-class',
    ]
    _run(cmd, f"evaluate single ({label})", fail_fast)


# ── Step 3/8: Evaluate ensemble / MI ───────────────────────────────────────────

def step_evaluate_ensemble_mi(attacks: list[str], models_dir: str, adv_dir: str,
                               label: str, output_csv: str, fail_fast: bool):
    """Evaluate ensemble and MI with per-component adversarial inputs from adv_dir.

    Each component model receives the adv X (from adv_eval/) generated against
    it specifically. GBT models without their own adv CSV fall back to DL adv.
    Pattern mirrors apelid/classifier_parallel_ens_eval.py.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  EVALUATE ensemble / MI — {label}")
    logger.info(f"  Per-component adv X (each model gets its own adversarial input)")
    logger.info(f"  adv_dir: {adv_dir}")
    logger.info(f"{'='*60}")

    cmd = [
        sys.executable, _EVAL_ENS_MI,
        '--attack',    *attacks,
        '--plain-in',  TEST_CSV,
        '--adv-dir',   adv_dir,
        '--models-dir', models_dir,
        '--output-csv', output_csv,
    ]
    _run(cmd, f"evaluate ensemble/mi ({label})", fail_fast)


# ── Step 4: Merge ───────────────────────────────────────────────────────────────

def step_merge(attacks: list[str], fail_fast: bool) -> bool:
    logger.info(f"\n{'='*60}")
    logger.info(f"  STEP 4 — Merge original train + adversarial examples")
    logger.info(f"  base-csv: {TRAIN_ORIG_CSV}  (NO TVAE synthetic)")
    logger.info(f"{'='*60}")

    compatible_attacks = sorted({
        a for t in SINGLE_TARGETS for a in attacks
        if a in _GEN_COMPAT.get(t, set())
    })

    cmd = [
        sys.executable, _MERGE,
        '--base-csv', TRAIN_ORIG_CSV,
        '--adv-dir',  AT_DIR,
        '--targets',  *SINGLE_TARGETS,
        '--attacks',  *compatible_attacks,
        '--out-csv',  AT_MERGED_CSV,
    ]
    return _run(cmd, "merge", fail_fast)


# ── Step 5: Retrain AT ──────────────────────────────────────────────────────────

def step_retrain(device: str, fail_fast: bool) -> bool:
    logger.info(f"\n{'='*60}")
    logger.info(f"  STEP 5 — Adversarial training (offline AT, all single models)")
    logger.info(f"{'='*60}")

    cmd = [
        sys.executable, _RETRAIN_AT,
        '--model',     *SINGLE_TARGETS,
        '--train-csv', AT_MERGED_CSV,
        '--test-csv',  TEST_CSV,
        '--models-dir', MODELS_DIR,
        '--device',    device,
    ]
    return _run(cmd, "retrain_at", fail_fast)


# ── Step 6: Stage AT models ─────────────────────────────────────────────────────

def step_stage_at_models(models_dir_at: str):
    """Copy AT models to staging dir with standard filenames for load_wrapper()."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  STEP 6 — Stage AT models → {models_dir_at}")
    logger.info(f"{'='*60}")

    os.makedirs(models_dir_at, exist_ok=True)
    for t in SINGLE_TARGETS:
        src_name, dst_name = _AT_NAMES[t]
        src = os.path.join(MODELS_DIR, src_name)
        dst = os.path.join(models_dir_at, dst_name)
        if not os.path.exists(src):
            logger.warning(f"  [!] Not found, skip: {src_name}")
            continue
        shutil.copy2(src, dst)
        logger.info(f"  {src_name}  →  {dst_name}")


# ── Step 9: Comparison table ────────────────────────────────────────────────────

def print_comparison(csv_before: str, csv_after: str):
    try:
        import pandas as pd
        from prettytable import PrettyTable
    except ImportError:
        return

    if not os.path.exists(csv_before) or not os.path.exists(csv_after):
        return

    df_b = pd.read_csv(csv_before).set_index(['target', 'attack'])
    df_a = pd.read_csv(csv_after).set_index(['target', 'attack'])
    common = df_b.index.intersection(df_a.index)
    if common.empty:
        return

    logger.info(f"\n{'='*80}")
    logger.info("  BEFORE vs AFTER Adversarial Training")
    logger.info(f"{'='*80}")

    tbl = PrettyTable([
        'Target', 'Attack',
        'Plain (B)', 'Adv Acc (B)', 'ASR (B)',
        'Plain (A)', 'Adv Acc (A)', 'ASR (A)',
        'ΔAdv Acc',  'ΔASR',
    ])
    tbl.align = 'r'
    tbl.align['Target'] = 'l'
    tbl.align['Attack'] = 'l'

    def pct(v):  return f"{v*100:.2f}%"
    def dpct(v): return f"{v*100:+.2f}%"

    for (target, attack) in sorted(common):
        b = df_b.loc[(target, attack)]
        a = df_a.loc[(target, attack)]
        tbl.add_row([
            target, attack,
            pct(b['plain_acc']), pct(b['adv_acc']), pct(b['asr']),
            pct(a['plain_acc']), pct(a['adv_acc']), pct(a['asr']),
            dpct(a['adv_acc'] - b['adv_acc']),
            dpct(a['asr']     - b['asr']),
        ])

    logger.info("\n" + tbl.get_string())


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full adversarial robustness benchmark — before & after AT"
    )
    parser.add_argument('--attacks', nargs='+', default=list(ALL_ATTACKS),
                        choices=list(ALL_ATTACKS),
                        help="Attacks to use (default: all, compatibility enforced per model)")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--samples-per-class', type=int, default=1000,
                        help="Samples per class for adv generation (default: 1000)")
    parser.add_argument('--skip-gen', action='store_true',
                        help="Skip Steps 1a+1b — reuse existing adv CSVs in both "
                             "adv_training/ and adv_eval/")
    parser.add_argument('--skip-eval-gen', action='store_true',
                        help="Skip Step 1b only — reuse existing adv_eval/ CSVs "
                             "(still generates adv_training/ if needed)")
    parser.add_argument('--skip-at', action='store_true',
                        help="Skip Steps 4-6 — reuse existing AT models in models_at/")
    parser.add_argument('--fail-fast', action='store_true',
                        help="Stop on first failure")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    models_dir_at = os.path.join(ROOT_DIR, 'models_at')
    report_dir    = os.path.join(_FOAMI, 'report')
    os.makedirs(report_dir, exist_ok=True)

    csv_before_single   = os.path.join(report_dir, 'results_before_at_single.csv')
    csv_before_ensemble = os.path.join(report_dir, 'results_before_at_ensemble.csv')
    csv_after_single    = os.path.join(report_dir, 'results_after_at_single.csv')
    csv_after_ensemble  = os.path.join(report_dir, 'results_after_at_ensemble.csv')

    logger.info("=" * 60)
    logger.info("  FULL ADVERSARIAL ROBUSTNESS BENCHMARK")
    logger.info(f"  Attacks : {args.attacks}")
    logger.info(f"  Device  : {args.device}")
    logger.info("  Data split (no leakage):")
    logger.info("    adv_eval/      ← from TEST SET  → evaluation only")
    logger.info("    adv_training/  ← from TVAE data → AT defense only")
    logger.info("=" * 60)

    # ── Step 1a: Generate adv from TVAE train data → adv_training/ (for AT) ──
    if not args.skip_gen:
        step_generate_at(
            attacks=args.attacks,
            samples_per_class=args.samples_per_class,
            device=args.device,
            fail_fast=args.fail_fast,
        )
    else:
        logger.info("\n[SKIP] Step 1a — adv_training/ generation (--skip-gen)")

    # ── Step 1b: Generate adv from TEST SET → adv_eval/ (for evaluation) ─────
    if not args.skip_gen and not args.skip_eval_gen:
        step_generate_eval(
            attacks=args.attacks,
            device=args.device,
            fail_fast=args.fail_fast,
        )
    else:
        logger.info("\n[SKIP] Step 1b — adv_eval/ generation (--skip-gen / --skip-eval-gen)")

    # ── Step 2: Evaluate single models BEFORE AT (on adv_eval/) ──────────────
    step_evaluate_single(
        attacks=args.attacks,
        models_dir=MODELS_DIR,
        adv_dir=ADV_EVAL_DIR,
        label="BEFORE AT",
        output_csv=csv_before_single,
        fail_fast=args.fail_fast,
    )

    # ── Step 3: Evaluate ensemble/MI BEFORE AT (on adv_eval/) ────────────────
    # Per-component: each model gets its own adv X from adv_eval/
    # No gradient needed for prediction → any adv X is valid
    step_evaluate_ensemble_mi(
        attacks=args.attacks,
        models_dir=MODELS_DIR,
        adv_dir=ADV_EVAL_DIR,
        label="BEFORE AT",
        output_csv=csv_before_ensemble,
        fail_fast=args.fail_fast,
    )

    if not args.skip_at:
        # ── Step 4: Merge (uses adv_training/ — from TVAE data, no test leakage)
        step_merge(attacks=args.attacks, fail_fast=args.fail_fast)

        # ── Step 5: Retrain AT ────────────────────────────────────────────────
        step_retrain(device=args.device, fail_fast=args.fail_fast)

        # ── Step 6: Stage AT models ───────────────────────────────────────────
        step_stage_at_models(models_dir_at)

    else:
        logger.info("\n[SKIP] Steps 4-6 — adversarial training (--skip-at)")

    # ── Step 7: Evaluate single models AFTER AT (same adv_eval/, AT models) ──
    step_evaluate_single(
        attacks=args.attacks,
        models_dir=models_dir_at,
        adv_dir=ADV_EVAL_DIR,
        label="AFTER AT",
        output_csv=csv_after_single,
        fail_fast=args.fail_fast,
    )

    # ── Step 8: Evaluate ensemble/MI AFTER AT (same adv_eval/, AT components) ─
    # Same adv X as Step 3, but component models are now AT-hardened
    step_evaluate_ensemble_mi(
        attacks=args.attacks,
        models_dir=models_dir_at,
        adv_dir=ADV_EVAL_DIR,
        label="AFTER AT",
        output_csv=csv_after_ensemble,
        fail_fast=args.fail_fast,
    )

    # ── Step 9: Comparison tables ─────────────────────────────────────────────
    logger.info("\n### Single models ###")
    print_comparison(csv_before_single, csv_after_single)

    logger.info("\n### Ensemble / MI ###")
    print_comparison(csv_before_ensemble, csv_after_ensemble)

    logger.info(f"\n[+] Reports saved to: {report_dir}")
    for name, path in [
        ("Before AT (single)",   csv_before_single),
        ("Before AT (ensemble)", csv_before_ensemble),
        ("After  AT (single)",   csv_after_single),
        ("After  AT (ensemble)", csv_after_ensemble),
    ]:
        logger.info(f"  {name:25s}: {path}")


if __name__ == '__main__':
    main()
