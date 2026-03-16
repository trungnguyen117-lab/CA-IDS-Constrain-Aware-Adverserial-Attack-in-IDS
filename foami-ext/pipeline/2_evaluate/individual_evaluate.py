"""Evaluate individual SOICT25 models on plain and adversarial samples.

For ensemble/MI evaluation, use evaluate_ensemble_mi.py instead.

Usage:
    # Single target, single attack
    python individual_evaluate.py --target cat --attack zoo

    # Multiple targets + multiple attacks
    python individual_evaluate.py --target cat rf lstm resdnn --attack zoo hsja

    # Explicit adv CSV paths
    python individual_evaluate.py --target cat --adv-in ../../adv_samples/cat/cat_zoo_adv.csv

    # Save results to CSV
    python individual_evaluate.py --target cat rf --attack zoo --output-csv results.csv
"""

import os
import sys
import json
import argparse

# ── macOS / PyTorch compatibility (must be set before torch is imported) ────────
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
from sklearn.metrics import classification_report

# ── Path bootstrap (minimal — resolves foami+/ then delegates to utils.paths) ─
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths import setup_paths, MODELS_DIR, ADV_DIR, REPORT_DIR, TEST_CSV
setup_paths()

from utils.logging    import setup_logging, get_logger
from utils.constants  import (SINGLE_TARGETS, ALL_ATTACKS, LABEL_COL, MODEL_FILENAMES,
                              MODEL_AT_FILENAMES, MODEL_SCL_FILENAMES)
from utils.loaders    import load_features_csv, ModelLoader
from utils.data       import DataManager
from utils.evaluation import Evaluator

logger = get_logger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SOICT25 models on plain and adversarial samples"
    )
    parser.add_argument('--target', '-t', nargs='+', required=True,
                        choices=SINGLE_TARGETS)
    parser.add_argument('--attack', '-a', nargs='*', default=None,
                        choices=ALL_ATTACKS,
                        help="Attack name(s); resolves default adv CSV path")
    parser.add_argument('--plain-in', default=None,
                        help="Plain test CSV (default: <ROOT>/DS/test_shap_66.csv)")
    parser.add_argument('--adv-in', nargs='+', default=None,
                        help="Explicit adversarial CSV path(s)")
    parser.add_argument('--adv-dir', default=None,
                        help="Root dir for adv CSVs (default: <ROOT>/adv_samples)")
    parser.add_argument('--models-dir', default=None,
                        help="Models directory (default: <ROOT>/models)")
    parser.add_argument('--device', '-d', default='cpu',
                        choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--per-class', action='store_true',
                        help="Print per-class classification report")
    parser.add_argument('--confusion-matrix', '--cm', action='store_true',
                        help="Print confusion matrix for plain and adversarial data")
    parser.add_argument('--report-dir', default=None,
                        help="Directory to save CM plots (default: <foami+>/report)")
    parser.add_argument('--output-csv', default=None,
                        help="Save summary results to CSV")
    parser.add_argument('--fallback-target', default=None,
                        choices=SINGLE_TARGETS,
                        help="Fallback model to use when target's adv CSV is missing "
                             "(e.g. --fallback-target lstm for tree models)")
    parser.add_argument('--model-type', default='plain',
                        choices=['plain', 'at', 'scl'],
                        help="Model checkpoint type: plain (default), at (adversarial training), or scl (contrastive)")
    parser.add_argument('--checkpoint-suffix', default=None,
                        help="Override checkpoint suffix for all targets, e.g. '_at_pgd_fgsm'. "
                             "Result: framework_{target}_TVAE{suffix}.pth/pkl. Takes priority over --model-type.")
    parser.add_argument('--defense-params', type=str, default=None,
                        help="JSON dict of preprocessing defence params, e.g. "
                             "'{\"gaussian_augmentation\":true,\"ga_sigma\":0.1}'")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.checkpoint_suffix is not None:
        suffix = args.checkpoint_suffix
        for key in MODEL_FILENAMES:
            base = f'framework_{key}_TVAE'
            ext = '.pth' if key in ('lstm', 'resdnn') else '.pkl'
            MODEL_FILENAMES[key] = f'{base}{suffix}{ext}'
        logger.info(f"[+] Using checkpoint suffix: {suffix}")
    elif args.model_type == 'at':
        MODEL_FILENAMES.update(MODEL_AT_FILENAMES)
        logger.info("[+] Using adversarial training (AT) model checkpoints")
    elif args.model_type == 'scl':
        MODEL_FILENAMES.update(MODEL_SCL_FILENAMES)
        logger.info("[+] Using supervised contrastive learning (SCL) model checkpoints")

    if not args.attack and not args.adv_in:
        raise SystemExit("Provide --attack and/or --adv-in")

    # ── Resolve default paths ─────────────────────────────────────────────────
    plain_in   = args.plain_in   or TEST_CSV
    models_dir = args.models_dir or MODELS_DIR
    adv_dir    = args.adv_dir    or ADV_DIR
    report_dir = args.report_dir or REPORT_DIR
    if args.confusion_matrix:
        os.makedirs(report_dir, exist_ok=True)
        logger.info(f"[+] CM plots will be saved to: {report_dir}")

    # ── Load plain data via DataManager ───────────────────────────────────────
    logger.info(f"[+] Plain: {plain_in}")
    dm = DataManager(test_csv=plain_in)
    X_plain, y_plain = dm.test_data
    class_names = [str(c) for c in sorted(np.unique(y_plain).tolist())]
    logger.info(f"[+] Shape={X_plain.shape}, classes={dm.num_classes}")

    # ── Evaluator + ModelLoader ───────────────────────────────────────────────
    ev = Evaluator(class_names=class_names, report_dir=report_dir)
    loader = ModelLoader(models_dir, dm.clip_values, dm.num_classes, dm.input_dim, args.device)

    global_adv_tasks = ([(os.path.basename(p), p) for p in args.adv_in]
                        if args.adv_in else None)

    results = []
    defense_params = json.loads(args.defense_params) if args.defense_params else None
    if defense_params:
        logger.info(f"[+] Defence params: {defense_params}")

    for target in args.target:
        logger.info(f"\n{'='*60}\n  Target: {target}\n{'='*60}")

        model_wrapper = loader.load(target, params=defense_params)

        # Plain evaluation
        logger.info("[+] Plain evaluation ...")
        y_plain_pred = ev.predict_safe(model_wrapper, X_plain)
        plain_metrics = ev.report(f'{target}-plain', y_plain, y_plain_pred)
        if args.per_class:
            logger.info("\n" + classification_report(
                y_plain, y_plain_pred, target_names=class_names, zero_division=0))
        if args.confusion_matrix:
            ev.confusion_matrix(f'{target}_plain', y_plain, y_plain_pred, save_plot=True)

        # Adversarial evaluation
        adv_tasks = global_adv_tasks or [
            (atk, os.path.join(adv_dir, target, f"{target}_{atk}_adv.csv"))
            for atk in (args.attack or [])
        ]

        for tag, adv_path in adv_tasks:
            if not os.path.exists(adv_path):
                if args.fallback_target and args.fallback_target != target:
                    fb_path = os.path.join(adv_dir, args.fallback_target,
                                           f"{args.fallback_target}_{tag}_adv.csv")
                    if os.path.exists(fb_path):
                        logger.info(f"[~] Fallback [{tag}]: {target} → {args.fallback_target}: {fb_path}")
                        adv_path = fb_path
                    else:
                        logger.warning(f"[!] Not found: {adv_path} (fallback {fb_path} also missing) — skip")
                        continue
                else:
                    logger.warning(f"[!] Not found: {adv_path} — skip")
                    continue

            logger.info(f"[+] Adversarial [{tag}]: {adv_path}")
            X_adv, y_adv = load_features_csv(adv_path, label_col=LABEL_COL)

            y_adv_pred = ev.predict_safe(model_wrapper, X_adv)
            adv_metrics = ev.report(f'{target}-{tag}', y_adv, y_adv_pred)
            attack_sr = ev.asr(y_plain_pred, y_adv_pred, y_plain)
            logger.info(f"    ASR={attack_sr*100:.2f}%")

            if args.per_class:
                logger.info("\n" + classification_report(
                    y_adv, y_adv_pred, target_names=class_names, zero_division=0))
            if args.confusion_matrix:
                ev.confusion_matrix(f'{target}_{tag}', y_adv, y_adv_pred, save_plot=True)

            results.append({
                'target': target, 'attack': tag,
                'plain_acc': plain_metrics['accuracy'], 'plain_f1': plain_metrics['f1'],
                'adv_acc': adv_metrics['accuracy'], 'adv_f1': adv_metrics['f1'],
                'asr': attack_sr,
            })

    if not results:
        logger.warning("[!] No results.")
        return

    logger.info("\n" + ev.results_table(results).get_string())
    logger.info("\n" + ev.compact_table(results).get_string())
    if args.output_csv:
        ev.save_csv(results, args.output_csv)


if __name__ == '__main__':
    main()
