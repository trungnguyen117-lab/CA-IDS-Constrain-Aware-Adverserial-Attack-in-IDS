"""Evaluate SOICT25 models on plain and adversarial samples.

Usage:
    # Single target, single attack
    python individual_evaluate.py --target cat --attack zoo

    # Multiple targets + multiple attacks
    python individual_evaluate.py --target cat rf lstm resdnn --attack zoo hsja

    # Ensemble / MI targets
    python individual_evaluate.py --target ensemble --attack zoo
    python individual_evaluate.py --target mi --attack zoo hsja

    # Explicit adv CSV paths
    python individual_evaluate.py --target cat --adv-in ../../adv_samples/cat/cat_zoo_adv.csv

    # Save results to CSV
    python individual_evaluate.py --target cat rf --attack zoo --output-csv results.csv
"""

import os
import sys
import argparse

# ── macOS / PyTorch compatibility (must be set before torch is imported) ────────
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ── Path bootstrap (minimal — resolves foami+/ then delegates to utils.paths) ─
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths import (
    setup_paths, ROOT_DIR, FOAMI_DIR, MODELS_DIR, ADV_DIR, REPORT_DIR, TEST_CSV,
)
setup_paths()

from utils.logging    import setup_logging, get_logger
from utils.constants  import ALL_TARGETS, ALL_ATTACKS, LABEL_COL
from utils.loaders    import require_file, build_predictor, load_features_csv, parse_ensemble_config
from utils.evaluation import predict_safe, asr, format_cm, save_cm_plot

logger = get_logger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SOICT25 models on plain and adversarial samples"
    )
    parser.add_argument('--target', '-t', nargs='+', required=True,
                        choices=ALL_TARGETS)
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
    parser.add_argument('--ensemble-weights', type=str, default=None)
    parser.add_argument('--mi-params', type=str, default=None)
    parser.add_argument('--per-class', action='store_true',
                        help="Print per-class classification report")
    parser.add_argument('--confusion-matrix', '--cm', action='store_true',
                        help="Print confusion matrix for plain and adversarial data")
    parser.add_argument('--report-dir', default=None,
                        help="Directory to save CM plots (default: <foami+>/report)")
    parser.add_argument('--output-csv', default=None,
                        help="Save summary results to CSV")
    parser.add_argument('--fallback-target', default=None,
                        choices=ALL_TARGETS,
                        help="Fallback model to use when target's adv CSV is missing "
                             "(e.g. --fallback-target lstm for tree models)")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

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

    # ── Load plain data ───────────────────────────────────────────────────────
    require_file(plain_in)
    logger.info(f"[+] Plain: {plain_in}")
    X_plain, y_plain = load_features_csv(plain_in, label_col=LABEL_COL)
    num_classes = len(np.unique(y_plain))
    input_dim   = X_plain.shape[1]
    clip_values = (float(X_plain.min()), float(X_plain.max()))
    class_names = [str(c) for c in sorted(np.unique(y_plain).tolist())]
    logger.info(f"[+] Shape={X_plain.shape}, classes={num_classes}")

    # ── Parse ensemble/MI configs ─────────────────────────────────────────────
    ew, mi_cfg, w_gbt_base = parse_ensemble_config(args)

    global_adv_tasks = ([(os.path.basename(p), p) for p in args.adv_in]
                        if args.adv_in else None)

    # ── Evaluate each target ──────────────────────────────────────────────────
    results = []

    for target in args.target:
        logger.info(f"\n{'='*60}\n  Target: {target}\n{'='*60}")

        predictor = build_predictor(target, models_dir, clip_values,
                                    num_classes, input_dim, args.device,
                                    ew, mi_cfg, w_gbt_base)

        # Plain
        logger.info("[+] Plain evaluation ...")
        y_plain_pred = predict_safe(predictor, X_plain)
        plain_acc = float(accuracy_score(y_plain, y_plain_pred))
        plain_f1  = float(f1_score(y_plain, y_plain_pred, average='macro', zero_division=0))
        logger.info(f"    Acc={plain_acc*100:.2f}%  Macro-F1={plain_f1*100:.2f}%")
        if args.per_class:
            logger.info("\n" + classification_report(
                y_plain, y_plain_pred, target_names=class_names, zero_division=0))
        if args.confusion_matrix:
            cm = confusion_matrix(y_plain, y_plain_pred)
            logger.info("\n" + format_cm(cm, class_names))
            save_cm_plot(cm, class_names,
                         title=f"Confusion Matrix — {target} / plain",
                         out_path=os.path.join(report_dir, f"cm_{target}_plain.png"))

        # Adversarial
        adv_tasks = global_adv_tasks or [
            (atk, os.path.join(adv_dir, target, f"{target}_{atk}_adv.csv"))
            for atk in (args.attack or [])
        ]

        for tag, adv_path in adv_tasks:
            if not os.path.exists(adv_path):
                # Try fallback target (e.g. lstm) when tree model has no adv CSV
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

            y_adv_pred = predict_safe(predictor, X_adv)
            adv_acc    = float(accuracy_score(y_adv, y_adv_pred))
            adv_f1     = float(f1_score(y_adv, y_adv_pred, average='macro', zero_division=0))
            attack_sr  = asr(y_plain_pred, y_adv_pred, y_plain)

            logger.info(f"    Adv Acc={adv_acc*100:.2f}%  F1={adv_f1*100:.2f}%  ASR={attack_sr*100:.2f}%")
            if args.per_class:
                logger.info("\n" + classification_report(
                    y_adv, y_adv_pred, target_names=class_names, zero_division=0))
            if args.confusion_matrix:
                cm = confusion_matrix(y_adv, y_adv_pred)
                logger.info("\n" + format_cm(cm, class_names))
                save_cm_plot(cm, class_names, title=f"Confusion Matrix — {target} / {tag}",
                             out_path=os.path.join(report_dir, f"cm_{target}_{tag}.png"))

            results.append({
                'target': target, 'attack': tag,
                'plain_acc': plain_acc, 'plain_f1': plain_f1,
                'adv_acc': adv_acc, 'adv_f1': adv_f1, 'asr': attack_sr,
            })

    if not results:
        logger.warning("[!] No results.")
        return

    # ── Detailed table ────────────────────────────────────────────────────────
    t1 = PrettyTable(['Target', 'Attack', 'Plain Acc', 'Plain F1',
                      'Adv Acc', 'Adv F1', 'ASR'])
    t1.align = 'r'
    t1.align['Target'] = 'l'
    t1.align['Attack'] = 'l'
    for r in results:
        t1.add_row([r['target'], r['attack'],
                    f"{r['plain_acc']*100:.2f}%", f"{r['plain_f1']*100:.2f}%",
                    f"{r['adv_acc']*100:.2f}%",   f"{r['adv_f1']*100:.2f}%",
                    f"{r['asr']*100:.2f}%"])
    logger.info("\n" + t1.get_string())

    # ── Compact 2D table: rows=attacks, cols=targets ──────────────────────────
    targets_order = list(dict.fromkeys(r['target'] for r in results))
    attacks_order = list(dict.fromkeys(r['attack']  for r in results))

    t2 = PrettyTable(['Attack'] + targets_order)
    t2.align = 'r'
    t2.align['Attack'] = 'l'

    plain_row = ['original']
    for t in targets_order:
        v = next((r['plain_acc'] for r in results if r['target'] == t), float('nan'))
        plain_row.append(f"{v*100:.2f}%" if v == v else 'nan')
    t2.add_row(plain_row)

    for atk in attacks_order:
        row = [atk]
        for t in targets_order:
            m = next((r for r in results if r['target'] == t and r['attack'] == atk), None)
            row.append(f"{m['adv_acc']*100:.2f}% (asr:{m['asr']*100:.2f}%)" if m else '—')
        t2.add_row(row)

    logger.info("\n" + t2.get_string())

    if args.output_csv:
        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        logger.info(f"[+] Saved: {args.output_csv}")


if __name__ == '__main__':
    main()
