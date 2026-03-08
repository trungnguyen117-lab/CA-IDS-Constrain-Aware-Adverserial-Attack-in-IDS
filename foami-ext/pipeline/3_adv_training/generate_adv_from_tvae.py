"""Generate adversarial samples from TVAE-augmented training data.

Input  : adv_samples/adv_training/train_at.csv  (TVAE-augmented, from prepare_adv_data.py)
Output : adv_samples/adv_training/{target}/{target}_{attack}_adv.csv

test_data (test_shap_66.csv) is NOT used here — it is reserved for evaluation only.

Pipeline position:
    DS/train_shap_66.csv
        ↓  prepare_adv_data.py
    adv_samples/adv_training/train_at.csv     (TVAE-augmented training data)
        ↓  generate_adv_from_tvae.py          ← this script
    adv_samples/adv_training/{target}/{target}_{attack}_adv.csv
        ↓  merge_adv_data.py  (--base-csv train_at.csv --adv-dir adv_samples/adv_training)
    adv_samples/adv_training/train_at_merged.csv
        ↓  retrain_at.py / adv_train_dl.py

Usage:
    # Single model, single attack
    python generate_adv_from_tvae.py --target lstm  --attack pgd
    python generate_adv_from_tvae.py --target resdnn --attack fgsm
    python generate_adv_from_tvae.py --target xgb   --attack zoo

    # Ensemble / MI (black-box attacks only)
    python generate_adv_from_tvae.py --target ensemble --attack zoo
    python generate_adv_from_tvae.py --target mi       --attack hsja

    # Sinh đều nhau theo class (khuyến nghị)
    python generate_adv_from_tvae.py --target lstm --attack pgd \\
        --samples-per-class 1000

    # Giới hạn tổng số mẫu (không đảm bảo cân bằng)
    python generate_adv_from_tvae.py --target lstm --attack pgd \\
        --samples 5000 --sampling-mode random

    # Custom TVAE input (e.g., a specific augmented split)
    python generate_adv_from_tvae.py --target lstm --attack pgd \\
        --data-in /path/to/train_at.csv
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

# ── Path bootstrap ──────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths import setup_paths, MODELS_DIR, AT_DIR, AT_TRAIN_CSV
setup_paths()

from utils.logging   import setup_logging, get_logger
from utils.constants import (
    ALL_TARGETS, ALL_ATTACKS, BLACKBOX_ATTACKS,
    GBT_TARGETS, ENSEMBLE_TARGETS, SINGLE_TARGETS,
    LABEL_COL,
)
from utils.loaders import load_wrapper, parse_ensemble_config
from utils.attacks import build_meta, make_generator

from art_classifier.ensemble_classifier import EnsembleEstimator
from art_classifier.mi_classifier       import MIEstimator

logger = get_logger(__name__)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate adversarial samples from TVAE-augmented training data. "
            "test_data is NOT used — reserved for evaluation only."
        )
    )
    parser.add_argument('--target', '-t', required=True, choices=ALL_TARGETS,
                        help="Model target to attack")
    parser.add_argument('--attack', '-a', required=True, choices=ALL_ATTACKS,
                        help="Attack algorithm")
    parser.add_argument('--data-in', '-i', default=AT_TRAIN_CSV,
                        help=f"Input CSV — TVAE-augmented training data "
                             f"(default: {AT_TRAIN_CSV})")
    parser.add_argument('--models-dir', default=MODELS_DIR,
                        help=f"Directory containing model files (default: {MODELS_DIR})")
    parser.add_argument('--output-dir', default=None,
                        help="Output directory "
                             "(default: adv_samples/adv_training/{target})")
    parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--samples-per-class', type=int, default=-1,
                        help="Max samples per class — ensures balanced attack input "
                             "(-1 = use all). Takes priority over --samples.")
    parser.add_argument('--samples', type=int, default=-1,
                        help="Global row limit when --samples-per-class is not set (-1 = all)")
    parser.add_argument('--sampling-mode', default='random',
                        choices=['sequential', 'random'],
                        help="How to pick rows per class / globally (default: random)")
    parser.add_argument('--batch-size', type=int, default=-1,
                        help="Batch size for HSJA/JSMA (-1 = full)")
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=-1)
    parser.add_argument('--placeholder', default='original', choices=['original', 'drop'])
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--ensemble-weights', type=str, default=None,
                        help="JSON dict of model weights, e.g. '{\"cat\":0.4,\"rf\":0.6}'")
    parser.add_argument('--mi-params', type=str, default=None,
                        help="JSON dict: alpha, beta, threshold, w_gbt_base [cat_w, rf_w]")
    parser.add_argument('--attack-params', type=str, default=None,
                        help="JSON dict of attack-specific params")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # ── Validate attack compatibility ─────────────────────────────────────────
    if args.target in GBT_TARGETS and args.attack not in BLACKBOX_ATTACKS:
        raise SystemExit(
            f"Target '{args.target}' is a tree model. "
            f"Only black-box attacks are supported: {BLACKBOX_ATTACKS}. "
            f"Got: {args.attack}"
        )
    if args.target in ENSEMBLE_TARGETS and args.attack not in BLACKBOX_ATTACKS:
        raise SystemExit(
            f"Ensemble/MI targets require black-box attacks: {BLACKBOX_ATTACKS}. "
            f"Got: {args.attack}"
        )

    # ── Default output under adv_training/ (not adv_samples/{target}/) ────────
    if args.output_dir is None:
        args.output_dir = os.path.join(AT_DIR, args.target)

    # ── Load TVAE-augmented training data ─────────────────────────────────────
    if not os.path.exists(args.data_in):
        raise SystemExit(
            f"TVAE training data not found: {args.data_in}\n"
            "Run prepare_adv_data.py first to generate train_at.csv."
        )

    logger.info(f"[+] Input (TVAE-augmented training data): {args.data_in}")
    df_in = pd.read_csv(args.data_in, low_memory=False)
    logger.info(f"[+] Input shape: {df_in.shape}")

    label_col = LABEL_COL

    # Build meta from the full dataset so clip_values / feature names are correct
    meta      = build_meta(df_in, label_col)
    feat_cols = meta['feature_names']

    logger.info(f"[+] Features: {len(feat_cols)}, binary: {len(meta['binary_feature_indices'])}")
    logger.info(f"[+] Classes: {meta['class_names']}")
    logger.info(f"[+] clip_values: {meta['clip_values']}")

    # ── Per-class balanced sampling (recommended) ──────────────────────────────
    if args.samples_per_class > 0:
        parts = []
        for cls in sorted(df_in[label_col].unique()):
            df_cls   = df_in[df_in[label_col] == cls]
            n        = min(args.samples_per_class, len(df_cls))
            df_sampled = (df_cls.sample(n=n, random_state=42)
                          if args.sampling_mode == 'random'
                          else df_cls.iloc[:n])
            parts.append(df_sampled)
            logger.info(f"  Class {cls}: {len(df_cls):>7,} available → sampled {n:>7,}")
        df_in = pd.concat(parts, ignore_index=True)
        logger.info(f"[+] Balanced input shape: {df_in.shape}")

    # ── Fallback: global row limit ─────────────────────────────────────────────
    elif args.samples > 0:
        n = min(args.samples, len(df_in))
        df_in = (df_in.sample(n=n, random_state=42)
                 if args.sampling_mode == 'random'
                 else df_in.iloc[:n])
        logger.info(f"[+] Global limit: using {n} {args.sampling_mode} samples "
                    f"(class balance NOT guaranteed)")

    X_all = df_in[feat_cols].values.astype(np.float32)
    y_all = df_in[label_col].values.astype(np.int64)

    num_classes = len(meta['class_names'])
    input_dim   = X_all.shape[1]
    clip_values = meta['clip_values']

    # ── Build ART estimator ───────────────────────────────────────────────────
    logger.info(f"[+] Building estimator for target={args.target}")

    if args.target in SINGLE_TARGETS:
        estimator = load_wrapper(args.target, args.models_dir,
                                 clip_values, num_classes, input_dim,
                                 args.device).get_estimator()

    # elif args.target == 'ensemble':
    #     ew, _, _ = parse_ensemble_config(args)
    #     wrappers = {}
    #     for t in SINGLE_TARGETS:
    #         if ew.get(t, 0.0) > 0:
    #             logger.info(f"  Loading {t} ...")
    #             wrappers[t] = load_wrapper(t, args.models_dir,
    #                                        clip_values, num_classes, input_dim, args.device)
    #     estimator = EnsembleEstimator(wrappers=wrappers, weights=ew,
    #                                   num_classes=num_classes, clip_values=clip_values)

    # elif args.target == 'mi':
    #     _, mi_cfg, w_gbt_base = parse_ensemble_config(args)
    #     logger.info(f"  MI params: alpha={mi_cfg['alpha']}, beta={mi_cfg['beta']}, "
    #                 f"threshold={mi_cfg['threshold']}")
    #     logger.info("  Loading GBT wrappers (cat, rf) ...")
    #     gbt = {k: load_wrapper(k, args.models_dir, clip_values, num_classes, input_dim, args.device)
    #            for k in ('cat', 'rf')}
    #     logger.info("  Loading DL wrappers (lstm, resdnn) ...")
    #     dl  = {k: load_wrapper(k, args.models_dir, clip_values, num_classes, input_dim, args.device)
    #            for k in ('lstm', 'resdnn')}
    #     estimator = MIEstimator(gbt_wrappers=gbt, dl_wrappers=dl,
    #                             num_classes=num_classes, clip_values=clip_values,
    #                             w_gbt_base=w_gbt_base, **mi_cfg)

    # ── Generate adversarial samples ──────────────────────────────────────────
    logger.info(f"[+] Initialising attack: {args.attack}")
    attack_params = json.loads(args.attack_params) if args.attack_params else {}
    generator     = make_generator(args.attack, estimator, attack_params)

    logger.info("[+] Generating adversarial samples from TVAE data ...")
    mutate_indices  = meta['cat_feature_indices'] + meta['binary_feature_indices']
    generate_kwargs = {}
    if args.attack in ['hsja', 'jsma']:
        generate_kwargs = {
            'batch_size':  args.batch_size,
            'max_retries': args.max_retries,
            'timeout':     args.timeout,
            'placeholder': args.placeholder,
            'verbose':     args.verbose,
        }

    df_adv = generator.generate(X_all, y_all,
                                 input_metadata=meta,
                                 mutate_indices=mutate_indices,
                                 **generate_kwargs)

    logger.info(f"[+] Label distribution:\n{df_adv[label_col].value_counts().to_string()}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{args.target}_{args.attack}_adv.csv")
    df_adv.to_csv(out_csv, index=False)
    logger.info(f"[+] Saved to: {out_csv}")

    logger.info(
        "\n[+] Next step — merge with TVAE base data:\n"
        f"    python merge_adv_data.py "
        f"--base-csv {args.data_in} "
        f"--adv-dir {AT_DIR} "
        f"--targets {args.target} --attacks {args.attack}"
    )


if __name__ == '__main__':
    main()
