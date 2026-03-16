"""Gaussian Augmentation — offline data augmentation for adversarial training.

Adds Gaussian noise copies to training data to improve model robustness.

Input  : adv_samples/adv_training/train_at.csv  (or custom via --data-in)
Output : adv_samples/adv_training/train_at_ga.csv

Pipeline position:
    DS/train_shap_66.csv
        ↓  prepare_adv_data.py
    adv_samples/adv_training/train_at.csv
        ↓  gaussian_augment.py           ← this script
    adv_samples/adv_training/train_at_ga.csv
        ↓  generate_adv_from_tvae.py / merge_adv_data.py / retrain_at.py

Usage:
    python gaussian_augment.py --sigma 0.1 --ratio 1
    python gaussian_augment.py --data-in /path/to/train_at.csv --sigma 0.05 --ratio 2
    python gaussian_augment.py --samples-per-class 1000 --sigma 0.1
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

# ── Path bootstrap ──────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami-ext/
sys.path.insert(0, _FOAMI)

from utils.paths import setup_paths, AT_DIR, AT_TRAIN_CSV, AT_GA_CSV
setup_paths()

from utils.logging   import setup_logging, get_logger
from utils.constants import LABEL_COL
from utils.config    import ConfigLoader
from utils.data      import DataManager, gaussian_augment

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian Augmentation — add Gaussian noise copies to training data."
    )
    parser.add_argument('--data-in', '-i', default=AT_TRAIN_CSV,
                        help=f"Input CSV (default: {AT_TRAIN_CSV})")
    parser.add_argument('--out-csv', '-o', default=AT_GA_CSV,
                        help=f"Output CSV (default: {AT_GA_CSV})")
    parser.add_argument('--sigma', type=float, default=None,
                        help="Noise std dev (overrides YAML)")
    parser.add_argument('--ratio', type=int, default=None,
                        help="Number of noisy copies per sample (overrides YAML)")
    parser.add_argument('--mode', default='augment',
                        choices=['augment', 'replace'],
                        help="'augment' = keep original + noisy copies; "
                             "'replace' = only output noisy version (same size)")
    parser.add_argument('--samples-per-class', type=int, default=-1,
                        help="Balanced sampling per class (-1 = all)")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # ── Load config with CLI overrides ────────────────────────────────────────
    cfg = ConfigLoader().load_with_overrides(
        'gaussian_augment', 'adv_training',
        overrides={'sigma': args.sigma, 'ratio': args.ratio},
    )
    sigma = cfg.get('sigma', 0.1)
    ratio = cfg.get('ratio', 1)
    do_clip = cfg.get('clip', True)

    logger.info(f"[+] GA params: sigma={sigma}, ratio={ratio}, clip={do_clip}")

    # ── Load data ─────────────────────────────────────────────────────────────
    dm = DataManager(train_csv=args.data_in)
    X, y = dm.train_data
    feat_cols = [c for c in pd.read_csv(args.data_in, nrows=0).columns if c != LABEL_COL]

    logger.info(f"[+] Input: {args.data_in} — shape ({len(X)}, {len(feat_cols)})")

    # ── Per-class balanced sampling ───────────────────────────────────────────
    if args.samples_per_class > 0:
        parts_X, parts_y = [], []
        for cls in np.unique(y):
            mask = y == cls
            X_cls, y_cls = X[mask], y[mask]
            n = min(args.samples_per_class, len(X_cls))
            idx = np.random.choice(len(X_cls), size=n, replace=False)
            parts_X.append(X_cls[idx])
            parts_y.append(y_cls[idx])
            logger.info(f"  Class {cls}: {mask.sum():>7,} available → sampled {n:>7,}")
        X = np.concatenate(parts_X, axis=0)
        y = np.concatenate(parts_y, axis=0)
        logger.info(f"[+] After balanced sampling: {len(X)} samples")

    # ── Augment ───────────────────────────────────────────────────────────────
    clip_range = dm.clip_values if do_clip else None
    X_aug, y_aug = gaussian_augment(X, y, sigma, ratio, clip_range, mode=args.mode)

    logger.info(f"[+] Augmented: {len(X)} original + {len(X_aug) - len(X)} noisy = {len(X_aug)} total")

    # ── Save ──────────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(X_aug, columns=feat_cols)
    df_out[LABEL_COL] = y_aug
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    logger.info(f"[+] Saved to: {args.out_csv}")

    # ── Class distribution ────────────────────────────────────────────────────
    dist = pd.Series(y_aug).value_counts().sort_index()
    logger.info(f"[+] Label distribution:\n{dist.to_string()}")


if __name__ == '__main__':
    main()
