"""Generate adversarial samples for SOICT25 models.

Kịch bản 1 (single model):
    python generate_adv_soict.py --target xgb   --attack zoo
    python generate_adv_soict.py --target lstm  --attack fgsm
    python generate_adv_soict.py --target resdnn --attack pgd

Kịch bản 2 (ensemble / MI):
    python generate_adv_soict.py --target ensemble --attack zoo
    python generate_adv_soict.py --target mi       --attack hsja \\
        --mi-params '{"alpha":0.3,"beta":1.2,"threshold":0.5}'
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

# ── Path bootstrap (minimal — resolves foami+/ then delegates to utils.paths) ─
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths import setup_paths, MODELS_DIR, ADV_DIR, TEST_CSV
setup_paths()

from utils.logging   import setup_logging, get_logger
from utils.constants import (
    ALL_TARGETS, ALL_ATTACKS, BLACKBOX_ATTACKS,
    GBT_TARGETS, ENSEMBLE_TARGETS, SINGLE_TARGETS,
    DEFAULT_ENSEMBLE_WEIGHTS, DEFAULT_MI_W_GBT_BASE, DEFAULT_MI_PARAMS,
    validate_attack_target,
)
from utils.loaders import ModelLoader
from utils.attacks import build_meta, make_generator

from art_classifier.ensemble_classifier import EnsembleEstimator
from art_classifier.mi_classifier       import MIEstimator

logger = get_logger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial samples for SOICT25 IEC-104 models"
    )
    parser.add_argument('--target', '-t', required=True, choices=ALL_TARGETS,
                        help="Model target to attack")
    parser.add_argument('--attack', '-a', required=True, choices=ALL_ATTACKS,
                        help="Attack algorithm")
    parser.add_argument('--data-in', '-i', default=TEST_CSV,
                        help=f"Input CSV file (default: {TEST_CSV})")
    parser.add_argument('--models-dir', default=MODELS_DIR,
                        help=f"Directory containing model files (default: {MODELS_DIR})")
    parser.add_argument('--output-dir', default=None,
                        help="Output directory (default: ./adv_samples/{target})")
    parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--samples', type=int, default=-1,
                        help="Limit number of rows (-1 = all)")
    parser.add_argument('--sampling-mode', default='sequential',
                        choices=['sequential', 'random'])
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
    parser.add_argument('--defense-params', type=str, default=None,
                        help="JSON dict of preprocessing defence params, e.g. "
                             "'{\"gaussian_augmentation\":true,\"ga_sigma\":0.1}'")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # ── Validate attack compatibility ─────────────────────────────────────────
    validate_attack_target(args.target, args.attack)

    if args.output_dir is None:
        args.output_dir = os.path.join(ADV_DIR, args.target)

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.data_in):
        raise SystemExit(f"Input data file not found: {args.data_in}")
    logger.info(f"[+] Loading input data: {args.data_in}")
    df_in = pd.read_csv(args.data_in, low_memory=False)
    logger.info(f"[+] Input shape: {df_in.shape}")

    label_col = 'Label'
    meta      = build_meta(df_in, label_col)
    feat_cols = meta['feature_names']
    X_all     = df_in[feat_cols].values.astype(np.float32)
    y_all     = df_in[label_col].values.astype(np.int64)

    logger.info(f"[+] Features: {X_all.shape[1]}, binary: {len(meta['binary_feature_indices'])}")
    logger.info(f"[+] Classes: {meta['class_names']}")
    logger.info(f"[+] clip_values: {meta['clip_values']}")

    # ── Optional sampling ─────────────────────────────────────────────────────
    if args.samples > 0:
        n = min(args.samples, X_all.shape[0])
        idx = (np.random.permutation(X_all.shape[0])[:n]
               if args.sampling_mode == 'random' else np.arange(n))
        X_all = X_all[idx]
        y_all = y_all[idx]
        logger.info(f"[+] Using {n} {args.sampling_mode} samples")

    num_classes = len(meta['class_names'])
    input_dim   = X_all.shape[1]
    clip_values = meta['clip_values']

    # ── Build ART estimator ───────────────────────────────────────────────────
    logger.info(f"[+] Building estimator for target={args.target}")
    defense_params = json.loads(args.defense_params) if args.defense_params else None
    if defense_params:
        logger.info(f"[+] Defence params: {defense_params}")

    loader = ModelLoader(args.models_dir, clip_values, num_classes, input_dim, args.device)

    estimator = loader.load(args.target, params=defense_params).get_estimator()

    # ── Generate ──────────────────────────────────────────────────────────────
    logger.info(f"[+] Initialising attack: {args.attack}")
    attack_params = json.loads(args.attack_params) if args.attack_params else {}
    generator     = make_generator(args.attack, estimator, attack_params)

    logger.info("[+] Generating adversarial samples ...")
    mutate_indices = meta['cat_feature_indices'] + meta['binary_feature_indices']
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


if __name__ == '__main__':
    main()
