"""Retrain ALL models on the merged adversarial-training dataset.

Trains XGBoost, CatBoost, RandomForest, LSTM, and ResDNN on the merged
CSV produced by merge_adv_data.py (base TVAE data + adversarial examples).

Saved checkpoints use the suffix '_at' to distinguish from baseline models:
    models/framework_xgb_TVAE_at.pkl
    models/framework_cat_TVAE_at.pkl
    models/framework_rf_TVAE_at.pkl
    models/framework_lstm_TVAE_at.pth
    models/framework_resdnn_TVAE_at.pth

Pipeline position:
    adv_samples/adv_training/train_at_merged.csv   (from merge_adv_data.py)
        ↓  retrain_at.py
    models/framework_{model}_TVAE_at.{ext}

Usage:
    python retrain_at.py --model all
    python retrain_at.py --model xgb cat rf
    python retrain_at.py --model lstm resdnn --device cuda
"""

import os
import sys
import argparse

# ── macOS / PyTorch compatibility ──────────────────────────────────────────────
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.realpath(__file__))
_FOAMI      = os.path.dirname(os.path.dirname(_HERE))   # foami+/
_TRAINING   = os.path.join(os.path.dirname(_HERE), '0_training')
sys.path.insert(0, _FOAMI)
sys.path.insert(0, _TRAINING)   # makes train_tree / train_dl importable

from utils.paths import setup_paths, MODELS_DIR, TEST_CSV, AT_MERGED_CSV
setup_paths()

from utils.logging import setup_logging, get_logger
from utils.constants import LABEL_COL, MODEL_AT_FILENAMES
from utils.loaders import load_features_csv
from train_tree import train_xgb, train_cat, train_rf
from train_dl   import train_lstm, train_resdnn

import torch
torch.set_num_threads(1)

logger = get_logger(__name__)

ALL_MODELS  = ['xgb', 'cat', 'rf', 'lstm', 'resdnn']
_ALL_CHOICE = 'all'


def main():
    parser = argparse.ArgumentParser(
        description="Retrain all models on merged adversarial training data"
    )
    parser.add_argument('--model', '-m', nargs='+', required=True,
                        choices=ALL_MODELS + [_ALL_CHOICE],
                        help="Model(s) to retrain: xgb cat rf lstm resdnn all")
    parser.add_argument('--train-csv', default=None,
                        help=f"Merged training CSV (default: {AT_MERGED_CSV}). "
                             "Run merge_adv_data.py first.")
    parser.add_argument('--test-csv', default=None,
                        help=f"Test CSV (default: {TEST_CSV})")
    parser.add_argument('--models-dir', default=None,
                        help=f"Output directory (default: {MODELS_DIR})")
    parser.add_argument('--device', '-d', default='cpu',
                        choices=['cpu', 'cuda', 'gpu', 'auto'])
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    train_csv  = args.train_csv  or AT_MERGED_CSV
    test_csv   = args.test_csv   or TEST_CSV
    models_dir = args.models_dir or MODELS_DIR

    if not os.path.exists(train_csv):
        raise SystemExit(
            f"Merged training CSV not found: {train_csv}\n"
            "Run merge_adv_data.py first."
        )
    if not os.path.exists(test_csv):
        raise SystemExit(f"Test CSV not found: {test_csv}")
    os.makedirs(models_dir, exist_ok=True)

    logger.info(f"[+] Train : {train_csv}")
    logger.info(f"[+] Test  : {test_csv}")
    logger.info(f"[+] Output: {models_dir}")

    X_train, y_train = load_features_csv(train_csv, label_col=LABEL_COL)
    X_test,  y_test  = load_features_csv(test_csv,  label_col=LABEL_COL)
    num_class = int(len(np.unique(y_train)))
    input_dim = X_train.shape[1]

    logger.info(
        f"[+] Train={X_train.shape}, Test={X_test.shape}, "
        f"Classes={num_class}, Features={input_dim}"
    )

    models = ALL_MODELS if _ALL_CHOICE in args.model else args.model
    for m in models:
        out = MODEL_AT_FILENAMES[m]
        if m == 'xgb':
            train_xgb(X_train, y_train, X_test, y_test, num_class, models_dir, args.device, out_name=out)
        elif m == 'cat':
            train_cat(X_train, y_train, X_test, y_test, num_class, models_dir, args.device, out_name=out)
        elif m == 'rf':
            train_rf(X_train, y_train, X_test, y_test, num_class, models_dir, out_name=out)
        elif m == 'lstm':
            train_lstm(X_train, y_train, X_test, y_test, input_dim, num_class, models_dir, args.device, out_name=out)
        elif m == 'resdnn':
            train_resdnn(X_train, y_train, X_test, y_test, input_dim, num_class, models_dir, args.device, out_name=out)


if __name__ == '__main__':
    main()
