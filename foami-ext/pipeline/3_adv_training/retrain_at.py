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

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.realpath(__file__))
_FOAMI      = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths import setup_paths, MODELS_DIR, TEST_CSV, AT_MERGED_CSV
setup_paths()

from utils.logging import setup_logging, get_logger
from utils.constants import MODEL_AT_FILENAMES, SINGLE_TARGETS
from utils.data import DataManager
from utils.training import ModelManager

import torch
torch.set_num_threads(1)

logger = get_logger(__name__)

_ALL_CHOICE = 'all'


def main():
    parser = argparse.ArgumentParser(
        description="Retrain all models on merged adversarial training data"
    )
    parser.add_argument('--model', '-m', nargs='+', required=True,
                        choices=SINGLE_TARGETS + [_ALL_CHOICE],
                        help="Model(s) to retrain: xgb cat rf lstm resdnn all")
    parser.add_argument('--train-csv', default=None,
                        help=f"Merged training CSV for shared mode (default: {AT_MERGED_CSV}). "
                             "Run merge_adv_data.py first.")
    parser.add_argument('--per-model-dir', default=None,
                        help="Directory containing per-model CSVs (train_at_{model}.csv). "
                             "When set, each model loads its own CSV instead of --train-csv.")
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

    test_csv   = args.test_csv   or TEST_CSV
    models_dir = args.models_dir or MODELS_DIR

    if not os.path.exists(test_csv):
        raise SystemExit(f"Test CSV not found: {test_csv}")
    os.makedirs(models_dir, exist_ok=True)

    models = SINGLE_TARGETS if _ALL_CHOICE in args.model else args.model

    if args.per_model_dir:
        logger.info(f"[+] Per-model mode: {args.per_model_dir}")
        logger.info(f"[+] Test  : {test_csv}")
        logger.info(f"[+] Output: {models_dir}")

        for m in models:
            csv_path = os.path.join(args.per_model_dir, f"train_at_{m}.csv")
            if not os.path.exists(csv_path):
                raise SystemExit(
                    f"Per-model CSV not found: {csv_path}\n"
                    "Run merge_adv_data.py --per-model first."
                )

            dm = DataManager(train_csv=csv_path, test_csv=test_csv)
            logger.info(f"[+] {m}: Train={dm.train_data[0].shape}, "
                        f"Classes={dm.num_classes}, Features={dm.input_dim}")

            mm = ModelManager(dm, models_dir, args.device)
            mm.train(m, out_name=MODEL_AT_FILENAMES[m])
    else:
        train_csv = args.train_csv or AT_MERGED_CSV

        if not os.path.exists(train_csv):
            raise SystemExit(
                f"Merged training CSV not found: {train_csv}\n"
                "Run merge_adv_data.py first."
            )

        logger.info(f"[+] Train : {train_csv}")
        logger.info(f"[+] Test  : {test_csv}")
        logger.info(f"[+] Output: {models_dir}")

        dm = DataManager(train_csv, test_csv)
        logger.info(f"[+] Train={dm.train_data[0].shape}, Test={dm.test_data[0].shape}, "
                    f"Classes={dm.num_classes}, Features={dm.input_dim}")

        mm = ModelManager(dm, models_dir, args.device)
        mm.train_multiple(models, out_names=MODEL_AT_FILENAMES)


if __name__ == '__main__':
    main()
