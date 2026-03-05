"""Train tree-based models: XGBoost, CatBoost, RandomForest.

Hyperparameters match notebook 6 (6_train_models_with_tvae.ipynb).
Models are saved to <models-dir>/ as joblib .pkl files.

Usage:
    # Train all tree models (shorthand)
    python train_tree.py --model all

    # Train specific models
    python train_tree.py --model xgb cat rf

    # Train only CatBoost with GPU
    python train_tree.py --model cat --device GPU

    # Custom data / output paths
    python train_tree.py --model rf --train-csv /data/train.csv --models-dir /out/models
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths      import setup_paths, MODELS_DIR, TRAIN_CSV, TEST_CSV
setup_paths()

from utils.logging    import setup_logging, get_logger
from utils.evaluation import report_metrics
from utils.config     import load_training_config
from training.xgb     import XGBModel
from training.catb    import CatBoostModel
from training.rf      import RFModel

logger = get_logger(__name__)

TREE_MODELS = ['xgb', 'cat', 'rf']
_ALL_CHOICE = 'all'


# ── Per-model trainers ────────────────────────────────────────────────────────

def train_xgb(X_train, y_train, X_test, y_test, num_class, models_dir, device,
              out_name='framework_xgb_TVAE.pkl'):
    """XGBoost — params from foami+/config/training/xgb.yaml."""
    logger.info("[XGB] Starting training ...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    cfg = load_training_config('xgb')
    cfg['device'] = 'cuda' if device.lower() in ('cuda', 'gpu') else 'cpu'
    model = XGBModel(num_class=num_class, params=cfg)
    model.fit(X_tr, y_tr, X_val, y_val)
    logger.info("[XGB] Training complete")
    report_metrics('XGB', y_test, model.predict(X_test))
    model.save_model(os.path.join(models_dir, out_name))


def train_cat(X_train, y_train, X_test, y_test, num_class, models_dir, device,
              out_name='framework_cat_TVAE.pkl'):
    """CatBoost — params from foami+/config/training/cat.yaml."""
    logger.info("[CatBoost] Starting training ...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    cfg = load_training_config('cat')
    cfg['task_type']     = 'GPU' if device.lower() in ('cuda', 'gpu') else 'CPU'
    cfg['classes_count'] = num_class   # inferred from data at runtime
    model = CatBoostModel(num_class=num_class, params=cfg)
    model.fit(X_tr, y_tr, X_val, y_val)
    logger.info("[CatBoost] Training complete")
    report_metrics('CatBoost', y_test, model.predict(X_test))
    model.save_model(os.path.join(models_dir, out_name))


def train_rf(X_train, y_train, X_test, y_test, num_class, models_dir,
             out_name='framework_rf_TVAE.pkl'):
    """RandomForest — params from foami+/config/training/rf.yaml."""
    logger.info("[RF] Starting training ...")
    cfg = load_training_config('rf')
    random_state = cfg.pop('random_state', 0)
    model = RFModel(num_class=num_class, params=cfg, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("[RF] Training complete")
    report_metrics('RF', y_test, model.predict(X_test))
    model.save_model(os.path.join(models_dir, out_name))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train tree-based models for SOICT25 IEC-104 dataset"
    )
    parser.add_argument('--model', '-m', nargs='+', required=True,
                        choices=TREE_MODELS + [_ALL_CHOICE],
                        help="Model(s) to train: xgb cat rf all")
    parser.add_argument('--train-csv', default=None,
                        help=f"Training CSV (default: {TRAIN_CSV})")
    parser.add_argument('--test-csv', default=None,
                        help=f"Validation/test CSV (default: {TEST_CSV})")
    parser.add_argument('--models-dir', default=None,
                        help=f"Output directory for saved models (default: {MODELS_DIR})")
    parser.add_argument('--device', '-d', default='cpu',
                        choices=['cpu', 'CPU', 'cuda', 'gpu', 'GPU'],
                        help="Compute device for XGBoost/CatBoost GPU acceleration")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    train_csv  = args.train_csv  or TRAIN_CSV
    test_csv   = args.test_csv   or TEST_CSV
    models_dir = args.models_dir or MODELS_DIR

    if not os.path.exists(train_csv):
        raise SystemExit(f"Train CSV not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise SystemExit(f"Test CSV not found: {test_csv}")
    os.makedirs(models_dir, exist_ok=True)

    logger.info(f"[+] Train: {train_csv}")
    logger.info(f"[+] Test : {test_csv}")
    df_train = pd.read_csv(train_csv, low_memory=False)
    df_test  = pd.read_csv(test_csv,  low_memory=False)

    label_col = 'Label'
    feat_cols = [c for c in df_train.columns if c != label_col]

    X_train   = df_train[feat_cols].values.astype(np.float32)
    y_train   = df_train[label_col].values.astype(np.int64)
    X_test    = df_test[feat_cols].values.astype(np.float32)
    y_test    = df_test[label_col].values.astype(np.int64)
    num_class = int(len(np.unique(y_train)))

    logger.info(f"[+] Train={X_train.shape}, Test={X_test.shape}, Classes={num_class}")

    models = TREE_MODELS if _ALL_CHOICE in args.model else args.model
    for m in models:
        if m == 'xgb':
            train_xgb(X_train, y_train, X_test, y_test, num_class, models_dir, args.device)
        elif m == 'cat':
            train_cat(X_train, y_train, X_test, y_test, num_class, models_dir, args.device)
        elif m == 'rf':
            train_rf(X_train, y_train, X_test, y_test, num_class, models_dir)


if __name__ == '__main__':
    main()
