"""Train deep learning models: LSTM, ResDNN.

Hyperparameters match notebook 8 (8_multi_method.ipynb).
Models are saved to <models-dir>/ as PyTorch .pth checkpoints compatible
with LSTMWrapper.from_checkpoint() and ResDNNWrapper.from_checkpoint()
in src/art_classifier/.

Usage:
    # Train all DL models (shorthand)
    python train_dl.py --model all

    # Train specific models
    python train_dl.py --model lstm resdnn

    # Train only LSTM with GPU
    python train_dl.py --model lstm --device cuda

    # Custom data / output paths
    python train_dl.py --model resdnn --train-csv /data/train.csv --models-dir /out/models
"""

import os
import sys
import argparse

# ── macOS / PyTorch compatibility (must be set before torch is imported) ────────
# KMP_DUPLICATE_LIB_OK  : suppress crash when numpy + PyTorch each load libomp.
# OMP_NUM_THREADS=1     : prevent OpenMP from spawning threads that conflict with
#                         numpy's threading — fixes BatchNorm segfault on macOS.
# MKL_NUM_THREADS=1     : same for Intel MKL (relevant on Intel Macs).
# PYTORCH_ENABLE_MPS_FALLBACK: fall back to CPU for unsupported MPS ops.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

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
from model.lstm    import LSTMModel      # torch is first imported here
from model.resdnn  import ResDNNModel

import torch
torch.set_num_threads(1)   # prevent OMP thread-pool conflicts on macOS / CPU

logger = get_logger(__name__)

DL_MODELS   = ['lstm', 'resdnn']
_ALL_CHOICE = 'all'


# ── Per-model trainers ────────────────────────────────────────────────────────

def train_lstm(X_train, y_train, X_test, y_test, input_dim, num_class, models_dir, device,
               out_name='framework_lstm_TVAE.pth'):
    """BiLSTM + Attention — params from foami+/config/training/lstm.yaml."""
    logger.info("[LSTM] Starting training ...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    cfg = load_training_config('lstm')
    model = LSTMModel(
        input_dim=input_dim,
        num_class=num_class,
        device=device,
        **cfg,
    )
    model.fit(X_tr, y_tr, X_val, y_val)
    logger.info("[LSTM] Training complete")
    report_metrics('LSTM', y_test, model.predict(X_test))
    model.save_model(os.path.join(models_dir, out_name))


def train_resdnn(X_train, y_train, X_test, y_test, input_dim, num_class, models_dir, device,
                 out_name='framework_resdnn_TVAE.pth'):
    """Residual DNN — params from foami+/config/training/resdnn.yaml."""
    logger.info("[ResDNN] Starting training ...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    cfg = load_training_config('resdnn')
    model = ResDNNModel(
        input_dim=input_dim,
        num_class=num_class,
        device=device,
        **cfg,
    )
    model.fit(X_tr, y_tr, X_val, y_val)
    logger.info("[ResDNN] Training complete")
    report_metrics('ResDNN', y_test, model.predict(X_test))
    model.save_model(os.path.join(models_dir, out_name))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train deep learning models for SOICT25 IEC-104 dataset"
    )
    parser.add_argument('--model', '-m', nargs='+', required=True,
                        choices=DL_MODELS + [_ALL_CHOICE],
                        help="Model(s) to train: lstm resdnn all")
    parser.add_argument('--train-csv', default=None,
                        help=f"Training CSV (default: {TRAIN_CSV})")
    parser.add_argument('--test-csv', default=None,
                        help=f"Validation/test CSV (default: {TEST_CSV})")
    parser.add_argument('--models-dir', default=None,
                        help=f"Output directory for saved models (default: {MODELS_DIR})")
    parser.add_argument('--device', '-d', default='auto',
                        choices=['cpu', 'cuda', 'auto'],
                        help="PyTorch device (auto = use CUDA if available)")
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
    input_dim = X_train.shape[1]

    logger.info(
        f"[+] Train={X_train.shape}, Test={X_test.shape}, "
        f"Classes={num_class}, Features={input_dim}"
    )

    models = DL_MODELS if _ALL_CHOICE in args.model else args.model
    for m in models:
        if m == 'lstm':
            train_lstm(X_train, y_train, X_test, y_test,
                       input_dim, num_class, models_dir, args.device)
        elif m == 'resdnn':
            train_resdnn(X_train, y_train, X_test, y_test,
                         input_dim, num_class, models_dir, args.device)


if __name__ == '__main__':
    main()
