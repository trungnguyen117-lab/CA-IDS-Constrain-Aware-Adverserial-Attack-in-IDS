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

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths      import setup_paths, MODELS_DIR, TRAIN_CSV, TEST_CSV
setup_paths()

from utils.logging    import setup_logging, get_logger
from utils.constants  import DL_TARGETS
from utils.data       import DataManager
from utils.training   import ModelManager

import torch
torch.set_num_threads(1)   # prevent OMP thread-pool conflicts on macOS / CPU

logger = get_logger(__name__)

_ALL_CHOICE = 'all'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train deep learning models for SOICT25 IEC-104 dataset"
    )
    parser.add_argument('--model', '-m', nargs='+', required=True,
                        choices=DL_TARGETS + [_ALL_CHOICE],
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

    logger.info(f"[+] Train: {train_csv}")
    logger.info(f"[+] Test : {test_csv}")
    dm = DataManager(train_csv, test_csv)
    logger.info(f"[+] Train={dm.train_data[0].shape}, Test={dm.test_data[0].shape}, "
                f"Classes={dm.num_classes}, Features={dm.input_dim}")

    models = DL_TARGETS if _ALL_CHOICE in args.model else args.model
    mm = ModelManager(dm, models_dir, device=args.device)
    mm.train_multiple(models)


if __name__ == '__main__':
    main()
