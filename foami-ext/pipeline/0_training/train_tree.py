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
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)
from utils.paths      import setup_paths, MODELS_DIR, TRAIN_CSV, TEST_CSV
setup_paths()
from utils.logging    import setup_logging, get_logger
from utils.constants  import GBT_TARGETS
from utils.data       import DataManager
from utils.training import ModelManager
logger = get_logger(__name__)
_ALL_CHOICE = 'all'

def main():
    parser = argparse.ArgumentParser(
        description="Train tree-based models for SOICT25 IEC-104 dataset"
    )
    parser.add_argument('--model', '-m', nargs='+', required=True,
                        choices=GBT_TARGETS,
                        help="Tree model(s) to train: xgb cat rf all")
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
    dm = DataManager(train_csv, test_csv)
    X_train, y_train = dm.train_data
    X_test, y_test = dm.test_data
    num_class = dm.num_classes

    logger.info(f"[+] Train={X_train.shape}, Test={X_test.shape}, Classes={num_class}")
    models = GBT_TARGETS if _ALL_CHOICE in args.model else args.model
    mm = ModelManager(dm, models_dir, device=args.device)
    mm.train_multiple(models) 
if __name__ == '__main__':
    main()
