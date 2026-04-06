"""Train baseline models: XGB, CAT, RF, ET, LGBM, FTT."""

import argparse
import os
import sys

import numpy as np
import yaml

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_MODBUS = os.path.dirname(_CLI)
sys.path.insert(0, _MODBUS)
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import ALL_TARGETS, TREE_TARGETS
from utils.evaluation import report_metrics
from utils.paths import get_path, training_config_path
from utils.loaders import load_train_test
from model import get_model

logger = get_logger(__name__)


def load_yaml_config(model_name):
    path = training_config_path(model_name)
    if os.path.isfile(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def train_single(model_name, X_train, y_train, X_test, y_test, device):
    models_dir = get_path("models")
    os.makedirs(models_dir, exist_ok=True)
    cfg = load_yaml_config(model_name)
    logger.info(f"Training {model_name.upper()} with config: {cfg}")

    m = get_model(model_name)
    m.train(X_train, y_train, X_test, y_test, cfg=cfg, device=device)

    preds = m.predict(X_test)
    report_metrics(f"{model_name.upper()} baseline", y_test, preds)

    ext = ".pth" if model_name in ("ftt",) else ".pkl"
    m.save(os.path.join(models_dir, f"framework_{model_name}_TabDiff{ext}"))


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model", "-m", nargs="+", required=True,
                        choices=ALL_TARGETS + ["all"],
                        help="Models to train")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    models = ALL_TARGETS if "all" in args.model else args.model
    X_train, y_train, X_test, y_test, _ = load_train_test()
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    for name in models:
        logger.info(f"{'='*50}")
        logger.info(f"Training: {name.upper()}")
        logger.info(f"{'='*50}")
        train_single(name, X_train, y_train, X_test, y_test, args.device)


if __name__ == "__main__":
    main()
