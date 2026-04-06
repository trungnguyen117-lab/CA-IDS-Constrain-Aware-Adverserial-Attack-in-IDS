"""Train Deep Similarity Encoder (DSE) for adversarial query detection."""

import argparse
import os
import sys

import yaml

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_MODBUS = os.path.dirname(_CLI)
sys.path.insert(0, _MODBUS)
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.loaders import load_dataset
from utils.paths import get_path, training_config_path
from model.dse import DSEModel

logger = get_logger(__name__)


def load_dse_config():
    path = training_config_path("dse")
    if os.path.isfile(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Train DSE encoder")
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

    cfg = load_dse_config()
    logger.info(f"DSE config: {cfg}")

    _, X_train, y_train, feature_names = load_dataset("train_tvae")
    logger.info(f"Training data: {X_train.shape}")

    n_features = X_train.shape[1]
    dse = DSEModel(
        encoder_type=cfg.get("encoder_type", "mlp"),
        n_features=n_features,
        embedding_dim=cfg.get("embedding_dim", 3),
        hidden_dims=cfg.get("hidden_dims"),
        cnn_channels=cfg.get("cnn_channels"),
        cnn_kernel_size=cfg.get("cnn_kernel_size", 3),
        device=args.device,
    )

    dse.train(X_train, cfg=cfg)

    models_dir = get_path("models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "dse_encoder.pth")
    dse.save(save_path)

    logger.info("DSE training complete.")


if __name__ == "__main__":
    main()
