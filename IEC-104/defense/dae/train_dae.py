"""Train DAE on clean IEC-104 training data.

Usage:
    python defense/dae/train_dae.py --arch residual --epochs 2000 --data train_tvae
    python defense/dae/train_dae.py --arch vanilla --epochs 2000 --data train_tvae
    python defense/dae/train_dae.py --arch residual --data train_tvae --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

# Bootstrap paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_IEC = os.path.dirname(os.path.dirname(_HERE))  # IEC-104/
_CLI = os.path.join(_IEC, "cli")
sys.path.insert(0, _IEC)
sys.path.insert(0, _HERE)
sys.path.insert(0, _CLI)

from dae_model import DAE_ARCH_REGISTRY, build_dae, train_dae, save_checkpoint
from utils.loaders import load_dataset
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train DAE for adversarial purification")
    parser.add_argument("--device", "-d", default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--arch", "-a", default="residual",
                        choices=list(DAE_ARCH_REGISTRY.keys()),
                        help="Model architecture (default: residual)")
    parser.add_argument("--epochs", "-e", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--data", default="train_tvae",
                        choices=["train", "train_tvae"],
                        help="Training data key (default: train_tvae)")
    parser.add_argument("--noise-factor", type=float, default=None,
                        help="Override noise_factor from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size from config")
    parser.add_argument("--n-res-blocks", type=int, default=None,
                        help="Number of residual blocks (residual arch only)")
    parser.add_argument("--config", default=os.path.join(_HERE, "config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Cast numeric values
    for key in ("hidden_dim", "bottleneck_dim", "n_res_blocks", "epochs", "batch_size"):
        if key in cfg:
            cfg[key] = int(cfg[key])
    for key in ("lr", "noise_factor", "dropout"):
        if key in cfg:
            cfg[key] = float(cfg[key])

    epochs = args.epochs or cfg["epochs"]
    noise_factor = args.noise_factor or cfg["noise_factor"]
    batch_size = args.batch_size or cfg["batch_size"]
    n_res_blocks = args.n_res_blocks or cfg.get("n_res_blocks", 2)

    # Load clean training data
    _, X_train, _, feature_names = load_dataset(args.data)
    n_features = X_train.shape[1]
    logger.info(f"Training data: {X_train.shape[0]} samples, {n_features} features")

    # Fit MinMaxScaler on training data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)
    logger.info(f"MinMax scaled to [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")

    # Convert to tensor
    x_train = torch.from_numpy(X_scaled).to(args.device)

    # Build model
    extra = {}
    if args.arch == "residual":
        extra["n_res_blocks"] = n_res_blocks

    model = build_dae(
        arch=args.arch,
        data_dim=n_features,
        hidden_dim=cfg["hidden_dim"],
        bottleneck_dim=cfg["bottleneck_dim"],
        device=args.device,
        dropout=cfg.get("dropout", 0.1),
        **extra,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Architecture: {args.arch} ({n_params:,} params)")

    noise_schedule = cfg.get("noise_schedule")
    logger.info(
        f"Training: {epochs} epochs, lr={cfg['lr']}, batch_size={batch_size}, "
        f"noise_factor={noise_factor}, noise_schedule={noise_schedule}"
    )

    train_loss = train_dae(
        model, x_train,
        epochs=epochs, lr=cfg["lr"],
        noise_factor=noise_factor,
        noise_schedule=noise_schedule,
        batch_size=batch_size,
        device=args.device,
    )

    # Save checkpoint
    save_cfg = {
        "arch": args.arch,
        "data_dim": n_features,
        "hidden_dim": cfg["hidden_dim"],
        "bottleneck_dim": cfg["bottleneck_dim"],
        "dropout": cfg.get("dropout", 0.1),
        "noise_factor": noise_factor,
        "noise_schedule": noise_schedule,
    }
    if args.arch == "residual":
        save_cfg["n_res_blocks"] = n_res_blocks

    os.makedirs(os.path.join(_HERE, "models"), exist_ok=True)
    out_path = os.path.join(_HERE, "models", f"dae_iec104_{args.arch}.pth")
    save_checkpoint(out_path, model, save_cfg, train_loss, scaler)
    logger.info(f"Saved checkpoint to {out_path}")
    logger.info(f"Final MSE: {train_loss[-1]:.6f}")


if __name__ == "__main__":
    main()
