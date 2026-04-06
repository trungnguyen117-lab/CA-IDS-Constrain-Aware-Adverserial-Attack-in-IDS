"""Purify adversarial CSVs using trained DAE model.

Walks adv_samples/adv_eval/ and produces purified CSVs under
defense/dae/purified/{target}/{filename}.

Usage:
    python defense/dae/purify_dae.py --device cuda
    python defense/dae/purify_dae.py --device mps --clean-test
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_IEC = os.path.dirname(os.path.dirname(_HERE))
_CLI = os.path.join(_IEC, "cli")
sys.path.insert(0, _IEC)
sys.path.insert(0, _HERE)
sys.path.insert(0, _CLI)

from dae_model import load_checkpoint
from utils.loaders import load_dataset
from utils.paths import get_path
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def purify_array(model, scaler, X_raw, device, batch_size=1024):
    """Purify raw feature array via DAE. Returns purified raw features."""
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    X_purified_parts = []

    n = X_scaled.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = torch.from_numpy(X_scaled[start:end]).to(device)
        with torch.no_grad():
            x_pure = model(x_batch)
        x_pure = torch.clamp(x_pure, 0.0, 1.0)
        X_purified_parts.append(x_pure.cpu().numpy())

    X_purified_scaled = np.concatenate(X_purified_parts, axis=0)
    X_purified_raw = scaler.inverse_transform(X_purified_scaled)
    return X_purified_raw.astype(np.float32)


def purify_csv(model, scaler, csv_path, feature_names, device, batch_size=1024):
    """Load an adversarial CSV, purify features, return purified DataFrame."""
    df = pd.read_csv(csv_path)
    X_adv = df[feature_names].values.astype(np.float32)
    labels = df["Label"].values

    X_purified = purify_array(model, scaler, X_adv, device, batch_size)

    df_out = pd.DataFrame(X_purified, columns=feature_names)
    df_out["Label"] = labels
    return df_out


def discover_adv_csvs(adv_eval_root):
    """Find all adversarial eval CSVs. Returns list of (target_dir, csv_path)."""
    results = []
    if not os.path.isdir(adv_eval_root):
        return results
    for target_dir in sorted(os.listdir(adv_eval_root)):
        target_path = os.path.join(adv_eval_root, target_dir)
        if not os.path.isdir(target_path):
            continue
        for csv_file in sorted(glob.glob(os.path.join(target_path, "*_adv.csv"))):
            results.append((target_dir, csv_file))
    return results


def main():
    parser = argparse.ArgumentParser(description="Purify adversarial samples via DAE")
    parser.add_argument("--device", "-d", default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--checkpoint",
                        default=os.path.join(_HERE, "models", "dae_iec104_residual.pth"),
                        help="Path to DAE checkpoint")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--clean-test", action="store_true",
                        help="Also purify clean test data (to measure accuracy drop)")
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

    # Load DAE model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, scaler, _ = load_checkpoint(args.checkpoint, device=args.device)

    # Get feature names from test dataset
    _, _, _, feature_names = load_dataset("test")

    # Discover adversarial CSVs
    adv_eval_root = get_path("adv_eval")
    adv_csvs = discover_adv_csvs(adv_eval_root)
    logger.info(f"Found {len(adv_csvs)} adversarial CSVs to purify")

    out_root = os.path.join(_HERE, "purified")

    # Purify adversarial CSVs
    for target_dir, csv_path in adv_csvs:
        basename = os.path.basename(csv_path)
        out_dir = os.path.join(out_root, target_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, basename)

        logger.info(f"  {target_dir}/{basename}")
        df_purified = purify_csv(
            model, scaler, csv_path, feature_names,
            args.device, args.batch_size,
        )
        df_purified.to_csv(out_path, index=False)

    # Optionally purify clean test data
    if args.clean_test:
        _, X_test, y_test, _ = load_dataset("test")
        X_test_purified = purify_array(
            model, scaler, X_test, args.device, args.batch_size,
        )
        df_clean = pd.DataFrame(X_test_purified, columns=feature_names)
        df_clean["Label"] = y_test
        os.makedirs(out_root, exist_ok=True)
        clean_out = os.path.join(out_root, "clean_test_purified.csv")
        df_clean.to_csv(clean_out, index=False)
        logger.info(f"  Clean test purified -> {clean_out}")

    logger.info(f"Done: outputs in {out_root}")


if __name__ == "__main__":
    main()
