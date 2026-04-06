"""Adversarial training: merge clean + adv data, retrain models."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_MODBUS = os.path.dirname(_CLI)
sys.path.insert(0, _MODBUS)
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import (
    DL_TARGETS, ALL_TARGETS,
    AT_WEIGHTS_FTT, AT_WEIGHTS_ML,
    AT_BB_WEIGHT_ZOO, AT_BB_WEIGHT_HSJA,
    AT_CLEAN_ADV_RATIO, AT_TRANSFER_SOURCES,
)
from utils.evaluation import report_metrics
from utils.loaders import load_dataset
from utils.paths import get_path, adv_train_dir
from model import get_model

logger = get_logger(__name__)


def _load_adv_csv(path):
    if not os.path.exists(path):
        logger.info(f"  [SKIP] {path}")
        return None
    df = pd.read_csv(path)
    logger.info(f"  Loaded {len(df)} samples from {os.path.basename(path)}")
    return df


def _weighted_sample_parts(sources, n_budget, rng):
    total_weight = sum(w for _, _, w in sources if w > 0)
    if total_weight == 0:
        return [], []

    parts_X, parts_y = [], []
    for name, df, weight in sources:
        if df is None or weight <= 0:
            continue
        feature_names = [c for c in df.columns if c != "Label"]
        X = df[feature_names].values.astype(np.float32)
        y = df["Label"].values.astype(int)

        n_sample = int(n_budget * weight / total_weight)
        n_sample = min(n_sample, len(X))
        idx = rng.choice(len(X), size=n_sample, replace=len(X) < n_sample)
        parts_X.append(X[idx])
        parts_y.append(y[idx])
        logger.info(f"  {name:>20s}: {len(X)} -> {n_sample}  (weight={weight})")

    return parts_X, parts_y


def _assemble_clean_and_adv(feature_names, X_clean, y_clean, parts_X, parts_y):
    if not parts_X:
        logger.warning("No AT adversarial data found")
        df = pd.DataFrame(X_clean, columns=feature_names)
        df["Label"] = y_clean
        return df

    adv_X = np.concatenate(parts_X)
    adv_y = np.concatenate(parts_y)

    X_merged = np.concatenate([X_clean, adv_X])
    y_merged = np.concatenate([y_clean, adv_y])

    df_tmp = pd.DataFrame(X_merged, columns=feature_names)
    df_tmp["Label"] = y_merged
    n_before = len(df_tmp)
    df_tmp = df_tmp.drop_duplicates().reset_index(drop=True)

    logger.info(f"Clean: {len(X_clean)} | Adv: {len(adv_X)} | Dedup: {n_before - len(df_tmp)}")
    logger.info(f"AT train: {df_tmp.shape}")
    return df_tmp


def assemble_dl_at_data(target):
    """Assemble AT data for DL: clean + self-attack WB adv (weighted)."""
    _, X_clean, y_clean, feature_names = load_dataset("train_tvae")
    n_clean = len(X_clean)
    logger.info(f"Clean train: {n_clean} samples")

    ratio = AT_CLEAN_ADV_RATIO.get(target, 1.0)
    n_budget = int(n_clean / ratio)

    adv_dir = adv_train_dir(f"{target}_sc")
    sources = []
    at_weights = AT_WEIGHTS_FTT
    for atk_name, weight in at_weights.items():
        path = os.path.join(adv_dir, f"{target}_{atk_name}_train_adv.csv")
        df = _load_adv_csv(path)
        sources.append((f"{target}_{atk_name}", df, weight))

    rng = np.random.RandomState(42)
    logger.info(f"Budget: {n_budget} (clean={n_clean}, ratio={ratio})")
    parts_X, parts_y = _weighted_sample_parts(sources, n_budget, rng)

    return _assemble_clean_and_adv(feature_names, X_clean, y_clean, parts_X, parts_y)


def assemble_ml_at_data(target):
    """Assemble AT data for ML: clean + transfer WB from ftt_sc + BB adv."""
    _, X_clean, y_clean, feature_names = load_dataset("train_tvae")
    n_clean = len(X_clean)
    logger.info(f"Clean train: {n_clean} samples")

    ratio = AT_CLEAN_ADV_RATIO.get(target, 0.1)
    n_budget = int(n_clean / ratio)

    all_sources = []

    # Transfer from ftt_sc (WB)
    for src in AT_TRANSFER_SOURCES.get(target, []):
        src_dir = adv_train_dir(src)
        src_model = src.replace("_sc", "")
        for weight_key, weight in AT_WEIGHTS_ML.items():
            if not weight_key.startswith(f"{src_model}_"):
                continue
            atk_name = weight_key[len(f"{src_model}_"):]
            path = os.path.join(src_dir, f"{src_model}_{atk_name}_train_adv.csv")
            df = _load_adv_csv(path)
            all_sources.append((weight_key, df, weight))

    # Direct BB: own zoo/hsja
    bb_dir = adv_train_dir(target)
    bb_weights = {"zoo": AT_BB_WEIGHT_ZOO, "hsja": AT_BB_WEIGHT_HSJA}
    for atk_name, bb_w in bb_weights.items():
        path = os.path.join(bb_dir, f"{target}_{atk_name}_train_adv.csv")
        df = _load_adv_csv(path)
        if df is not None:
            all_sources.append((f"{target}_{atk_name}", df, bb_w))

    rng = np.random.RandomState(42)
    logger.info(f"Budget: {n_budget} (clean={n_clean}, ratio={ratio})")
    parts_X, parts_y = _weighted_sample_parts(all_sources, n_budget, rng)

    df_merged = _assemble_clean_and_adv(feature_names, X_clean, y_clean, parts_X, parts_y)
    y_merged = df_merged["Label"].values
    logger.info(f"Label distribution:\n{pd.Series(y_merged).value_counts().sort_index()}")
    return df_merged


def retrain_model(target, df_merged, device="cpu"):
    feature_names = [c for c in df_merged.columns if c != "Label"]
    X = df_merged[feature_names].values.astype(np.float32)
    y = df_merged["Label"].values.astype(int)

    _, X_test, y_test, _ = load_dataset("test")

    models_at = get_path("models_at")
    os.makedirs(models_at, exist_ok=True)

    at_cfg = None
    if target == "cat":
        at_cfg = dict(
            iterations=5000, depth=8, learning_rate=0.05,
            l2_leaf_reg=3, early_stopping_rounds=None,
        )

    m = get_model(target)
    X_val = None if target == "cat" else X_test
    y_val = None if target == "cat" else y_test
    m.train(X_val=X_val, y_val=y_val, X_train=X, y_train=y, cfg=at_cfg, device=device)

    preds = m.predict(X_test)
    report_metrics(f"{target.upper()} AT", y_test, preds)

    ext = ".pth" if target in DL_TARGETS else ".pkl"
    m.save(os.path.join(models_at, f"framework_{target}_TabDiff_at{ext}"))


def main():
    parser = argparse.ArgumentParser(description="Adversarial training")
    parser.add_argument("--model", "-m", nargs="+", required=True,
                        choices=ALL_TARGETS + ["all"])
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

    for target in models:
        logger.info(f"{'='*60}")
        logger.info(f"Adversarial Training: {target.upper()}")
        logger.info(f"{'='*60}")

        if target in DL_TARGETS:
            df_merged = assemble_dl_at_data(target)
        else:
            df_merged = assemble_ml_at_data(target)

        retrain_model(target, df_merged, args.device)


if __name__ == "__main__":
    main()
