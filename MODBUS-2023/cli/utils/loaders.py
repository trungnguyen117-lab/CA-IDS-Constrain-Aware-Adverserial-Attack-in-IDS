"""Centralized data loading for MODBUS-2023 pipeline."""

import logging

import numpy as np
import pandas as pd

from utils.paths import get_path

logger = logging.getLogger(__name__)


def load_dataset(key, max_samples=None):
    """Load a dataset by config key. Returns (df, X, y, feature_names).

    key: 'train_tvae', 'test', 'train', etc.
    max_samples: if set, stratified-subsample down to this size.
    """
    path = get_path(key)
    logger.info(f"Loading {key}: {path}")
    df = pd.read_csv(path)

    if max_samples and max_samples < len(df):
        from sklearn.model_selection import StratifiedShuffleSplit
        y_all = df["Label"].values
        ratio = max_samples / len(df)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=42)
        idx, _ = next(sss.split(df, y_all))
        df = df.iloc[idx].reset_index(drop=True)
        logger.info(f"Subsampled to {len(df)} samples (stratified)")

    feature_names = [c for c in df.columns if c != "Label"]
    X = df[feature_names].values.astype(np.float32)
    y = df["Label"].values.astype(int)
    return df, X, y, feature_names


def load_clip_values():
    """Per-feature clip_values from clean training data (train_tvae).

    Returns (min_per_feature, max_per_feature) as float32 arrays.
    """
    _, X_train, _, _ = load_dataset("train_tvae")
    return (X_train.min(axis=0).astype(np.float32),
            X_train.max(axis=0).astype(np.float32))


def load_train_test():
    """Load train_tvae and test datasets.

    Returns (X_train, y_train, X_test, y_test, feature_names).
    """
    _, X_train, y_train, feature_names = load_dataset("train_tvae")
    _, X_test, y_test, _ = load_dataset("test")
    return X_train, y_train, X_test, y_test, feature_names
