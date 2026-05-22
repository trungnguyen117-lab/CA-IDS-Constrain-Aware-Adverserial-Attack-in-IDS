"""Dataset / model loading helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def load_dataset(cfg: Config, key_or_path: str | os.PathLike,
                 max_samples: int | None = None):
    """Load a labeled CSV from ``cfg.paths`` key or direct path.

    Returns ``(df, X, y, feats)``. ``key_or_path`` may be a config path key
    such as ``"train"``/``"test"`` or a CLI override path.
    """
    path = cfg.paths.get(str(key_or_path))
    if path is None:
        path = cfg.resolve(key_or_path)
    logger.info("Loading %s", path)
    df = pd.read_csv(path)
    if max_samples and max_samples < len(df):
        from sklearn.model_selection import StratifiedShuffleSplit
        ratio = max_samples / len(df)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=42)
        idx, _ = next(sss.split(df, df[cfg.label_col].values))
        df = df.iloc[idx].reset_index(drop=True)
        logger.info("Subsampled → %d", len(df))
    feats = [c for c in df.columns if c != cfg.label_col]
    X = df[feats].values.astype(np.float32)
    y = df[cfg.label_col].values.astype(int)
    return df, X, y, feats


def load_model(cfg: Config, target: str, defense: str | None = None,
               device: str = "cpu",
               model_dir: Path | None = None,
               defense_model_dir: Path | None = None):
    """Load a trained model. Filename = ``{target}[_{defense}].{ext}``.

    ``defense=None`` → baseline. ``defense="at"|"pgd_at"|"distill"`` → defense output.
    Override base dir via ``model_dir`` (baseline) or ``defense_model_dir`` (defense).
    """
    from .models import build_model
    base_dir = defense_model_dir if defense else model_dir
    path = cfg.model_path(target, defense=defense, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    cls = build_model(cfg, target).__class__
    logger.info("Loading %s (%s): %s", target, defense or "baseline", path)
    return cls.load(str(path), device=device)


def load_models(cfg: Config, targets, **kwargs):
    """Load multiple targets. Missing files are skipped with a warning so a
    partial set still allows downstream eval; other errors propagate."""
    out = {}
    for t in targets:
        try:
            out[t] = load_model(cfg, t, **kwargs)
        except FileNotFoundError as e:
            logger.warning("Skip %s: %s", t, e)
    return out
