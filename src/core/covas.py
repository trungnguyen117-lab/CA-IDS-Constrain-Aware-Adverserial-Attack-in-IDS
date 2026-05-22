"""CovaS pair-feature analysis: detect features that fail to separate two classes.

A feature is *dead* when, restricted to two specific classes ``a`` and ``b``,
its empirical distributions overlap heavily, are statistically indistinguishable
(KS), and have nearly identical means. Used to drop uninformative features
before SHAP selection.

Implementation mirrors MODBUS-2023/script/2_covaS_shap_2_8.ipynb (cells 3, 6,
8, 13): robust 1-99 percentile range for histograms, density-based overlap,
inf/NaN coercion, and ``len < 10`` skip for KS/dist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class CovasThresholds:
    overlap: float = 0.90
    ks: float = 0.05
    dist: float = 0.01


def _clean_series(s: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(s, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=np.float64)
    )


def histogram_overlap(xa: np.ndarray, xb: np.ndarray, bins: int, robust: bool) -> float:
    """Notebook-compatible overlap: density histograms over a shared range."""
    if len(xa) == 0 or len(xb) == 0:
        return float("nan")
    if robust:
        amin, amax = np.percentile(xa, [1, 99])
        bmin, bmax = np.percentile(xb, [1, 99])
        lo, hi = min(amin, bmin), max(amax, bmax)
    else:
        lo, hi = min(xa.min(), xb.min()), max(xa.max(), xb.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 1.0
    ha, edges = np.histogram(xa, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(xb, bins=bins, range=(lo, hi), density=True)
    bin_width = edges[1] - edges[0]
    return float(np.minimum(ha, hb).sum() * bin_width)


def pair_feature_stats(
    df: pd.DataFrame,
    label_col: str,
    id_a: int,
    id_b: int,
    features: list[str],
    bins: int = 100,
    robust: bool = True,
) -> pd.DataFrame:
    """Compute per-feature (overlap, ks_stat, p_value, dist) for class pair ``(a, b)``."""
    a_mask = df[label_col] == id_a
    b_mask = df[label_col] == id_b
    if not a_mask.any() or not b_mask.any():
        raise ValueError(
            f"CovaS: empty class — id_a={id_a} (n={int(a_mask.sum())}), "
            f"id_b={id_b} (n={int(b_mask.sum())})"
        )

    rows = []
    for feat in features:
        xa = _clean_series(df.loc[a_mask, feat])
        xb = _clean_series(df.loc[b_mask, feat])
        if len(xa) < 10 or len(xb) < 10:
            continue

        overlap = histogram_overlap(xa, xb, bins=bins, robust=robust)
        ks = ks_2samp(xa, xb, alternative="two-sided", mode="auto")
        sa, sb = float(xa.std()), float(xb.std())
        denom = sa + sb + 1e-8
        dist = float(abs(xa.mean() - xb.mean()) / denom)

        rows.append({
            "feature": feat,
            "overlap": overlap,
            "ks_stat": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "dist": dist,
        })

    return pd.DataFrame(rows, columns=["feature", "overlap", "ks_stat", "p_value", "dist"])


def dead_features(stats: pd.DataFrame, thresholds: CovasThresholds) -> list[str]:
    """Return features satisfying *all three* conditions simultaneously."""
    mask = (
        (stats["overlap"] >= thresholds.overlap)
        & (stats["ks_stat"] <= thresholds.ks)
        & (stats["dist"] <= thresholds.dist)
    )
    return stats.loc[mask, "feature"].tolist()
