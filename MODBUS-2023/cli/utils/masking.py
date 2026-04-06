"""Utility helpers for adversarial attack masking.

Detects binary/categorical features and provides masked generation.
"""

import numpy as np
import pandas as pd


def get_mutate_indices(df: pd.DataFrame, label_col: str = "Label") -> list[int]:
    """Return column indices of binary and categorical features that should NOT be perturbed.

    Binary:      exactly 2 unique values in {0, 1}.
    Categorical: integer-valued with <= 20 unique values.

    Returns indices (0-based, relative to feature columns only) to protect.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    indices = []
    for i, col in enumerate(feature_cols):
        vals = df[col].dropna()
        n_unique = vals.nunique()
        is_int = (vals == vals.astype(int)).all()
        if (n_unique == 2 and set(vals.unique()).issubset({0, 1, 0.0, 1.0})) or (
            is_int and n_unique <= 20
        ):
            indices.append(i)
    return indices
