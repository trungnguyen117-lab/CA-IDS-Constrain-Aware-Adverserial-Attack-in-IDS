"""Utility helpers for adversarial attack masking.

Detects binary/categorical features and provides masked generation
following the same pattern as IEC-104/art_generator/.
"""

import numpy as np
import pandas as pd


def get_mutate_indices(df: pd.DataFrame, label_col: str = "Label") -> list[int]:
    """Return column indices of binary and categorical features that should NOT be perturbed.

    Binary:      exactly 2 unique values in {0, 1}.
    Categorical: integer-valued with <= 20 unique values.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe (features + label).
    label_col : str
        Name of the label column to exclude.

    Returns
    -------
    list[int]
        Indices (0-based, relative to feature columns only) to protect.
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


def generate_masked(
    attack,
    x: np.ndarray,
    mutate_indices: list[int],
    use_mask: bool = False,
) -> np.ndarray:
    """Generate adversarial samples while protecting binary/categorical features.

    Two strategies (mirroring IEC-104/art_generator/):

    * ``use_mask=True``  – for attacks that natively support the *mask* kwarg
      (FGSM, PGD, HSJA). A mask array is built and passed to ``generate()``.
    * ``use_mask=False`` – for attacks without mask support
      (ZOO, DeepFool, CW, JSMA). Adversarial samples are generated normally,
      then protected features are restored from *x* via post-processing.

    Parameters
    ----------
    attack
        An ART ``EvasionAttack`` instance (already configured).
    x : np.ndarray
        Clean input samples, shape ``(N, F)``.
    mutate_indices : list[int]
        Feature indices to **protect** (not perturb).
    use_mask : bool
        If True, pass a mask array to the attack's ``generate()`` method.

    Returns
    -------
    np.ndarray
        Adversarial samples with protected features unchanged.
    """
    if use_mask and mutate_indices:
        mask = np.ones(x.shape, dtype=x.dtype)
        mask[:, mutate_indices] = 0
        x_adv = attack.generate(x=x, mask=mask)
    else:
        x_adv = attack.generate(x=x)
        if mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]
    return x_adv
