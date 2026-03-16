"""Attack generator factory and input metadata builder (used by generate pipeline)."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_meta(df: pd.DataFrame, label_col: str = 'Label',
               cat_max_unique: int = 20) -> dict:
    """Build the input_metadata dict expected by ART attack generators.

    Detects binary and categorical feature indices automatically:
      - binary:      unique values ⊆ {0, 1}
      - categorical: all values are integer-like AND nunique ≤ cat_max_unique
                     (excludes binary)
    """
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values.astype(np.float64)

    binary_indices = []
    cat_indices = []
    cont_indices = []

    for i in range(len(feature_cols)):
        col_vals = X[:, i]
        unique = set(np.unique(col_vals))
        if unique <= {0.0, 1.0}:
            binary_indices.append(i)
        elif len(unique) <= cat_max_unique and np.all(col_vals == col_vals.astype(int)):
            cat_indices.append(i)
        else:
            cont_indices.append(i)

    logger.info(f"[meta] Features: {len(feature_cols)} total, "
                f"{len(binary_indices)} binary, {len(cat_indices)} categorical, "
                f"{len(cont_indices)} continuous")

    return {
        'feature_names':          feature_cols,
        'label_column':           label_col,
        'clip_values':            (float(X.min()), float(X.max())),
        'cat_feature_indices':    cat_indices,
        'binary_feature_indices': binary_indices,
        'cont_feature_indices':   cont_indices,
        'class_names':            sorted(df[label_col].unique().tolist()),
    }


def make_generator(attack: str, estimator, attack_params: dict = None):
    """Return the attack generator for *attack* bound to *estimator*.

    Parameter priority (highest wins):
      1. attack_params  — from CLI ``--attack-params`` JSON
      2. foami+/config/attacks/{attack}.yaml
      3. hardcoded defaults inside each ART generator class
    """
    from art_generator.zoo      import ZooAttackGenerator
    from art_generator.deepfool import DeepFoolAttackGenerator
    from art_generator.fgsm     import FGSMAttackGenerator
    from art_generator.cw       import CWAttackGenerator
    from art_generator.pgd      import PGDAttackGenerator
    from art_generator.hsja     import HSJAAttackGenerator
    from art_generator.jsma     import JSMAAttackGenerator
    from .config import load_attack_config

    # Merge: YAML config first, then CLI overrides on top
    params = load_attack_config(attack)
    if attack_params:
        params.update(attack_params)

    generators = {
        'zoo':      ZooAttackGenerator,
        'deepfool': DeepFoolAttackGenerator,
        'fgsm':     FGSMAttackGenerator,
        'cw':       CWAttackGenerator,
        'pgd':      PGDAttackGenerator,
        'hsja':     HSJAAttackGenerator,
        'jsma':     JSMAAttackGenerator,
    }
    if attack not in generators:
        raise ValueError(f"Unknown attack: {attack}")
    return generators[attack](estimator, generator_params=params)
