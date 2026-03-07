"""ART-compatible black-box wrapper for the Mutual Inference (MI) ensemble.

Implements the 3-parameter MI mechanism from notebook 9 as a callable
estimator that ART's black-box generators (ZOO, HSJA) can query.

Parameters
----------
alpha : float [0, 1]
    GBT confidence modulation. At 0 → fixed GBT weights.
beta : float [0, 3]
    DL contribution strength. At 0 → pure GBT voting.
threshold : float [0, 1]
    DL activation threshold (sparse activation).
"""
from typing import Dict, Optional, Tuple

import numpy as np


class MIEstimator:
    """Mutual Inference ensemble estimator (black-box ART interface).

    Combines CatBoost + RF (GBT) with LSTM + ResDNN (DL) using the MI
    mechanism optimised in notebook 9.

    Attributes
    ----------
    gbt_wrappers : dict[str, wrapper]  — {"cat": ..., "rf": ...}
    dl_wrappers  : dict[str, wrapper]  — {"lstm": ..., "resdnn": ...}
    w_gbt_base   : np.ndarray shape (2,) — baseline GBT weights [cat, rf]
    alpha, beta, threshold : float — MI hyper-parameters
    """

    def __init__(
        self,
        gbt_wrappers: Dict[str, object],
        dl_wrappers: Dict[str, object],
        num_classes: int,
        clip_values: Tuple[float, float],
        w_gbt_base: Optional[np.ndarray] = None,
        alpha: float = 0.0,
        beta: float = 0.0,
        threshold: float = 0.5,
    ):
        self.gbt_wrappers = gbt_wrappers      # ordered: [cat, rf]
        self.dl_wrappers  = dl_wrappers       # ordered: [lstm, resdnn]
        self.num_classes  = num_classes
        self.clip_values  = clip_values
        self.alpha        = float(alpha)
        self.beta         = float(beta)
        self.threshold    = float(threshold)

        # Default equal GBT weights if not provided
        n_gbt = len(gbt_wrappers)
        if w_gbt_base is None:
            self.w_gbt_base = np.ones(n_gbt, dtype=np.float64) / n_gbt
        else:
            arr = np.asarray(w_gbt_base, dtype=np.float64)
            self.w_gbt_base = arr / arr.sum()

    # ── Core MI logic ─────────────────────────────────────────────────────────

    def _mi_predict_proba(self, x: np.ndarray) -> np.ndarray:
        from utils.ensemble import mi_combine

        x_f32 = x.astype(np.float32, copy=False)
        proba_map = {}
        for name, w in self.gbt_wrappers.items():
            proba_map[name] = w.predict_proba(x_f32)
        for name, w in self.dl_wrappers.items():
            proba_map[name] = w.predict_proba(x_f32)

        return mi_combine(proba_map, self.num_classes, self.w_gbt_base,
                          self.alpha, self.beta, self.threshold)

    # ── ART interface ─────────────────────────────────────────────────────────

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._mi_predict_proba(x)

    # ── Properties required by some ART internals ─────────────────────────────

    @property
    def nb_classes(self) -> int:
        return self.num_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        first = next(iter(self.gbt_wrappers.values()))
        return getattr(first, 'input_shape', (None,))

    def get_params(self):
        return {}
