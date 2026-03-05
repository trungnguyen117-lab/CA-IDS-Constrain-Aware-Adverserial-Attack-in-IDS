"""ART-compatible black-box wrapper for weighted ensemble of multiple models.

Wraps several AdversarialWrapper instances into a single predict() interface
that ART's black-box attack generators (ZOO, HSJA) can query.
"""
from typing import Dict, Optional, Tuple

import numpy as np


class EnsembleEstimator:
    """Thin ART-compatible estimator that combines weighted predictions.

    Designed to be passed as the ``classifier`` argument to black-box
    ART attack generators (ZooAttack, HopSkipJumpAttack).

    Attributes
    ----------
    wrappers : dict[str, AdversarialWrapper]
        Mapping of model name → wrapper instance.
    weights : dict[str, float]
        Per-model weight used for the weighted average.
    num_classes : int
    clip_values : tuple[float, float]
    """

    def __init__(
        self,
        wrappers: Dict[str, object],
        weights: Dict[str, float],
        num_classes: int,
        clip_values: Tuple[float, float],
    ):
        self.wrappers = wrappers
        self.weights = weights
        self.num_classes = num_classes
        self.clip_values = clip_values

        # Normalise weights
        total = sum(weights.get(k, 0.0) for k in wrappers)
        if total <= 0:
            raise ValueError("Sum of ensemble weights must be > 0")
        self._norm_weights = {k: weights.get(k, 0.0) / total for k in wrappers}

    # ART interface ────────────────────────────────────────────────────────────

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Return weighted-average probability distribution (N, C)."""
        x_f32 = x.astype(np.float32, copy=False)
        ensemble = np.zeros((x_f32.shape[0], self.num_classes), dtype=np.float64)
        for name, wrapper in self.wrappers.items():
            w = self._norm_weights[name]
            if w <= 0:
                continue
            proba = wrapper.predict_proba(x_f32)   # (N, C)
            ensemble += w * proba
        return ensemble.astype(np.float32)

    # Properties required by some ART internals ───────────────────────────────

    @property
    def nb_classes(self) -> int:
        return self.num_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        # Infer from first wrapper
        first = next(iter(self.wrappers.values()))
        return getattr(first, 'input_shape', (None,))

    def get_params(self):
        return {}
