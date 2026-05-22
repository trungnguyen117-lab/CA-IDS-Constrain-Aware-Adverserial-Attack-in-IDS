"""Ensemble voting: static, mi4, mi5."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-10
MI_DEFAULTS = {"alpha": 0.5, "beta": 1.0, "tau": 0.3}

_simplex_warned = False


def ensure_simplex(P):
    """Force last-axis to a valid probability simplex (softmax if needed)."""
    global _simplex_warned
    if P.min() < 0 or not np.allclose(P.sum(-1), 1.0, atol=0.01):
        if not _simplex_warned:
            logger.warning("DL outputs not on simplex — applying softmax fallback")
            _simplex_warned = True
        P = P - P.max(-1, keepdims=True)
        P = np.exp(P)
        P = P / P.sum(-1, keepdims=True)
    return P


class Ensemble:
    """Combine N model probabilities → ensemble probability + prediction.

    Strategies:
      - ``static``: soft voting với trọng số cố định.
      - ``mi4``:    Algorithm 1 — confidence × cosine agreement gating.
      - ``mi5``:    v5 — consistency-based tree, consensus-aligned DL gate.

    cfg.tree_targets defines the tree group; cfg.dl_targets the DL group
    (used by mi4/mi5).
    """

    STRATEGIES = ("static", "mi4", "mi5")

    def __init__(self, cfg, strategy: str = "static",
                 weights: dict | None = None, params: dict | None = None):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}")
        self.cfg = cfg
        self.strategy = strategy
        if weights is None:
            n = len(cfg.all_targets) or 1
            weights = {t: 1.0 / n for t in cfg.all_targets}
        if strategy == "static":
            total = sum(weights.values())
            if not np.isclose(total, 1.0, atol=1e-3):
                raise ValueError(
                    f"Static ensemble weights sum = {total:.4f}, must equal 1.0. "
                    f"Got: {weights}"
                )
        self.weights = dict(weights)
        if strategy == "mi4":
            self.params = {**MI_DEFAULTS, **(params or {})}
        elif strategy == "mi5":
            from .mutual_inference_v5 import MI5_DEFAULTS
            self.params = {**MI5_DEFAULTS, **(params or {})}
        else:
            self.params = params or {}

    def predict(self, preds: dict[str, np.ndarray]):
        names = list(preds)
        N, C = next(iter(preds.values())).shape

        if self.strategy == "static":
            ens = np.zeros((N, C))
            for n in names:
                ens += self.weights.get(n, 0.0) * preds[n]
        elif self.strategy == "mi4":
            from .mutual_inference_v4 import MutualInferenceV4
            ens, _, _ = MutualInferenceV4(
                self.cfg, weights=self.weights, params=self.params,
            ).predict(preds)
        else:  # mi5
            from .mutual_inference_v5 import MutualInferenceV5
            ens, _, _ = MutualInferenceV5(
                self.cfg, weights=self.weights, params=self.params,
            ).predict(preds)

        ens = ens / (ens.sum(axis=1, keepdims=True) + EPS)
        return ens, np.argmax(ens, axis=1)
