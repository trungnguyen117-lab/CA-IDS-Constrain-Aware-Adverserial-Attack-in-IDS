"""Predict-time wrappers: build ART predict_proba lambda cho eval pipeline."""

from __future__ import annotations

import numpy as np

from .attacks import wrap_for_art


def build_art_predictor(cfg, model, target: str, X_ref: np.ndarray,
                        device: str = "cpu", attack: str | None = None,
                        preprocessing_defences=None):
    """Return ``predict_proba(X) → (N, C) probability matrix``.

    Each model class supplies ``art_predict_proba`` (DL applies softmax to
    logits; ML returns probs as-is).
    """
    art_clf = wrap_for_art(cfg, model, target, X_ref, device=device,
                           attack=attack,
                           preprocessing_defences=preprocessing_defences)
    return lambda X: np.asarray(model.art_predict_proba(art_clf, X), dtype=np.float64)


def build_art_predictors(cfg, models: dict, X_ref: np.ndarray,
                         device: str = "cpu", preprocessing_defences=None) -> dict:
    fs_map = preprocessing_defences or {}
    return {
        n: build_art_predictor(cfg, m, n, X_ref, device=device,
                               preprocessing_defences=fs_map.get(n))
        for n, m in models.items()
    }
