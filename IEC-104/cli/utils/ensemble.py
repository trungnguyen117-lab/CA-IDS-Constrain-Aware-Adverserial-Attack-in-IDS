"""Weighted Soft Voting ensemble for FOAMI+ pipeline."""

import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .constants import ALL_TARGETS
from .evaluation import macro_tpr_fpr, compute_asr

logger = logging.getLogger(__name__)

# Default weights from notebook 7 grid search (XGB excluded → weight 0)
# Order: cat, rf, lstm, resdnn
DEFAULT_WEIGHTS = {
    "cat": 0.25,
    "rf": 0.35,
    "lstm": 0.10,
    "resdnn": 0.30,
}


def weighted_soft_voting(preds_dict, weights=None):
    """Combine probability matrices via weighted soft voting.

    Args:
        preds_dict: {target_name: proba_matrix (n_samples, n_classes)}
        weights: {target_name: float} — must sum to 1.
                 Defaults to DEFAULT_WEIGHTS.

    Returns:
        ensemble_proba: (n_samples, n_classes)
        ensemble_pred:  (n_samples,) integer class predictions
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    ensemble_proba = None
    w_sum = 0.0

    for name, proba in preds_dict.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        if ensemble_proba is None:
            ensemble_proba = w * proba
        else:
            ensemble_proba += w * proba
        w_sum += w

    if ensemble_proba is None or w_sum == 0.0:
        raise ValueError("No valid predictions with non-zero weights")

    # Normalize in case weights don't sum to 1
    ensemble_proba = ensemble_proba / w_sum
    ensemble_pred = ensemble_proba.argmax(axis=1)

    return ensemble_proba, ensemble_pred


def kmeans_corrected_ensemble(corrected_preds_dict, weights=None):
    """Weighted soft voting on KMeans-corrected probability vectors.

    Probabilities are already corrected per-model by KMeansDefense.correct_proba()
    before being passed here. This is just standard weighted_soft_voting on the
    corrected inputs — no weight tricks needed.

    Args:
        corrected_preds_dict: {name: corrected_proba (n_samples, n_classes)}
        weights: {name: float} — defaults to DEFAULT_WEIGHTS

    Returns:
        ensemble_proba: (n_samples, n_classes)
        ensemble_pred:  (n_samples,) integer predictions
    """
    return weighted_soft_voting(corrected_preds_dict, weights)
