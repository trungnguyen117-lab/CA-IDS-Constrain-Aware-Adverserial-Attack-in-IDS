"""Weighted Soft Voting + MI ensemble for MODBUS-2023 pipeline."""

import logging

import numpy as np
from sklearn.metrics import f1_score

from .constants import DEFAULT_ENSEMBLE_WEIGHTS, GBT_GROUP, DL_GROUP

logger = logging.getLogger(__name__)


def weighted_soft_voting(preds_dict, weights=None):
    """Combine probability matrices via weighted soft voting.

    Args:
        preds_dict: {target_name: proba_matrix (n_samples, n_classes)}
        weights: {target_name: float}. Defaults to DEFAULT_ENSEMBLE_WEIGHTS.

    Returns:
        ensemble_proba: (n_samples, n_classes)
        ensemble_pred:  (n_samples,) integer class predictions
    """
    if weights is None:
        weights = DEFAULT_ENSEMBLE_WEIGHTS

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

    ensemble_proba = ensemble_proba / w_sum
    ensemble_pred = ensemble_proba.argmax(axis=1)
    return ensemble_proba, ensemble_pred


def mutual_inference(P_gbt_dict, P_dl_list, w_gbt_base, alpha, beta, threshold):
    """Mutual Inference ensemble (from notebook 6_MI).

    Args:
        P_gbt_dict: {model_name: proba (N, C)} for GBT group models
        P_dl_list: list of proba (N, C) for DL seeds
        w_gbt_base: array of base weights for GBT models (normalized)
        alpha: GBT confidence modulation (0-1)
        beta: DL contribution strength (0-3)
        threshold: DL activation threshold (0-1)

    Returns:
        y_pred: (N,) integer predictions
        p_ens: (N, C) ensemble probabilities
    """
    gbt_names = list(P_gbt_dict.keys())
    n_gbt = len(gbt_names)
    n_dl = len(P_dl_list)
    if n_gbt == 0:
        raise ValueError("No GBT models provided")

    # Stack probabilities
    P_gbt = np.stack([P_gbt_dict[n] for n in gbt_names], axis=1)  # (N, n_gbt, C)
    P_dl = np.stack(P_dl_list, axis=1)  # (N, n_dl, C)
    N, _, C = P_gbt.shape

    # GBT confidence: prediction margin (max - second max)
    sorted_gbt = np.sort(P_gbt, axis=-1)
    C_gbt = sorted_gbt[:, :, -1] - sorted_gbt[:, :, -2]  # (N, n_gbt)
    C_gbt_std = C_gbt / (C_gbt.max(axis=1, keepdims=True) + 1e-10)

    # DL confidence: entropy-based (lower entropy = higher confidence)
    eps = 1e-10
    H_dl = -np.sum(P_dl * np.log(P_dl + eps), axis=-1)  # (N, n_dl)
    H_max = np.log(C)
    C_dl = 1.0 - H_dl / H_max  # (N, n_dl)
    C_dl_std = C_dl / (C_dl.max(axis=1, keepdims=True) + 1e-10)

    # Inter-group agreement: cosine similarity between avg GBT and avg DL
    avg_gbt = P_gbt.mean(axis=1)  # (N, C)
    avg_dl = P_dl.mean(axis=1)    # (N, C)
    dot = np.sum(avg_gbt * avg_dl, axis=1)
    norm_g = np.linalg.norm(avg_gbt, axis=1) + 1e-10
    norm_d = np.linalg.norm(avg_dl, axis=1) + 1e-10
    agreement = dot / (norm_g * norm_d)  # (N,)

    A_mean_row = agreement[:, None]  # (N, 1) broadcast to GBT
    A_mean_col = agreement[:, None]  # (N, 1) broadcast to DL

    # GBT disagreement (for DL activation)
    preds_gbt = P_gbt.argmax(axis=-1)  # (N, n_gbt)
    gbt_disagree = np.array([
        1.0 - np.mean(preds_gbt[i] == preds_gbt[i, 0]) for i in range(N)
    ])  # (N,)

    # GBT weights: confidence-modulated baseline
    w_gbt_raw = alpha * C_gbt_std * A_mean_row + (1 - alpha) * w_gbt_base[None, :]

    # DL activation signal
    dl_signal = C_dl_std * A_mean_col * (1 + gbt_disagree[:, None])
    dl_active = np.maximum(dl_signal - threshold, 0)
    w_dl_raw = beta * dl_active

    # Normalize
    total = w_gbt_raw.sum(1, keepdims=True) + w_dl_raw.sum(1, keepdims=True)
    total = np.maximum(total, 1e-10)
    w_gbt = w_gbt_raw / total
    w_dl = w_dl_raw / total

    # Ensemble prediction
    p_ens = (np.einsum('nk,nkc->nc', w_gbt, P_gbt) +
             np.einsum('nl,nlc->nc', w_dl, P_dl))

    return p_ens.argmax(1), p_ens
