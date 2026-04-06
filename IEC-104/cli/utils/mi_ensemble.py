"""MI (Mutual Information) Adaptive Ensemble for FOAMI+ pipeline.

Two strategies:
1. Dynamic weighting: per-sample weight shift based on agreement + robustness bias
2. Confidence-gated rejection: low-confidence ensemble predictions fall back to GBT-only

When groups agree → use base weights.
When groups disagree → upweight GBT group (more robust to adversarial perturbations).
"""

import logging

import numpy as np

from .ensemble import DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)

# MI groups
GBT_GROUP = ["cat", "rf"]
DL_GROUP = ["lstm", "resdnn"]

MI_DEFAULTS = {
    "alpha": 0.0,
    "beta": 0.0,
    "agreement_threshold": 0.9768,
    "robustness_bias": 13.07,
    "confidence_gate": 0.071,
    "class_disagree_gate": False,
    "dl_temperature": 1.0,
    "dl_entropy_gate": 0.0,
}

# Adversarial-robustness optimized weights (via differential evolution
# jointly minimizing worst-case + avg ASR). Key change: LSTM nearly zeroed
# (least robust), resdnn boosted (most robust DL model on adversarial data).
ADV_WEIGHTS = {"cat": 0.076, "rf": 0.119, "lstm": 0.043, "resdnn": 0.761}


def _margin(proba):
    """Per-sample margin: top-1 minus top-2 probability."""
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    return sorted_p[:, 0] - sorted_p[:, 1]


def _entropy(proba):
    """Per-sample entropy (lower = more confident)."""
    p = np.clip(proba, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _js_divergence(p, q):
    """Per-sample Jensen-Shannon divergence between two probability distributions.

    Symmetric, bounded [0, log(2)]. Lower = more agreement.
    """
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def _group_proba(preds_dict, group, base_weights):
    """Compute weighted average probability for a model group."""
    n_samples = next(iter(preds_dict.values())).shape[0]
    n_classes = next(iter(preds_dict.values())).shape[1]
    proba = np.zeros((n_samples, n_classes))
    total_w = 0.0
    for name in group:
        if name in preds_dict:
            w = base_weights.get(name, 0.0)
            proba += w * preds_dict[name]
            total_w += w
    if total_w > 0:
        proba /= total_w
    return proba, total_w


def mi_adaptive_voting(preds_dict, base_weights=None, mi_params=None):
    """MI adaptive ensemble with per-sample dynamic weighting + optional rejection.

    Args:
        preds_dict: {target_name: proba_matrix (n_samples, n_classes)}
        base_weights: {target_name: float} — defaults to DEFAULT_WEIGHTS
        mi_params: dict with alpha, beta, agreement_threshold, robustness_bias,
                   confidence_gate

    Returns:
        ensemble_proba: (n_samples, n_classes)
        ensemble_pred: (n_samples,) integer class predictions
    """
    if base_weights is None:
        base_weights = DEFAULT_WEIGHTS
    params = {**MI_DEFAULTS, **(mi_params or {})}

    n_samples = next(iter(preds_dict.values())).shape[0]
    n_classes = next(iter(preds_dict.values())).shape[1]

    # Compute group probability averages
    gbt_proba, gbt_w = _group_proba(preds_dict, GBT_GROUP, base_weights)
    dl_proba, dl_w = _group_proba(preds_dict, DL_GROUP, base_weights)

    # Temperature scaling for DL group: soften overconfident predictions
    dl_temp = params.get("dl_temperature", 1.0)
    if dl_temp != 1.0 and dl_w > 0:
        log_dl = np.log(np.clip(dl_proba, 1e-10, 1.0)) / dl_temp
        log_dl -= log_dl.max(axis=1, keepdims=True)  # numerical stability
        dl_proba = np.exp(log_dl)
        dl_proba /= dl_proba.sum(axis=1, keepdims=True)

    # Per-sample confidence
    gbt_confidence = _margin(gbt_proba)
    dl_entropy = _entropy(dl_proba)
    max_entropy = np.log(n_classes)
    dl_confidence = 1.0 - (dl_entropy / max_entropy)

    # Inter-group agreement via Jensen-Shannon divergence
    # JS ∈ [0, log(2)], convert to agreement ∈ [0, 1] (1 = perfect agreement)
    js_div = _js_divergence(gbt_proba, dl_proba)
    agreement = 1.0 - js_div / np.log(2)

    # Per-sample dynamic weight adjustment
    alpha = params["alpha"]
    beta = params["beta"]
    threshold = params["agreement_threshold"]
    robustness_bias = params.get("robustness_bias", 0.0)

    conf_diff = gbt_confidence - dl_confidence
    disagree_factor = np.clip(1.0 - agreement / threshold, 0.0, 1.0)

    conf_adj = alpha * conf_diff
    disagree_adj = disagree_factor * (beta * conf_diff + robustness_bias)

    gbt_dynamic = gbt_w + conf_adj + disagree_adj
    dl_dynamic = dl_w - conf_adj - disagree_adj

    gbt_dynamic = np.maximum(gbt_dynamic, 0.05)
    dl_dynamic = np.maximum(dl_dynamic, 0.05)
    total = gbt_dynamic + dl_dynamic

    gbt_weight = (gbt_dynamic / total)[:, np.newaxis]
    dl_weight = (dl_dynamic / total)[:, np.newaxis]

    ensemble_proba = gbt_weight * gbt_proba + dl_weight * dl_proba

    # DL entropy gate: when DL group is uncertain (high entropy),
    # fall back to GBT — targets attacks like deepfool that create
    # ambiguous predictions near decision boundaries
    dl_entropy_gate = params.get("dl_entropy_gate", 0.0)
    if dl_entropy_gate > 0:
        dl_ent_normalized = dl_entropy / max_entropy  # [0, 1]
        high_ent_mask = dl_ent_normalized > dl_entropy_gate
        n_high_ent = np.sum(high_ent_mask)
        if n_high_ent > 0:
            ensemble_proba[high_ent_mask] = gbt_proba[high_ent_mask]
            logger.debug(f"DL entropy gate: {n_high_ent}/{n_samples} "
                        f"({100*n_high_ent/n_samples:.1f}%) → GBT fallback")

    # Class-disagreement gating: when GBT and DL groups predict different
    # top classes, fall back to GBT (trees more robust to adversarial)
    if params.get("class_disagree_gate", False):
        gbt_pred = gbt_proba.argmax(axis=1)
        dl_pred = dl_proba.argmax(axis=1)
        disagree_mask = gbt_pred != dl_pred
        n_disagree = np.sum(disagree_mask)
        if n_disagree > 0:
            ensemble_proba[disagree_mask] = gbt_proba[disagree_mask]
            logger.debug(f"Class disagree gate: {n_disagree}/{n_samples} "
                        f"({100*n_disagree/n_samples:.1f}%) → GBT fallback")

    # Confidence-gated rejection: samples with low ensemble margin
    # fall back to GBT-only prediction (trees are more robust)
    gate = params.get("confidence_gate", 0.0)
    if gate > 0:
        ensemble_margin = _margin(ensemble_proba)
        low_conf_mask = ensemble_margin < gate
        n_rejected = np.sum(low_conf_mask)
        if n_rejected > 0:
            ensemble_proba[low_conf_mask] = gbt_proba[low_conf_mask]
            logger.debug(f"Confidence gate: rejected {n_rejected}/{n_samples} "
                        f"({100*n_rejected/n_samples:.1f}%) → GBT fallback")

    ensemble_pred = ensemble_proba.argmax(axis=1)

    return ensemble_proba, ensemble_pred
