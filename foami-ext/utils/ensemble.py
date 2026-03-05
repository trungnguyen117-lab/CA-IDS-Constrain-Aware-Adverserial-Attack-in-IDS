"""Ensemble and MI probability-combination helpers.

Used by evaluate_ensemble_mi and any script that needs per-component
probability aggregation without going through ART estimators.
"""
import numpy as np

# ── Component model groups ─────────────────────────────────────────────────────
ENSEMBLE_COMPONENTS = ['cat', 'rf', 'lstm', 'resdnn']
MI_GBT              = ['cat', 'rf']
MI_DL               = ['lstm', 'resdnn']
# DL fallback order when a component has no adv CSV for the requested attack
DL_FALLBACK         = ['lstm', 'resdnn']


# ── Combination functions ──────────────────────────────────────────────────────

def weighted_combine(
    proba_map: dict,
    weights: dict,
    num_classes: int,
) -> tuple:
    """Weighted average of per-component probability arrays.

    Args:
        proba_map:   {name: (N, C) float array}
        weights:     {name: float} — unnormalised weights
        num_classes: C (used to initialise the accumulator)

    Returns:
        (ensemble_proba, preds) as numpy arrays (float32, int64).
    """
    n        = next(iter(proba_map.values())).shape[0]
    ensemble = np.zeros((n, num_classes), dtype=np.float64)
    total_w  = sum(weights.get(k, 0.0) for k in proba_map)
    if total_w <= 0:
        total_w = 1.0
    for name, proba in proba_map.items():
        w = weights.get(name, 0.0) / total_w
        ensemble += w * proba
    preds = ensemble.argmax(axis=1).astype(np.int64)
    return ensemble.astype(np.float32), preds


def mi_combine(
    proba_map: dict,
    num_classes: int,
    w_gbt_base: np.ndarray,
    alpha: float,
    beta: float,
    threshold: float,
) -> np.ndarray:
    """MI mechanism: dynamic weighting of GBT and DL probabilities.

    Replicates MIEstimator._mi_predict_proba() operating on pre-computed
    per-component probability arrays instead of raw feature inputs.

    Args:
        proba_map:   {name: (N, C) float array}
        num_classes: number of output classes C
        w_gbt_base:  base weights for GBT models (will be normalised)
        alpha:       GBT confidence mixing coefficient
        beta:        DL confidence boost coefficient
        threshold:   DL activation threshold

    Returns:
        (N, C) float32 ensemble probability array.
    """
    gbt_keys = [k for k in MI_GBT if k in proba_map]
    dl_keys  = [k for k in MI_DL  if k in proba_map]

    if not gbt_keys or not dl_keys:
        # Fallback to uniform weighted average if components missing
        all_keys = list(proba_map.keys())
        w = {k: 1.0 / len(all_keys) for k in all_keys}
        ens, _ = weighted_combine(proba_map, w, num_classes)
        return ens

    P_gbt = np.stack([proba_map[k] for k in gbt_keys], axis=1).astype(np.float64)
    P_dl  = np.stack([proba_map[k] for k in dl_keys],  axis=1).astype(np.float64)

    w_base = np.array([w_gbt_base[i] for i in range(len(gbt_keys))], dtype=np.float64)
    w_base /= w_base.sum()

    # GBT confidence: prediction margin
    P_gbt_sorted = np.sort(P_gbt, axis=2)
    C_gbt     = P_gbt_sorted[:, :, -1] - P_gbt_sorted[:, :, -2]
    C_gbt_std = (C_gbt - C_gbt.mean(axis=0)) / (C_gbt.std(axis=0) + 1e-10)

    # DL confidence: 1 - normalised entropy
    P_dl_clip = np.clip(P_dl, 1e-10, 1.0)
    H_dl      = -np.sum(P_dl_clip * np.log(P_dl_clip), axis=2)
    C_dl      = 1.0 - H_dl / np.log(num_classes + 1e-10)
    C_dl_std  = (C_dl - C_dl.mean(axis=0)) / (C_dl.std(axis=0) + 1e-10)

    # Agreement: cosine similarity GBT ↔ DL
    ng = P_gbt / (np.linalg.norm(P_gbt, axis=2, keepdims=True) + 1e-10)
    nd = P_dl  / (np.linalg.norm(P_dl,  axis=2, keepdims=True) + 1e-10)
    A     = np.einsum('nik,njk->nij', ng, nd)
    A_row = A.mean(axis=2)   # (N, n_gbt)
    A_col = A.mean(axis=1)   # (N, n_dl)

    # GBT internal disagreement
    gbt_labels   = P_gbt.argmax(axis=2)
    gbt_agree    = np.all(gbt_labels == gbt_labels[:, :1], axis=1).astype(float)
    gbt_disagree = 1.0 - gbt_agree

    # Dynamic weights
    w_gbt_raw = alpha * C_gbt_std * A_row + (1.0 - alpha) * w_base[None, :]
    w_gbt_raw = np.clip(w_gbt_raw, 0.0, None)
    dl_signal = C_dl_std * A_col * (1.0 + gbt_disagree[:, None])
    w_dl_raw  = beta * np.maximum(dl_signal - threshold, 0.0)

    total = (w_gbt_raw.sum(axis=1, keepdims=True)
             + w_dl_raw.sum(axis=1, keepdims=True))
    total = np.maximum(total, 1e-10)
    w_gbt = w_gbt_raw / total
    w_dl  = w_dl_raw  / total

    p_ens = (np.einsum('nk,nkc->nc', w_gbt, P_gbt)
             + np.einsum('nl,nlc->nc', w_dl,  P_dl))
    return p_ens.astype(np.float32)
