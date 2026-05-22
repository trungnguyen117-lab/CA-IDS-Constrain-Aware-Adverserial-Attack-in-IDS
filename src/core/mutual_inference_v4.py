"""Mutual Inference Ensemble — v4.

Triển khai bám sát Algorithm 1 (Mutual Inference Ensemble Prediction):
- ``tree_internal_disagreement`` dùng cosine giữa các tree-model (không phải vote).
- ``tree_model_confidence`` / ``dl_model_confidence`` giữ giá trị thô,
  không min-max như bản trong ``ensemble.py``.
"""

from __future__ import annotations

import numpy as np

from .ensemble import EPS as EPSILON

DEFAULTS = {"alpha": 0.5, "beta": 1.0, "tau": 0.3}


def tree_model_confidence(tree_probs: np.ndarray) -> np.ndarray:
    """Margin top1 − top2 cho từng tree-model. Shape: (N, k_tree)."""
    sorted_probs = np.sort(tree_probs, axis=-1)
    return sorted_probs[..., -1] - sorted_probs[..., -2]


def dl_model_confidence(dl_probs: np.ndarray) -> np.ndarray:
    """1 − H/log(C) cho từng DL-model. Shape: (N, k_dl)."""
    num_classes = dl_probs.shape[-1]
    entropy = -np.sum(dl_probs * np.log(dl_probs + EPSILON), axis=-1)
    return 1.0 - entropy / np.log(num_classes)


def inter_group_agreement_matrix(
    tree_probs: np.ndarray, dl_probs: np.ndarray,
) -> np.ndarray:
    """Cosine A[n, k, l] giữa tree-model k và dl-model l, theo từng sample."""
    tree_norm = np.linalg.norm(tree_probs, axis=-1, keepdims=True) + EPSILON
    dl_norm = np.linalg.norm(dl_probs, axis=-1, keepdims=True) + EPSILON
    return np.einsum(
        "nkc,nlc->nkl",
        tree_probs / tree_norm,
        dl_probs / dl_norm,
    )


def tree_internal_disagreement(tree_probs: np.ndarray) -> np.ndarray:
    """1 − mean cosine giữa các cặp tree-model. Shape: (N,)."""
    num_trees = tree_probs.shape[1]
    if num_trees < 2:
        return np.zeros(tree_probs.shape[0])
    norms = np.linalg.norm(tree_probs, axis=-1, keepdims=True) + EPSILON
    unit = tree_probs / norms
    cosine_matrix = np.einsum("nic,njc->nij", unit, unit)
    upper_idx = np.triu_indices(num_trees, k=1)
    pair_cosines = cosine_matrix[:, upper_idx[0], upper_idx[1]]
    return 1.0 - pair_cosines.mean(axis=1)


def compute_tree_weights(
    tree_confidence: np.ndarray,
    agreement_matrix: np.ndarray,
    base_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """w_tree_raw[n,k] = α·c_tree·mean_l(A[n,k,l]) + (1−α)·w_base[k]."""
    mean_agreement = agreement_matrix.mean(axis=2)
    return alpha * tree_confidence * mean_agreement + (1 - alpha) * base_weights[None, :]


def compute_dl_weights(
    dl_confidence: np.ndarray,
    agreement_matrix: np.ndarray,
    tree_disagreement: np.ndarray,
    beta: float,
    tau: float,
) -> np.ndarray:
    """w_dl_raw[n,l] = β · max(c_dl · mean_k(A[n,k,l]) · (1+D_tree) − τ, 0)."""
    mean_agreement = agreement_matrix.mean(axis=1)
    gated = dl_confidence * mean_agreement * (1.0 + tree_disagreement[:, None])
    return beta * np.maximum(gated - tau, 0.0)


def normalize_weights(
    tree_weights_raw: np.ndarray, dl_weights_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = tree_weights_raw.sum(axis=1, keepdims=True) \
        + dl_weights_raw.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        idx = np.where((total <= 0).ravel())[0]
        raise RuntimeError(
            f"MI degenerate: Σw_raw = 0 cho {len(idx)} sample(s) "
            f"(first idx: {idx[:5].tolist()}). "
            "Kiểm tra α (=1?), β (=0?), τ (quá lớn?), hoặc w_base."
        )
    return tree_weights_raw / total, dl_weights_raw / total


def aggregate_probabilities(
    tree_probs: np.ndarray, tree_weights: np.ndarray,
    dl_probs: np.ndarray, dl_weights: np.ndarray,
) -> np.ndarray:
    return (
        np.einsum("nk,nkc->nc", tree_weights, tree_probs)
        + np.einsum("nl,nlc->nc", dl_weights, dl_probs)
    )


class MutualInferenceV4:
    """Ensemble theo Algorithm 1, API: ``predict(preds_dict)``."""

    def __init__(
        self,
        cfg,
        weights: dict[str, float] | None = None,
        params: dict | None = None,
    ):
        self.cfg = cfg
        self.params = {**DEFAULTS, **(params or {})}
        if weights is None:
            n = len(cfg.all_targets) or 1
            weights = {target: 1.0 / n for target in cfg.all_targets}
        self.weights = dict(weights)

    def predict(
        self, preds: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tree_names = [n for n in self.cfg.tree_targets if n in preds]
        dl_names = [n for n in self.cfg.dl_targets if n in preds]
        if not tree_names:
            raise ValueError(f"No tree models in preds: {list(preds)}")

        tree_probs = np.stack([preds[n] for n in tree_names], axis=1)
        if not dl_names:
            ensemble_probs = tree_probs.mean(axis=1)
        else:
            dl_probs = np.stack([preds[n] for n in dl_names], axis=1)
            alpha = self.params["alpha"]
            beta = self.params["beta"]
            tau = self.params["tau"]

            tree_conf = tree_model_confidence(tree_probs)
            dl_conf = dl_model_confidence(dl_probs)
            agreement = inter_group_agreement_matrix(tree_probs, dl_probs)
            disagreement = tree_internal_disagreement(tree_probs)

            tree_base = np.array(
                [self.weights.get(n, 1.0 / len(tree_names)) for n in tree_names],
            )
            # Yaml weights gồm cả DL → subset tree không sum=1; renorm về simplex
            # để khớp contract w_base_tree của Algorithm 1.
            tree_base = tree_base / max(tree_base.sum(), EPSILON)
            tree_weights_raw = compute_tree_weights(
                tree_conf, agreement, tree_base, alpha,
            )
            dl_weights_raw = compute_dl_weights(
                dl_conf, agreement, disagreement, beta, tau,
            )
            tree_weights, dl_weights = normalize_weights(
                tree_weights_raw, dl_weights_raw,
            )
            ensemble_probs = aggregate_probabilities(
                tree_probs, tree_weights, dl_probs, dl_weights,
            )

        assert np.allclose(ensemble_probs.sum(axis=1), 1.0, atol=1e-5), \
            "MI: ensemble_probs không trên simplex — kiểm tra DL có softmax không."
        labels = np.argmax(ensemble_probs, axis=1)
        confidence = np.max(ensemble_probs, axis=1)
        return ensemble_probs, labels, confidence
