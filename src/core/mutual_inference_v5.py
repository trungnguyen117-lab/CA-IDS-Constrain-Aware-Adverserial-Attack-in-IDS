"""Mutual Inference Ensemble — v5.

Fix các vấn đề thiết kế của v4:
- Tree dynamic chỉ dùng ``tree_internal_consistency`` (cosine giữa các tree),
  DL không còn quyền "chấm điểm" tree → tránh khuếch đại co-fool.
- DL gating đối chiếu với ``tree_consensus`` (weighted sum tree probs) thay vì
  mean cosine với từng tree → 1 tree-lừa không kéo được tín hiệu lên.
- Bỏ ``(1 + D_tree)`` boost — heuristic này sai chiều trong adv scenario.
- Thêm ``consensus_penalty`` (γ): tree có argmax lệch consensus bị nhân γ<1
  vào dynamic term → co-fool tree bị hạ thêm.
"""

from __future__ import annotations

import numpy as np

from .ensemble import EPS as EPSILON

MI5_DEFAULTS = {"alpha": 0.5, "beta": 1.0, "tau": 0.3, "gamma": 0.5}


def tree_model_confidence(tree_probs: np.ndarray) -> np.ndarray:
    """Margin top1 − top2 cho từng tree. Shape: (N, K_T)."""
    s = np.sort(tree_probs, axis=-1)
    return s[..., -1] - s[..., -2]


def dl_model_confidence(dl_probs: np.ndarray) -> np.ndarray:
    """1 − H/log(C) cho từng DL-model. Shape: (N, K_D)."""
    C = dl_probs.shape[-1]
    H = -np.sum(dl_probs * np.log(dl_probs + EPSILON), axis=-1)
    return 1.0 - H / np.log(C)


def tree_internal_consistency(tree_probs: np.ndarray) -> np.ndarray:
    """C[n,k] = mean_{j≠k} cosine(tree_k, tree_j). Shape: (N, K_T)."""
    K = tree_probs.shape[1]
    if K < 2:
        return np.ones(tree_probs.shape[:2])
    norms = np.linalg.norm(tree_probs, axis=-1, keepdims=True) + EPSILON
    unit = tree_probs / norms
    cos = np.einsum("nic,njc->nij", unit, unit)
    mask = 1.0 - np.eye(K)
    return (cos * mask).sum(axis=-1) / (K - 1)


def tree_consensus(tree_probs: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
    """P_cons[n,c] = sum_k w_base[k] · tree_probs[n,k,c].

    Yêu cầu base_weights đã trên simplex (sum=1). Caller đảm bảo.
    """
    return np.einsum("k,nkc->nc", base_weights, tree_probs)


def dl_consensus_alignment(dl_probs: np.ndarray, consensus: np.ndarray) -> np.ndarray:
    """a[n,l] = cosine(dl_l, P_cons). Shape: (N, K_D)."""
    dl_n = np.linalg.norm(dl_probs, axis=-1, keepdims=True) + EPSILON
    cn_n = np.linalg.norm(consensus, axis=-1, keepdims=True) + EPSILON
    dl_unit = dl_probs / dl_n
    cons_unit = consensus / cn_n
    return np.einsum("nlc,nc->nl", dl_unit, cons_unit)


def consensus_penalty(
    tree_probs: np.ndarray, consensus: np.ndarray, gamma: float,
) -> np.ndarray:
    """penalty[n,k] = 1 nếu argmax tree_k == argmax P_cons; ngược lại = γ."""
    tree_arg = np.argmax(tree_probs, axis=-1)
    cons_arg = np.argmax(consensus, axis=-1)
    return np.where(tree_arg == cons_arg[:, None], 1.0, gamma)


def compute_tree_weights(
    c_tree: np.ndarray,
    consist: np.ndarray,
    penalty: np.ndarray,
    base_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """w_tree_raw[n,k] = α · c_tree · consist · penalty + (1−α) · w_base."""
    return alpha * c_tree * consist * penalty + (1 - alpha) * base_weights[None, :]


def compute_dl_weights(
    c_dl: np.ndarray, align_dl: np.ndarray, beta: float, tau: float,
) -> np.ndarray:
    """w_dl_raw[n,l] = β · max(c_dl · align_dl − τ, 0)."""
    return beta * np.maximum(c_dl * align_dl - tau, 0.0)


def normalize_weights(
    tw_raw: np.ndarray, dlw_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = tw_raw.sum(axis=1, keepdims=True) + dlw_raw.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        idx = np.where((total <= 0).ravel())[0]
        raise RuntimeError(
            f"MI5 degenerate: Σw_raw = 0 cho {len(idx)} sample(s) "
            f"(first idx: {idx[:5].tolist()}). "
            "Kiểm tra α (=1?), β (=0?), τ (quá lớn?), γ, hoặc w_base."
        )
    return tw_raw / total, dlw_raw / total


class MutualInferenceV5:
    """Ensemble v5 — consistency-based tree weight, consensus-aligned DL gate."""

    def __init__(
        self, cfg, weights: dict[str, float] | None = None,
        params: dict | None = None,
    ):
        self.cfg = cfg
        self.params = {**MI5_DEFAULTS, **(params or {})}
        if weights is None:
            n = len(cfg.all_targets) or 1
            weights = {t: 1.0 / n for t in cfg.all_targets}
        self.weights = dict(weights)

    def predict(
        self, preds: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tree_names = [n for n in self.cfg.tree_targets if n in preds]
        dl_names = [n for n in self.cfg.dl_targets if n in preds]
        if not tree_names:
            raise ValueError(f"No tree models in preds: {list(preds)}")

        tree_probs = np.stack([preds[n] for n in tree_names], axis=1)
        tree_base = np.array(
            [self.weights.get(n, 1.0 / len(tree_names)) for n in tree_names],
        )
        tree_base = tree_base / max(tree_base.sum(), EPSILON)

        if not dl_names:
            ensemble_probs = tree_consensus(tree_probs, tree_base)
        else:
            dl_probs = np.stack([preds[n] for n in dl_names], axis=1)
            alpha = self.params["alpha"]
            beta = self.params["beta"]
            tau = self.params["tau"]
            gamma = self.params["gamma"]

            c_tree = tree_model_confidence(tree_probs)
            c_dl = dl_model_confidence(dl_probs)
            consist = tree_internal_consistency(tree_probs)
            cons = tree_consensus(tree_probs, tree_base)
            align_dl = dl_consensus_alignment(dl_probs, cons)
            penalty = consensus_penalty(tree_probs, cons, gamma)

            tw_raw = compute_tree_weights(
                c_tree, consist, penalty, tree_base, alpha,
            )
            dlw_raw = compute_dl_weights(c_dl, align_dl, beta, tau)
            tw, dlw = normalize_weights(tw_raw, dlw_raw)

            ensemble_probs = (
                np.einsum("nk,nkc->nc", tw, tree_probs)
                + np.einsum("nl,nlc->nc", dlw, dl_probs)
            )

        assert np.allclose(ensemble_probs.sum(axis=1), 1.0, atol=1e-5), \
            "MI5: ensemble_probs không trên simplex — kiểm tra DL có softmax không."
        labels = np.argmax(ensemble_probs, axis=1)
        confidence = np.max(ensemble_probs, axis=1)
        return ensemble_probs, labels, confidence
