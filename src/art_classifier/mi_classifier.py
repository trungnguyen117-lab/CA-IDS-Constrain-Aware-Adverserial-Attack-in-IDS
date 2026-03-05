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


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


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

    # ── Core MI logic (vectorised numpy) ─────────────────────────────────────

    def _mi_predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_f32 = x.astype(np.float32, copy=False)

        # Collect GBT predictions → P_gbt: (N, n_gbt, C)
        P_gbt = np.stack(
            [w.predict_proba(x_f32) for w in self.gbt_wrappers.values()],
            axis=1,
        )
        # Collect DL predictions → P_dl: (N, n_dl, C)
        P_dl = np.stack(
            [w.predict_proba(x_f32) for w in self.dl_wrappers.values()],
            axis=1,
        )

        N = x.shape[0]

        # ── GBT confidence: prediction margin (top1 - top2) ──────────────────
        P_gbt_sorted = np.sort(P_gbt, axis=2)                  # (N, n_gbt, C)
        C_gbt = P_gbt_sorted[:, :, -1] - P_gbt_sorted[:, :, -2]  # (N, n_gbt)
        C_gbt_std = (C_gbt - C_gbt.mean(axis=0, keepdims=True)) / (
            C_gbt.std(axis=0, keepdims=True) + 1e-10)

        # ── DL confidence: 1 - normalised entropy ────────────────────────────
        P_dl_clip = np.clip(P_dl, 1e-10, 1.0)
        H_dl = -np.sum(P_dl_clip * np.log(P_dl_clip), axis=2)  # (N, n_dl)
        C_dl = 1.0 - H_dl / np.log(self.num_classes + 1e-10)
        C_dl_std = (C_dl - C_dl.mean(axis=0, keepdims=True)) / (
            C_dl.std(axis=0, keepdims=True) + 1e-10)

        # ── Agreement: cosine similarity between GBT and DL distributions ────
        norm_gbt = np.linalg.norm(P_gbt, axis=2, keepdims=True) + 1e-10  # (N,ng,1)
        norm_dl  = np.linalg.norm(P_dl,  axis=2, keepdims=True) + 1e-10  # (N,nd,1)
        A = np.einsum('nik,njk->nij',
                      P_gbt / norm_gbt,
                      P_dl  / norm_dl)           # (N, n_gbt, n_dl)
        A_mean_row = A.mean(axis=2)              # (N, n_gbt) GBT-DL agreement per GBT model
        A_mean_col = A.mean(axis=1)              # (N, n_dl)  DL-GBT agreement per DL model

        # ── GBT internal disagreement ─────────────────────────────────────────
        gbt_labels  = P_gbt.argmax(axis=2)      # (N, n_gbt)
        gbt_agree   = np.all(gbt_labels == gbt_labels[:, :1], axis=1).astype(float)
        gbt_disagree = 1.0 - gbt_agree          # (N,)

        # ── Weight computation ────────────────────────────────────────────────
        # GBT weights with confidence modulation
        w_gbt_raw = (self.alpha * C_gbt_std * A_mean_row
                     + (1.0 - self.alpha) * self.w_gbt_base[None, :])  # (N, n_gbt)
        w_gbt_raw = np.clip(w_gbt_raw, 0.0, None)

        # DL weights with sparse activation
        dl_signal = C_dl_std * A_mean_col * (1.0 + gbt_disagree[:, None])  # (N, n_dl)
        dl_active = np.maximum(dl_signal - self.threshold, 0.0)
        w_dl_raw  = self.beta * dl_active                                    # (N, n_dl)

        # Normalise
        total = (w_gbt_raw.sum(axis=1, keepdims=True)
                 + w_dl_raw.sum(axis=1, keepdims=True))
        total = np.maximum(total, 1e-10)
        w_gbt = w_gbt_raw / total
        w_dl  = w_dl_raw  / total

        # ── Final ensemble prediction ─────────────────────────────────────────
        p_ens = (np.einsum('nk,nkc->nc', w_gbt, P_gbt)
                 + np.einsum('nl,nlc->nc', w_dl,  P_dl))
        return p_ens.astype(np.float32)

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
