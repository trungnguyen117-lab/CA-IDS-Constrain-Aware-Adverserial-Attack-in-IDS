"""Base model interface: BaseModel, MLModel, DLModel."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common interface for tree and DL models."""

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        """Train the model. Returns self."""

    @abstractmethod
    def save(self, path: str):
        """Persist model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: str = "cpu"):
        """Load model from disk → new instance."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return integer class predictions."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return probability matrix (n_samples, n_classes)."""


class MLModel(BaseModel):
    """Shared helpers for sklearn-/CatBoost-/XGBoost-style estimators."""

    _model = None

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=np.float32)).astype(int)

    def predict_proba(self, X):
        return np.asarray(
            self._model.predict_proba(np.asarray(X, dtype=np.float32)),
            dtype=np.float64,
        )

    @staticmethod
    def art_predict_proba(art_clf, X):
        return art_clf.predict(X)


class DLModel(BaseModel):
    """Shared helpers for PyTorch DL models. Subclass exposes _net (with
    embedded InputNorm) + _device + _cfg. Operates on raw input space.
    """

    _net = None
    _device = "cpu"
    _cfg: dict | None = None

    def predict(self, X):
        import torch
        self._net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self._device)
            return torch.argmax(self._net(X_t), dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        self._net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self._device)
            return torch.softmax(self._net(X_t), dim=1).cpu().numpy().astype(np.float64)

    @staticmethod
    def art_predict_proba(art_clf, X):
        from scipy.special import softmax
        return softmax(art_clf.predict(X), axis=1)

    @property
    def net(self):
        return self._net

    @property
    def cfg(self):
        return self._cfg
