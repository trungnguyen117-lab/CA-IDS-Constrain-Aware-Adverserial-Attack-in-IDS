"""Abstract base class for all models in the MODBUS-2023 pipeline."""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common interface for tree and DL models."""

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        """Train the model. Returns self."""

    @abstractmethod
    def save(self, path):
        """Persist model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path, device="cpu"):
        """Load model from disk. Returns a new instance."""

    @abstractmethod
    def predict(self, X):
        """Return integer class predictions."""

    @abstractmethod
    def predict_proba(self, X):
        """Return probability matrix (n_samples, n_classes)."""

    @abstractmethod
    def wrap_for_art(self, X_ref, **kwargs):
        """Wrap as ART estimator for adversarial generation / evaluation."""


class TreeModel(BaseModel):
    """Shared helpers for joblib-based tree models."""

    _model = None

    def predict(self, X):
        return self._model.predict(X.astype(np.float32)).astype(int)

    def predict_proba(self, X):
        return np.asarray(self._model.predict_proba(X.astype(np.float32)),
                          dtype=np.float64)


class DLModel(BaseModel):
    """Shared helpers for PyTorch-based DL models."""

    _net = None
    _scaler = None  # QuantileTransformer or StandardScaler
    _cfg = None

    def predict(self, X):
        import torch
        X_sc = self._scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(X_sc).to(self._device))
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        X_sc = self._scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(X_sc).to(self._device))
            return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)

    @property
    def scaler(self):
        return self._scaler

    @property
    def net(self):
        return self._net

    @property
    def cfg(self):
        return self._cfg
