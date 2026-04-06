"""Abstract base class for all models in the FOAMI+ pipeline."""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common interface for tree and DL models.

    Subclasses must implement: train, save, load, predict, wrap_for_art.
    """

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
        """Wrap as ART estimator for adversarial generation / evaluation.

        X_ref is used to compute clip_values.
        """


class TreeModel(BaseModel):
    """Shared helpers for joblib-based tree models (CatBoost, RF)."""

    _model = None  # underlying sklearn/catboost estimator

    def predict(self, X):
        return self._model.predict(X.astype(np.float32)).astype(int)

    def predict_proba(self, X):
        return np.asarray(self._model.predict_proba(X.astype(np.float32)),
                          dtype=np.float64)

    def art_predict_proba(self, art_clf, X):
        """ART tree classifiers already return probabilities."""
        return art_clf.predict(X)


class DLModel(BaseModel):
    """Shared helpers for PyTorch-based DL models (LSTM, ResDNN).

    InputNorm is embedded in the model architecture — no external scaler needed.
    Model accepts raw input directly.
    """

    _net = None     # nn.Module (with InputNorm layer)
    _cfg = None     # hyperparameters dict

    def predict(self, X):
        import torch
        X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)
        with torch.no_grad():
            logits = self._net(X_t)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        X_t = torch.from_numpy(X.astype(np.float32)).to(self._device)
        with torch.no_grad():
            logits = self._net(X_t)
            return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)

    def art_predict_proba(self, art_clf, X):
        """ART PyTorchClassifier returns logits; convert to probabilities."""
        from scipy.special import softmax
        return softmax(art_clf.predict(X), axis=1)

    @property
    def net(self):
        return self._net

    @property
    def cfg(self):
        return self._cfg
