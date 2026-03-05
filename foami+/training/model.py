"""Base Model interface for foami+ training modules."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class Model(ABC):
    """Minimal training interface shared by all foami+ models."""

    def __init__(self, random_state: Optional[int] = 42) -> None:
        self.model: Any = None
        self.random_state: Optional[int] = random_state
        self._is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None

    @abstractmethod
    def save_model(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_model(cls, path: str, **kwargs) -> "Model":
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}
