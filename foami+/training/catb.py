"""CatBoost training module — save/load via joblib."""
from __future__ import annotations

import logging
from typing import Optional, Any, Dict

import joblib
import numpy as np
from catboost import CatBoostClassifier

from .model import Model

logger = logging.getLogger(__name__)


class CatBoostModel(Model):
    """CatBoost classifier with joblib-based persistence."""

    def __init__(
        self,
        *,
        num_class: int,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.num_class = int(num_class)
        base: Dict[str, Any] = {
            'loss_function': 'MultiClass',
            'random_seed': int(random_state),
            'verbose': False,
        }
        if params:
            base.update(params)
        self.model = CatBoostClassifier(**base)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Save fitted CatBoost model with joblib."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        joblib.dump(self.model, path)
        logger.info(f"[CatBoost] Saved → {path}")

    @classmethod
    def load_model(
        cls,
        path: str,
        *,
        num_class: Optional[int] = None,
        **kwargs,
    ) -> "CatBoostModel":
        """Load a CatBoost model saved with joblib."""
        mdl: CatBoostClassifier = joblib.load(path)
        inferred = int(len(getattr(mdl, 'classes_', [])) or num_class or 2)
        inst = cls(num_class=num_class or inferred)
        inst.model = mdl
        inst._is_fitted = True
        logger.info(f"[CatBoost] Loaded ← {path}")
        return inst
