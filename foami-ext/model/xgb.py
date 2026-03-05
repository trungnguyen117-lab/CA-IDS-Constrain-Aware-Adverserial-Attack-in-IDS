"""XGBoost training module — save/load via joblib."""
from __future__ import annotations

import logging
from typing import Optional, Any, Dict

import joblib
import numpy as np
from xgboost import XGBClassifier

from .model import Model

logger = logging.getLogger(__name__)


class XGBModel(Model):
    """XGBoost classifier (sklearn API) with joblib-based persistence."""

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
            'objective':     'multi:softmax',
            'num_class':     self.num_class,
            'eval_metric':   'mlogloss',
            'tree_method':   'hist',
            'max_depth':     15,
            'n_estimators':  5000,
            'learning_rate': 0.3,
            'booster':       'gbtree',
            'random_state':  int(random_state),
        }
        if params:
            base.update(params)
        self.model = XGBClassifier(**base)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        fit_kwargs: Dict[str, Any] = {'verbose': False}
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
        self.model.fit(X_train, y_train, **fit_kwargs)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        joblib.dump(self.model, path)
        logger.info(f"[XGB] Saved → {path}")

    @classmethod
    def load_model(
        cls,
        path: str,
        *,
        num_class: Optional[int] = None,
        **kwargs,
    ) -> "XGBModel":
        mdl: XGBClassifier = joblib.load(path)
        inferred = int(len(getattr(mdl, 'classes_', [])) or num_class or 2)
        inst = cls(num_class=num_class or inferred)
        inst.model = mdl
        inst._is_fitted = True
        logger.info(f"[XGB] Loaded ← {path}")
        return inst
