"""RandomForest training module — save/load via joblib."""
from __future__ import annotations

import logging
from typing import Optional, Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .model import Model

logger = logging.getLogger(__name__)


class RFModel(Model):
    """RandomForest classifier with joblib-based persistence.

    joblib is the sklearn-recommended serialization format; it handles
    embedded numpy arrays more efficiently than plain pickle and is
    consistent with how sklearn models are conventionally saved.
    """

    def __init__(
        self,
        *,
        num_class: int,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.num_class = int(num_class)
        kwargs: Dict[str, Any] = dict(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=int(random_state),
        )
        if params:
            kwargs.update(params)
        self.model = RandomForestClassifier(**kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        # RandomForest does not use a validation set during training
        self.model.fit(X_train, y_train)
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
        """Save fitted RandomForest model with joblib.

        joblib is preferred over pickle for sklearn models because it uses
        numpy memory-mapped IO for large arrays, resulting in smaller files
        and faster load times.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        joblib.dump(self.model, path)
        logger.info(f"[RF] Saved → {path}")

    @classmethod
    def load_model(
        cls,
        path: str,
        *,
        num_class: Optional[int] = None,
        **kwargs,
    ) -> "RFModel":
        """Load a RandomForest model saved with joblib."""
        mdl: RandomForestClassifier = joblib.load(path)
        inferred = int(len(getattr(mdl, 'classes_', [])) or num_class or 2)
        inst = cls(num_class=num_class or inferred)
        inst.model = mdl
        inst._is_fitted = True
        logger.info(f"[RF] Loaded ← {path}")
        return inst
