"""CatBoost tree model."""

from __future__ import annotations

import logging

import joblib
import numpy as np

from ..base import MLModel

logger = logging.getLogger(__name__)


class CatBoostModel(MLModel):
    DEFAULT_PARAMS = dict(
        iterations=3000, depth=5, learning_rate=0.2,
        loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced",
        early_stopping_rounds=50, random_seed=42,
        task_type="CPU", thread_count=-1, verbose=False,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        from catboost import CatBoostClassifier
        params = {**self.DEFAULT_PARAMS, **(cfg or {})}
        params["classes_count"] = int(len(np.unique(y_train)))
        params = {k: v for k, v in params.items() if v is not None}
        m = CatBoostClassifier(**params)
        eval_set = (X_val, y_val) if X_val is not None else None
        m.fit(X_train, y_train, eval_set=eval_set, use_best_model=eval_set is not None)
        logger.info("CatBoost trained — iters used=%d", m.tree_count_)
        self._model = m
        return self

    def save(self, path):
        joblib.dump(self._model, path)
        logger.info("Saved CatBoost → %s", path)

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls(); inst._model = joblib.load(path); return inst
