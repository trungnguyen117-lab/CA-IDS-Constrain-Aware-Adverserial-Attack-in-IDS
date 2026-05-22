"""XGBoost tree model."""

from __future__ import annotations

import logging

import joblib
import numpy as np

from ..base import MLModel

logger = logging.getLogger(__name__)


class XGBoostModel(MLModel):
    DEFAULT_PARAMS = dict(
        max_depth=15, n_estimators=3000, learning_rate=0.2,
        objective="multi:softprob", eval_metric="mlogloss",
        booster="gbtree", tree_method="hist",
        n_jobs=-1, random_state=42, early_stopping_rounds=50,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu",
              sample_weight=None, sample_weight_eval=None):
        from xgboost import XGBClassifier
        params = {**self.DEFAULT_PARAMS, **(cfg or {})}
        params["num_class"] = int(len(np.unique(y_train)))
        if X_val is None:
            params.pop("early_stopping_rounds", None)
        params = {k: v for k, v in params.items() if v is not None}
        m = XGBClassifier(**params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        eval_sw = [sample_weight_eval] if (sample_weight_eval is not None and eval_set) else None
        m.fit(X_train, y_train,
              sample_weight=sample_weight,
              eval_set=eval_set,
              sample_weight_eval_set=eval_sw,
              verbose=False)
        logger.info("XGBoost trained — best_iter=%s", getattr(m, "best_iteration", None))
        self._model = m
        return self

    def save(self, path):
        joblib.dump(self._model, path)
        logger.info("Saved XGBoost → %s", path)

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls(); inst._model = joblib.load(path); return inst
