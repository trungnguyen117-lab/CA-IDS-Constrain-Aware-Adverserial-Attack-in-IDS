"""Random Forest tree model."""

from __future__ import annotations

import logging

import joblib

from ..base import MLModel

logger = logging.getLogger(__name__)


class RandomForestModel(MLModel):
    DEFAULT_PARAMS = dict(
        n_estimators=850, max_leaf_nodes=15000, n_jobs=-1, bootstrap=True,
        criterion="entropy", class_weight="balanced", random_state=0,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        from sklearn.ensemble import RandomForestClassifier
        params = {**self.DEFAULT_PARAMS, **(cfg or {})}
        m = RandomForestClassifier(**params)
        m.fit(X_train, y_train)
        logger.info("RF trained — n_estimators=%d, n_classes=%d",
                    m.n_estimators, len(m.classes_))
        self._model = m
        return self

    def save(self, path):
        joblib.dump(self._model, path)
        logger.info("Saved RF → %s", path)

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls(); inst._model = joblib.load(path); return inst
