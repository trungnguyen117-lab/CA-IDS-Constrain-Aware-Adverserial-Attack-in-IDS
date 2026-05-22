"""Extra Trees tree model."""

from __future__ import annotations

import logging

import joblib

from ..base import MLModel

logger = logging.getLogger(__name__)


class ExtraTreesModel(MLModel):
    DEFAULT_PARAMS = dict(
        n_estimators=200, max_leaf_nodes=15000, criterion="entropy",
        class_weight="balanced", n_jobs=-1, bootstrap=True, random_state=0,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        from sklearn.ensemble import ExtraTreesClassifier
        params = {**self.DEFAULT_PARAMS, **(cfg or {})}
        m = ExtraTreesClassifier(**params)
        m.fit(X_train, y_train)
        logger.info("ET trained — n_estimators=%d", m.n_estimators)
        self._model = m
        return self

    def save(self, path):
        joblib.dump(self._model, path)
        logger.info("Saved ET → %s", path)

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls(); inst._model = joblib.load(path); return inst
