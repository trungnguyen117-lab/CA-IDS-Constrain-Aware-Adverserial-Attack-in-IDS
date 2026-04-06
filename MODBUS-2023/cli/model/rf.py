"""RandomForest model wrapper."""

import logging

import joblib
import numpy as np

from .base import TreeModel

logger = logging.getLogger(__name__)


class RandomForestModel(TreeModel):
    """RandomForest classifier for MODBUS-2023 pipeline."""

    DEFAULT_PARAMS = dict(
        n_estimators=850, max_leaf_nodes=15000,
        n_jobs=-1, bootstrap=True, criterion="entropy",
        class_weight="balanced", random_state=0,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        from sklearn.ensemble import RandomForestClassifier

        params = {**self.DEFAULT_PARAMS}
        if cfg:
            params.update(cfg)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logger.info(f"RF trained — n_estimators={model.n_estimators}")
        self._model = model
        return self

    def save(self, path):
        joblib.dump(self._model, path)
        logger.info(f"Saved RF → {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls()
        inst._model = joblib.load(path)
        logger.info(f"Loaded RF ← {path}")
        return inst

    def wrap_for_art(self, X_ref, clip_values=None, preprocessing_defences=None, **kwargs):
        from art.estimators.classification import SklearnClassifier

        if clip_values is None:
            clip_values = (X_ref.min(axis=0).astype(np.float32), X_ref.max(axis=0).astype(np.float32))
        return SklearnClassifier(
            model=self._model, clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
        )
