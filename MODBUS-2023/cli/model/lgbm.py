"""LightGBM model wrapper."""

import logging

import joblib
import numpy as np

from .base import TreeModel

logger = logging.getLogger(__name__)


class LightGBMModel(TreeModel):
    """LightGBM classifier for MODBUS-2023 pipeline."""

    DEFAULT_PARAMS = dict(
        boosting_type="gbdt", objective="multiclass",
        learning_rate=0.15, max_depth=5, n_estimators=3000,
        class_weight="balanced",
        metric="multi_logloss",
        n_jobs=-1, random_state=42, verbose=-1,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        from lightgbm import LGBMClassifier, early_stopping

        params = {**self.DEFAULT_PARAMS}
        if cfg:
            params.update(cfg)

        early_rounds = params.pop("early_stopping_rounds", 50)
        params = {k: v for k, v in params.items() if v is not None}

        model = LGBMClassifier(**params)
        callbacks = [early_stopping(early_rounds)] if X_val is not None else None
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)] if X_val is not None else None,
                  callbacks=callbacks)
        logger.info(f"LGBM trained — best iteration: {model.best_iteration_}")
        self._model = model
        return self

    def save(self, path):
        joblib.dump(self._model, path)
        logger.info(f"Saved LGBM → {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls()
        inst._model = joblib.load(path)
        logger.info(f"Loaded LGBM ← {path}")
        return inst

    def wrap_for_art(self, X_ref, clip_values=None, preprocessing_defences=None, **kwargs):
        from art.estimators.classification.scikitlearn import ScikitlearnClassifier

        if clip_values is None:
            clip_values = (X_ref.min(axis=0).astype(np.float32), X_ref.max(axis=0).astype(np.float32))
        return ScikitlearnClassifier(
            model=self._model, clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
        )
