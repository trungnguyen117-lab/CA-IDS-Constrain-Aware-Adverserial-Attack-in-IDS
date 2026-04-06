"""XGBoost model wrapper."""

import logging

import joblib
import numpy as np

from .base import TreeModel

logger = logging.getLogger(__name__)


class XGBoostModel(TreeModel):
    """XGBoost classifier for MODBUS-2023 pipeline."""

    DEFAULT_PARAMS = dict(
        max_depth=15, n_estimators=5000, learning_rate=0.25,
        objective="multi:softprob", eval_metric="mlogloss",
        booster="gbtree", tree_method="hist",
        n_jobs=-1, random_state=42, early_stopping_rounds=50,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        from xgboost import XGBClassifier

        params = {**self.DEFAULT_PARAMS}
        if cfg:
            params.update(cfg)
        params["num_class"] = int(len(np.unique(y_train)))
        params = {k: v for k, v in params.items() if v is not None}

        model = XGBClassifier(**params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        logger.info(f"XGBoost trained — best iteration: {model.best_iteration}")
        self._model = model
        return self

    def save(self, path):
        joblib.dump(self._model, path)
        logger.info(f"Saved XGBoost → {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls()
        inst._model = joblib.load(path)
        logger.info(f"Loaded XGBoost ← {path}")
        return inst

    def wrap_for_art(self, X_ref, clip_values=None, preprocessing_defences=None, **kwargs):
        from art.estimators.classification import XGBoostClassifier

        if clip_values is None:
            clip_values = (X_ref.min(axis=0).astype(np.float32), X_ref.max(axis=0).astype(np.float32))
        nb_classes = len(self._model.classes_)
        return XGBoostClassifier(
            model=self._model,
            clip_values=clip_values,
            nb_classes=nb_classes,
            nb_features=X_ref.shape[1],
            preprocessing_defences=preprocessing_defences,
        )
