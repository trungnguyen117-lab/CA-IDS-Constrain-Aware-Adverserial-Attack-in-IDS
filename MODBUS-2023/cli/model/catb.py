"""CatBoost model wrapper."""

import logging

import joblib
import numpy as np

from .base import TreeModel

logger = logging.getLogger(__name__)


class CatBoostModel(TreeModel):
    """CatBoost classifier for MODBUS-2023 pipeline."""

    DEFAULT_PARAMS = dict(
        iterations=3000, depth=5, learning_rate=0.2,
        loss_function="MultiClass", eval_metric="MultiClass",
        auto_class_weights="Balanced",
        early_stopping_rounds=50, random_seed=42,
        task_type="CPU", thread_count=-1, verbose=False,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        from catboost import CatBoostClassifier

        params = {**self.DEFAULT_PARAMS}
        if cfg:
            params.update(cfg)
        params["classes_count"] = int(len(np.unique(y_train)))
        params = {k: v for k, v in params.items() if v is not None}

        model = CatBoostClassifier(**params)
        eval_set = (X_val, y_val) if X_val is not None else None
        model.fit(X_train, y_train, eval_set=eval_set,
                  use_best_model=eval_set is not None)
        logger.info(f"CatBoost trained — iterations used: {model.tree_count_}")
        self._model = model
        return self

    def save(self, path):
        joblib.dump(self._model, path)
        logger.info(f"Saved CatBoost → {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls()
        inst._model = joblib.load(path)
        logger.info(f"Loaded CatBoost ← {path}")
        return inst

    def wrap_for_art(self, X_ref, clip_values=None, preprocessing_defences=None, **kwargs):
        from art.estimators.classification import CatBoostARTClassifier

        if clip_values is None:
            clip_values = (X_ref.min(axis=0).astype(np.float32), X_ref.max(axis=0).astype(np.float32))
        return CatBoostARTClassifier(
            model=self._model,
            clip_values=clip_values,
            nb_features=X_ref.shape[1],
            preprocessing_defences=preprocessing_defences,
        )
