"""CatBoost model wrapper."""

import logging

import joblib
import numpy as np

from .base import TreeModel

logger = logging.getLogger(__name__)


class CatBoostModel(TreeModel):
    """CatBoost classifier for FOAMI+ pipeline."""

    DEFAULT_PARAMS = dict(
        iterations=5000, depth=8, learning_rate=0.65,
        loss_function="MultiClass", eval_metric="TotalF1",
        od_type="Iter", od_wait=30, random_seed=42,
        task_type="CPU", thread_count=-1, verbose=False,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        from catboost import CatBoostClassifier

        params = {**self.DEFAULT_PARAMS}
        if cfg:
            params.update(cfg)
        params["classes_count"] = int(len(np.unique(y_train)))
        # Remove None values (e.g. od_type=None to disable early stopping)
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

    def wrap_for_art(self, X_ref, preprocessing_defences=None, **kwargs):
        from art.estimators.classification import CatBoostARTClassifier

        clip_values = (0.0, float(X_ref.max()))
        return CatBoostARTClassifier(
            model=self._model,
            clip_values=clip_values,
            nb_features=X_ref.shape[1],
            preprocessing_defences=preprocessing_defences,
        )
