"""ART classifier wrap + clip values helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as torch_nn

from art.estimators.classification import PyTorchClassifier, SklearnClassifier
from art.estimators.classification.catboost import CatBoostARTClassifier
from art.estimators.classification.lightgbm import LightGBMClassifier
from art.estimators.classification.xgboost import XGBoostClassifier

def get_clip_values(X_ref: np.ndarray, attack: str | None = None):
    """Pick clip_values for ART based on attack constraint.

    CW unbounded perturbations can blow features out of natural range, so we
    give it per-feature (min, max) arrays. Other attacks keep a global scalar
    range to stay compatible with their eps semantics.
    """
    if attack in ("cw", "deepfool"):
        return X_ref.min(axis=0).astype(np.float32), X_ref.max(axis=0).astype(np.float32)
    return float(X_ref.min()), float(X_ref.max())


def wrap_for_art(cfg, model, target: str, X_ref: np.ndarray,
                 device: str = "cpu", attack: str | None = None,
                 preprocessing_defences=None,
                 clip: bool = True):
    """Wrap a trained model into an ART classifier.

    Dispatch by ``target`` name (explicit per-library wrapping):
      - DL (mlp/dnn/lstm/resdnn/...) → :class:`PyTorchClassifier` (gradients).
      - ``cat`` → :class:`CatBoostARTClassifier`.
      - ``xgb`` → :class:`XGBoostClassifier`.
      - ``lgbm`` → :class:`LightGBMClassifier` wrapping the underlying Booster.
      - any other tree (``bme``, ``rf``, ``et``, …) → :class:`SklearnClassifier`.

    ``X_ref`` is RAW features. DL models embed ``InputNorm`` inside
    ``model.net`` so ART feeds raw inputs through. ``clip_values`` follows
    :func:`get_clip_values`.
    """
    clip_values = get_clip_values(X_ref, attack=attack) if clip else None
    n_features = X_ref.shape[1]
    common = dict(clip_values=clip_values, preprocessing_defences=preprocessing_defences)
    if cfg.is_dl(target):
        net = model.net
        net.eval()
        return PyTorchClassifier(
            model=net,
            loss=torch_nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(net.parameters()),
            input_shape=(n_features,),
            nb_classes=cfg.n_classes,
            device_type="gpu" if device.startswith("cuda") else "cpu",
            **common,
        )

    estimator = model._model
    if target == "cat":
        return CatBoostARTClassifier(model=estimator, nb_features=n_features, **common)
    if target == "xgb":
        return XGBoostClassifier(
            model=estimator, nb_features=n_features,
            nb_classes=int(len(estimator.classes_)), **common,
        )
    if target == "lgbm":
        return LightGBMClassifier(model=estimator.booster_, **common)
    return SklearnClassifier(model=estimator, **common)
