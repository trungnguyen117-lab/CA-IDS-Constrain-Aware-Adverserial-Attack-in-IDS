import abc
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch.nn as nn

_logger = logging.getLogger(__name__)



class AdversarialWrapper(abc.ABC):
    """Common interface for adversarial robustness wrappers using ART.

    This wrapper adapts a trained model to ART's estimator interface and exposes
    convenience methods to generate adversarial samples and evaluate robustness.
    """

    def __init__(
        self,
        *,
        model: Any,
        num_classes: int,
        input_shape: Tuple[int, ...],
        clip_values: Tuple[float, float],
        device: str | None = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.num_classes = int(num_classes)
        self.input_shape = tuple(input_shape)
        self.clip_values = tuple(clip_values)
        self.device = device or "cpu"
        self.params = params.copy() if params else {}
        self._estimator = None  # Underlying ART estimator

    @abc.abstractmethod
    def build_estimator(self) -> Any:
        """Create and return the ART estimator for the underlying model."""

    def get_estimator(self) -> Any:
        if self._estimator is None:
            self._estimator = self.build_estimator()
        return self._estimator

    # Optional preprocessing defences (Feature Squeezing, Gaussian Augmentation)
    def _build_preprocessing_defences(self) -> Optional[list]:
        """Create preprocessing defences list based on params.

        Supported params:
          Feature Squeezing:
          - feature_squeezing: bool (enable/disable)
          - fs_bit_depth: int (default 8)
          - fs_apply_fit: bool (default True)
          - fs_apply_predict: bool (default True)

          Gaussian Augmentation:
          - gaussian_augmentation: bool (enable/disable)
          - ga_sigma: float (default 0.1)
          - ga_apply_fit: bool (default True)
          - ga_apply_predict: bool (default True)
        """
        defences: list = []

        # ── Feature Squeezing ──────────────────────────────────────────
        try:
            fs_enabled = bool(
                self.params.get("feature_squeezing", False)
                or ("fs_bit_depth" in self.params)
            )
            if fs_enabled:
                from art.defences.preprocessor import FeatureSqueezing  # type: ignore

                bit_depth = int(self.params.get(
                    "fs_bit_depth",
                    self.params.get("feature_squeezing_bit_depth", 8),
                ))
                fs = FeatureSqueezing(
                    clip_values=self.clip_values,
                    bit_depth=bit_depth,
                    apply_fit=bool(self.params.get("fs_apply_fit", True)),
                    apply_predict=bool(self.params.get("fs_apply_predict", True)),
                )
                defences.append(fs)
        except Exception:
            pass

        # ── Gaussian Augmentation ──────────────────────────────────────
        try:
            ga_enabled = bool(self.params.get("gaussian_augmentation", False))
            if ga_enabled:
                from art.defences.preprocessor import GaussianAugmentation  # type: ignore

                ga = GaussianAugmentation(
                    sigma=float(self.params.get("ga_sigma", 0.1)),
                    augmentation=False,
                    clip_values=self.clip_values,
                    apply_fit=bool(self.params.get("ga_apply_fit", True)),
                    apply_predict=bool(self.params.get("ga_apply_predict", True)),
                )
                defences.append(ga)
        except Exception:
            pass

        return defences if defences else None

    # Basic API
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass and apply softmax to get probability distribution.

        ART's PyTorchClassifier returns raw logits; softmax is applied here.
        Sklearn/CatBoost subclasses that already return probabilities should
        override this method and call estimator.predict() directly.
        """
        estimator = self.get_estimator()
        X_f32 = X.astype(np.float32, copy=False)
        logits = estimator.predict(X_f32)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        accuracy = float((y_pred == y_true).mean())
        return {"accuracy": accuracy}

    @staticmethod
    def _place_model(model: nn.Module, device: str) -> nn.Module:
        """Move *model* to the appropriate device."""
        if device and device != "auto" and device.startswith("cuda"):
            return model.cuda()
        return model.cpu()

    # Params API
    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)





