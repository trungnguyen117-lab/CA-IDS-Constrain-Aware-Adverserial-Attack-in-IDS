"""ART wrapper for ResDNN model (Residual DNN for tabular data).

Architecture lives in foami-ext/model/resdnn.py (_ResDNN).
This module only contains the ART wrapper logic.
"""
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .art_classifier import AdversarialWrapper
from model.resdnn import _ResDNN as ResDNN  # single source of truth


# ── ART Wrapper ───────────────────────────────────────────────────────────────

class ResDNNWrapper(AdversarialWrapper):
    """ART-compatible wrapper for ResDNN trained in notebook 8.

    Normalization is handled via ART's built-in preprocessing parameter so that
    gradient-based attacks operate correctly in raw feature space.
    """

    def build_estimator(self) -> Any:
        from art.estimators.classification import PyTorchClassifier

        device_type = "gpu" if (self.device and (
            self.device.startswith("cuda") or self.device == "auto")) else "cpu"

        preprocessing = None
        if hasattr(self, 'scaler_mean') and self.scaler_mean is not None:
            preprocessing = (self.scaler_mean, self.scaler_scale)

        return PyTorchClassifier(
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            clip_values=self.clip_values,
            preprocessing=preprocessing,
            device_type=device_type,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        clip_values: Tuple[float, float],
        device: Optional[str] = None,
    ) -> "ResDNNWrapper":
        """Load ResDNN model from .pth saved by notebook 8 / ResDNNModel.save_model().

        Expected keys in checkpoint:
          state_dict, in_dim, n_classes, scaler_mean, scaler_scale
        """
        map_location = device if device and device != "auto" else "cpu"
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        resdnn = ResDNN(in_dim=int(ckpt['in_dim']), n_classes=int(ckpt['n_classes']))
        resdnn.load_state_dict(ckpt['state_dict'])
        resdnn.eval()
        resdnn = cls._place_model(resdnn, device)

        input_dim = int(ckpt['in_dim'])

        wrapper = cls(
            model=resdnn,
            num_classes=int(ckpt['n_classes']),
            input_shape=(input_dim,),
            clip_values=clip_values,
            device=device or "cpu",
        )
        wrapper.scaler_mean  = ckpt['scaler_mean'].astype(np.float32)
        wrapper.scaler_scale = ckpt['scaler_scale'].astype(np.float32)
        return wrapper
