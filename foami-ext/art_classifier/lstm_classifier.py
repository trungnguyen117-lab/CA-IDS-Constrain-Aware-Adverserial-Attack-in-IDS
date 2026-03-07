"""ART wrapper for LSTMTabular model (BiLSTM + Attention Pooling for tabular data).

Architecture lives in foami-ext/model/lstm.py (_LSTMTabular).
This module only contains the ART wrapper logic.
"""
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .art_classifier import AdversarialWrapper
from model.lstm import _LSTMTabular as LSTMTabular  # single source of truth


# ── ART Wrapper ───────────────────────────────────────────────────────────────

class LSTMWrapper(AdversarialWrapper):
    """ART-compatible wrapper for LSTMTabular trained in notebook 8.

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
    ) -> "LSTMWrapper":
        """Load LSTM model from .pth saved by notebook 8 / LSTMModel.save_model().

        Expected keys in checkpoint:
          state_dict, step_dim, hidden, layers, n_classes, dropout, bidir,
          scaler_mean, scaler_scale
        """
        map_location = device if device and device != "auto" else "cpu"
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        lstm = LSTMTabular(
            step_dim=int(ckpt['step_dim']),
            hidden=int(ckpt['hidden']),
            layers=int(ckpt['layers']),
            n_classes=int(ckpt['n_classes']),
            dropout=float(ckpt['dropout']),
            bidir=bool(ckpt['bidir']),
        )
        lstm.load_state_dict(ckpt['state_dict'])
        lstm.eval()
        lstm = cls._place_model(lstm, device)

        input_dim = int(ckpt['scaler_mean'].shape[0])

        wrapper = cls(
            model=lstm,
            num_classes=int(ckpt['n_classes']),
            input_shape=(input_dim,),
            clip_values=clip_values,
            device=device or "cpu",
        )
        wrapper.scaler_mean  = ckpt['scaler_mean'].astype(np.float32)
        wrapper.scaler_scale = ckpt['scaler_scale'].astype(np.float32)
        return wrapper
