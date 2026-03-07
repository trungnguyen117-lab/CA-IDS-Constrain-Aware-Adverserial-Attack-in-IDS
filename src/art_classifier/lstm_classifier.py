"""ART wrapper for LSTMTabular model (BiLSTM + Attention Pooling for tabular data)."""
import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .art_classifier import AdversarialWrapper


# ── Model architecture (mirrors notebook 8) ──────────────────────────────────

class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:  # H: [B, S, D]
        score = self.proj(H).squeeze(-1)           # [B, S]
        w = torch.softmax(score, dim=1)            # [B, S]
        return (H * w.unsqueeze(-1)).sum(1)        # [B, D]


class LSTMTabular(nn.Module):
    def __init__(self, step_dim: int, hidden: int = 256, layers: int = 2,
                 n_classes: int = 10, dropout: float = 0.2, bidir: bool = True):
        super().__init__()
        self.step_dim = step_dim
        self.in_norm = nn.LayerNorm(step_dim)
        self.lstm = nn.LSTM(
            input_size=step_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidir,
        )
        d_out = hidden * (2 if bidir else 1)
        self.pool = AttnPool(d_out)
        self.head = nn.Sequential(
            nn.Linear(d_out, d_out // 2), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(d_out // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, F]
        B, F = x.shape
        S = int(math.ceil(F / self.step_dim))
        pad = S * self.step_dim - F
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad), value=0.0)
        x = x.view(B, S, self.step_dim)
        x = self.in_norm(x)
        H, _ = self.lstm(x)
        z = self.pool(H)
        return self.head(z)


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
        """Load LSTM model from .pth saved by notebook 8.

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
