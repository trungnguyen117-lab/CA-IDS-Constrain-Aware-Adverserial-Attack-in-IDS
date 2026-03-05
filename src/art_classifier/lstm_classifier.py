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


# ── Scaled wrapper: bakes StandardScaler into forward pass ───────────────────

class _ScaledLSTM(nn.Module):
    """Wraps LSTMTabular with StandardScaler preprocessing inside the module.

    This lets ART operate in raw-feature space while the LSTM sees scaled input.
    """
    def __init__(self, lstm: LSTMTabular, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.lstm = lstm
        self.register_buffer('mean_', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('scale_', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sc = (x - self.mean_) / self.scale_
        return self.lstm(x_sc)


# ── ART Wrapper ───────────────────────────────────────────────────────────────

class LSTMWrapper(AdversarialWrapper):
    """ART-compatible wrapper for LSTMTabular trained in notebook 8.

    Handles StandardScaler internally so attack generators receive raw features.
    """

    def build_estimator(self) -> Any:
        from art.estimators.classification import PyTorchClassifier

        torch_model = self.model  # _ScaledLSTM instance
        loss = nn.CrossEntropyLoss()

        device_type = "gpu" if (self.device and (
            self.device.startswith("cuda") or self.device == "auto")) else "cpu"

        return PyTorchClassifier(
            model=torch_model,
            loss=loss,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            clip_values=self.clip_values,
            device_type=device_type,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass and apply softmax to get probability distribution."""
        estimator = self.get_estimator()
        X_f32 = X.astype(np.float32, copy=False)
        logits = estimator.predict(X_f32)   # ART returns raw logits for PyTorch
        # Softmax in numpy
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

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

        scaled_model = _ScaledLSTM(lstm, ckpt['scaler_mean'], ckpt['scaler_scale'])

        if device and device != "auto" and device.startswith("cuda"):
            scaled_model = scaled_model.cuda()
        else:
            scaled_model = scaled_model.cpu()
        scaled_model.eval()

        # Input dim from scaler mean shape (number of raw features)
        input_dim = int(ckpt['scaler_mean'].shape[0])

        return cls(
            model=scaled_model,
            num_classes=int(ckpt['n_classes']),
            input_shape=(input_dim,),
            clip_values=clip_values,
            device=device or "cpu",
        )
