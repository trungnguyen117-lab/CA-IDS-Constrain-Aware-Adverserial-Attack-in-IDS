"""ART wrapper for ResDNN model (Residual DNN for tabular data)."""
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .art_classifier import AdversarialWrapper


# ── Model architecture (mirrors notebook 8) ──────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_hid: int, p: float = 0.25):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hid)
        self.bn1  = nn.BatchNorm1d(d_hid)
        self.lin2 = nn.Linear(d_hid, d_in)
        self.ln2  = nn.LayerNorm(d_in)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch.relu(self.bn1(self.lin1(x))))
        h = self.lin2(h)
        return torch.relu(self.ln2(x + h))


class ResDNN(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        W = 512
        self.stem   = nn.Sequential(
            nn.Linear(in_dim, W), nn.BatchNorm1d(W), nn.ReLU(), nn.Dropout(0.30)
        )
        self.block1 = ResidualBlock(W, W // 2, p=0.30)
        self.block2 = ResidualBlock(W, W // 2, p=0.25)
        self.block3 = ResidualBlock(W, W // 2, p=0.20)
        self.head   = nn.Linear(W, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        return self.head(h)


# ── Scaled wrapper ────────────────────────────────────────────────────────────

class _ScaledResDNN(nn.Module):
    """Wraps ResDNN with StandardScaler preprocessing inside the module."""

    def __init__(self, resdnn: ResDNN, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.resdnn = resdnn
        self.register_buffer('mean_', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('scale_', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sc = (x - self.mean_) / self.scale_
        return self.resdnn(x_sc)


# ── ART Wrapper ───────────────────────────────────────────────────────────────

class ResDNNWrapper(AdversarialWrapper):
    """ART-compatible wrapper for ResDNN trained in notebook 8.

    Handles StandardScaler internally so attack generators receive raw features.
    """

    def build_estimator(self) -> Any:
        from art.estimators.classification import PyTorchClassifier

        loss = nn.CrossEntropyLoss()
        device_type = "gpu" if (self.device and (
            self.device.startswith("cuda") or self.device == "auto")) else "cpu"

        return PyTorchClassifier(
            model=self.model,           # _ScaledResDNN instance
            loss=loss,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            clip_values=self.clip_values,
            device_type=device_type,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run forward pass and apply softmax."""
        estimator = self.get_estimator()
        X_f32 = X.astype(np.float32, copy=False)
        logits = estimator.predict(X_f32)
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
    ) -> "ResDNNWrapper":
        """Load ResDNN model from .pth saved by notebook 8.

        Expected keys in checkpoint:
          state_dict, in_dim, n_classes, scaler_mean, scaler_scale
        """
        map_location = device if device and device != "auto" else "cpu"
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        resdnn = ResDNN(in_dim=int(ckpt['in_dim']), n_classes=int(ckpt['n_classes']))
        resdnn.load_state_dict(ckpt['state_dict'])
        resdnn.eval()

        scaled_model = _ScaledResDNN(resdnn, ckpt['scaler_mean'], ckpt['scaler_scale'])

        if device and device != "auto" and device.startswith("cuda"):
            scaled_model = scaled_model.cuda()
        else:
            scaled_model = scaled_model.cpu()
        scaled_model.eval()

        input_dim = int(ckpt['in_dim'])

        return cls(
            model=scaled_model,
            num_classes=int(ckpt['n_classes']),
            input_shape=(input_dim,),
            clip_values=clip_values,
            device=device or "cpu",
        )
