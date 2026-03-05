"""LSTM training module — architecture mirrors notebook 8, save/load via torch.save.

Checkpoint format is intentionally compatible with LSTMWrapper.from_checkpoint()
in src/art_classifier/lstm_classifier.py so the same .pth file can be used for
both inference and adversarial generation without conversion.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .model import Model

logger = logging.getLogger(__name__)


# ── Architecture (mirrors notebook 8 / lstm_classifier.py) ───────────────────

class _AttnPool(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:  # H: [B, S, D]
        w = torch.softmax(self.proj(H).squeeze(-1), dim=1)  # [B, S]
        return (H * w.unsqueeze(-1)).sum(1)                  # [B, D]


class _LSTMTabular(nn.Module):
    def __init__(
        self,
        step_dim: int,
        hidden: int = 256,
        layers: int = 2,
        n_classes: int = 12,
        dropout: float = 0.15,
        bidir: bool = True,
    ) -> None:
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
        self.pool = _AttnPool(d_out)
        self.head = nn.Sequential(
            nn.Linear(d_out, d_out // 2), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(d_out // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, F]
        B, F = x.shape
        S = int(math.ceil(F / self.step_dim))
        pad = S * self.step_dim - F
        if pad > 0:
            x = nn.functional.pad(x, (0, pad))
        x = self.in_norm(x.view(B, S, self.step_dim))
        H, _ = self.lstm(x)
        return self.head(self.pool(H))


# ── Training wrapper ──────────────────────────────────────────────────────────

class LSTMModel(Model):
    """BiLSTM + Attention for tabular classification.

    Preprocessing: StandardScaler is fitted during `fit()` and baked into the
    checkpoint so the model is self-contained at inference time.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_class: int,
        step_dim: int = 16,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.15,
        bidir: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 2e-4,
        batch_size: int = 2048,
        max_epochs: int = 120,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.input_dim = int(input_dim)
        self.num_class = int(num_class)
        self.step_dim = int(step_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.bidir = bool(bidir)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)

        self.device = torch.device(
            device if device and device != 'auto'
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        torch.manual_seed(random_state)

        self.model = _LSTMTabular(
            step_dim=self.step_dim,
            hidden=self.hidden,
            layers=self.layers,
            n_classes=self.num_class,
            dropout=self.dropout,
            bidir=self.bidir,
        ).to(self.device)

        self._scaler: Optional[StandardScaler] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self._scaler = StandardScaler()
        X_tr = self._scaler.fit_transform(X_train).astype(np.float32)
        X_v = self._scaler.transform(X_val).astype(np.float32) if X_val is not None else None

        def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                              pin_memory=self.device.type == 'cuda')

        train_loader = _loader(X_tr, y_train, shuffle=True)
        val_loader = _loader(X_v, y_val, shuffle=False) if X_v is not None and y_val is not None else None

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_loss = float('inf')
        best_state: Optional[dict] = None
        wait = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                criterion(self.model(xb), yb).backward()
                optimizer.step()

            if val_loader is None:
                continue

            self.model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += criterion(self.model(xb), yb).item() * yb.size(0)
                    n += yb.size(0)
            val_loss /= max(1, n)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info(f"[LSTM] Early stop at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._is_fitted = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Scaler not fitted — call fit() or load_model() first")
        return self._scaler.transform(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        self.model.eval()
        X_sc = torch.from_numpy(self._scale(X)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_sc).cpu().numpy()
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Save checkpoint compatible with LSTMWrapper.from_checkpoint().

        Keys: state_dict, step_dim, hidden, layers, n_classes, dropout,
              bidir, scaler_mean, scaler_scale
        """
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError("Model not fitted")
        torch.save({
            'state_dict':   self.model.state_dict(),
            'step_dim':     self.step_dim,
            'hidden':       self.hidden,
            'layers':       self.layers,
            'n_classes':    self.num_class,
            'dropout':      self.dropout,
            'bidir':        self.bidir,
            'scaler_mean':  self._scaler.mean_,
            'scaler_scale': self._scaler.scale_,
        }, path)
        logger.info(f"[LSTM] Saved → {path}")

    @classmethod
    def load_model(
        cls,
        path: str,
        *,
        device: Optional[str] = None,
        **kwargs,
    ) -> "LSTMModel":
        """Load checkpoint saved by save_model() or notebook 8."""
        map_loc = device if device and device != 'auto' else 'cpu'
        ckpt = torch.load(path, map_location=map_loc, weights_only=False)

        input_dim = int(ckpt['scaler_mean'].shape[0])
        inst = cls(
            input_dim=input_dim,
            num_class=int(ckpt['n_classes']),
            step_dim=int(ckpt['step_dim']),
            hidden=int(ckpt['hidden']),
            layers=int(ckpt['layers']),
            dropout=float(ckpt['dropout']),
            bidir=bool(ckpt['bidir']),
            device=device,
        )
        inst.model.load_state_dict(ckpt['state_dict'])
        inst.model.eval()

        inst._scaler = StandardScaler()
        inst._scaler.mean_  = ckpt['scaler_mean']
        inst._scaler.scale_ = ckpt['scaler_scale']
        inst._scaler.var_   = ckpt['scaler_scale'] ** 2
        inst._scaler.n_features_in_ = input_dim

        inst._is_fitted = True
        logger.info(f"[LSTM] Loaded ← {path}")
        return inst
