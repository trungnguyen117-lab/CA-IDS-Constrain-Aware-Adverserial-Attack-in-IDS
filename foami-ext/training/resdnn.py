"""ResDNN training module — architecture mirrors notebook 8, save/load via torch.save.

Checkpoint format is intentionally compatible with ResDNNWrapper.from_checkpoint()
in src/art_classifier/resdnn_classifier.py so the same .pth file works for both
inference and adversarial generation without conversion.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .model import Model

logger = logging.getLogger(__name__)


# ── Architecture (mirrors notebook 8 / resdnn_classifier.py) ─────────────────

class _ResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_hid: int, p: float = 0.25) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hid)
        self.bn1  = nn.BatchNorm1d(d_hid)
        self.lin2 = nn.Linear(d_hid, d_in)
        self.ln2  = nn.LayerNorm(d_in)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch.relu(self.bn1(self.lin1(x))))
        return torch.relu(self.ln2(x + self.lin2(h)))


class _ResDNN(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        W = 512
        self.stem   = nn.Sequential(
            nn.Linear(in_dim, W), nn.BatchNorm1d(W), nn.ReLU(), nn.Dropout(0.30)
        )
        self.block1 = _ResidualBlock(W, W // 2, p=0.30)
        self.block2 = _ResidualBlock(W, W // 2, p=0.25)
        self.block3 = _ResidualBlock(W, W // 2, p=0.20)
        self.head   = nn.Linear(W, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block3(self.block2(self.block1(self.stem(x)))))


# ── Training wrapper ──────────────────────────────────────────────────────────

class ResDNNModel(Model):
    """Residual DNN for tabular classification.

    Preprocessing: StandardScaler fitted during `fit()` and baked into the
    checkpoint so the model is self-contained at inference time.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_class: int,
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
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)

        _dev = device if device and device != 'auto' \
               else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(_dev)
        logger.info(f"[ResDNN] device={self.device}")

        torch.manual_seed(random_state)
        logger.info("[ResDNN] seed set — building model layers ...")

        self.model = _ResDNN(
            in_dim=self.input_dim,
            n_classes=self.num_class,
        )
        logger.info(f"[ResDNN] layers built — moving to {self.device} ...")
        self.model = self.model.to(self.device)
        logger.info("[ResDNN] model ready")

        self._scaler: Optional[StandardScaler] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        logger.info("[ResDNN] fit — scaling data ...")
        self._scaler = StandardScaler()
        X_tr = self._scaler.fit_transform(X_train).astype(np.float32)
        X_v = self._scaler.transform(X_val).astype(np.float32) if X_val is not None else None

        logger.info("[ResDNN] fit — creating DataLoaders ...")
        def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                              pin_memory=self.device.type == 'cuda')

        train_loader = _loader(X_tr, y_train, shuffle=True)
        val_loader = _loader(X_v, y_val, shuffle=False) if X_v is not None and y_val is not None else None

        logger.info("[ResDNN] fit — creating loss + optimizer ...")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_loss = float('inf')
        best_state: Optional[dict] = None
        wait = 0

        logger.info("[ResDNN] fit — entering training loop ...")
        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb in train_loader:
                if epoch == 0:
                    logger.info("[ResDNN] fit — first batch forward pass ...")
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(xb), yb).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                if epoch == 0:
                    logger.info("[ResDNN] fit — first batch done ✓")

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
                    logger.info(f"[ResDNN] Early stop at epoch {epoch + 1}")
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
        """Save checkpoint compatible with ResDNNWrapper.from_checkpoint().

        Keys: state_dict, in_dim, n_classes, scaler_mean, scaler_scale
        """
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError("Model not fitted")
        torch.save({
            'state_dict':   self.model.state_dict(),
            'in_dim':       self.input_dim,
            'n_classes':    self.num_class,
            'scaler_mean':  self._scaler.mean_,
            'scaler_scale': self._scaler.scale_,
        }, path)
        logger.info(f"[ResDNN] Saved → {path}")

    @classmethod
    def load_model(
        cls,
        path: str,
        *,
        device: Optional[str] = None,
        **kwargs,
    ) -> "ResDNNModel":
        """Load checkpoint saved by save_model() or notebook 8."""
        map_loc = device if device and device != 'auto' else 'cpu'
        ckpt = torch.load(path, map_location=map_loc, weights_only=False)

        inst = cls(
            input_dim=int(ckpt['in_dim']),
            num_class=int(ckpt['n_classes']),
            device=device,
        )
        inst.model.load_state_dict(ckpt['state_dict'])
        inst.model.eval()

        inst._scaler = StandardScaler()
        inst._scaler.mean_  = ckpt['scaler_mean']
        inst._scaler.scale_ = ckpt['scaler_scale']
        inst._scaler.var_   = ckpt['scaler_scale'] ** 2
        inst._scaler.n_features_in_ = int(ckpt['in_dim'])

        inst._is_fitted = True
        logger.info(f"[ResDNN] Loaded ← {path}")
        return inst
