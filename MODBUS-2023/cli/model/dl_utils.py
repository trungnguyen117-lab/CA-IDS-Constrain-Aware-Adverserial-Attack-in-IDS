"""Shared utilities for DL model training (FT-Transformer)."""

import logging
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ── ScaledModel (linear, for StandardScaler compatibility) ────────────────────


class ScaledModel(nn.Module):
    """Embeds a linear scaler (mean/scale) into any model for ART evaluation."""

    def __init__(self, model, scaler_mean, scaler_scale):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scaler_scale, dtype=torch.float32))

    def forward(self, x):
        x = x.float()
        x = (x - self.mean) / self.scale
        return self.model(x)


class QTScaledModel(nn.Module):
    """Embeds QuantileTransformer into forward pass for ART BB evaluation.

    QT is non-linear (rank-based), so we run it in numpy then convert back.
    This breaks autograd — use ONLY for black-box attacks / evaluation.
    """

    def __init__(self, model, qt_scaler, device="cpu"):
        super().__init__()
        self.model = model
        self._qt = qt_scaler
        self._device = device

    def forward(self, x):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        x_scaled = self._qt.transform(x_np).astype(np.float32)
        x_t = torch.from_numpy(x_scaled).to(self._device)
        return self.model(x_t)


# ── Data preparation ─────────────────────────────────────────────────────────


def prepare_qt_data(X_train, y_train, X_val=None, y_val=None):
    """Fit QuantileTransformer on train, split val if not provided.

    Returns (qt, X_tr, y_tr, X_va, y_va) — all transformed float32.
    """
    qt = QuantileTransformer(
        output_distribution='normal', n_quantiles=1000, random_state=42,
    )
    X_tr_sc = qt.fit_transform(X_train).astype(np.float32)

    if X_val is None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tr_idx, va_idx = next(sss.split(X_tr_sc, y_train))
        X_tr, y_tr = X_tr_sc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_tr_sc[va_idx], y_train[va_idx]
    else:
        X_tr, y_tr = X_tr_sc, y_train
        X_va = qt.transform(X_val).astype(np.float32)
        y_va = y_val

    return qt, X_tr, y_tr, X_va, y_va


def create_weighted_loaders(X_tr, y_tr, X_va, y_va, batch_size, num_classes):
    """Create balanced-sampler DataLoaders for train and val.

    Returns (train_dl, val_dl, counts).
    """
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    inv = counts.sum() / (counts + 1e-9)
    inv /= inv.mean()
    sampler = WeightedRandomSampler(inv[y_tr], num_samples=len(y_tr), replacement=True)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
        batch_size=batch_size, sampler=sampler,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long()),
        batch_size=4096, shuffle=False,
    )
    return train_dl, val_dl, counts


# ── Training loop ────────────────────────────────────────────────────────────


def make_cosine_scheduler(optimizer, max_epochs, steps_per_epoch, warmup_epochs=8):
    """Warmup + cosine annealing LR scheduler (per-step)."""
    total_steps = max_epochs * max(1, steps_per_epoch)
    warmup_steps = warmup_epochs * max(1, steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_loop(model, train_dl, val_dl, y_va, optimizer, scheduler, criterion,
               max_epochs, patience, device, model_name="DL"):
    """Generic early-stop training loop. Returns (best_state_dict, best_f1)."""
    best_f1, best_state, wait = -1.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val_dl:
                preds.extend(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
        vf1 = f1_score(y_va, preds, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if wait >= patience:
            logger.info(f"{model_name} early stop epoch {epoch+1}, val F1={best_f1:.4f}")
            break
        if (epoch + 1) % 20 == 0:
            logger.debug(f"{model_name} epoch {epoch+1}, val F1={vf1:.4f}")

    return best_state, best_f1


# ── Save / Load helpers ──────────────────────────────────────────────────────


def restore_qt(ckpt):
    """Restore QuantileTransformer from checkpoint dict."""
    qt = ckpt.get("qt_scaler")
    if qt is None and "qt_bytes" in ckpt:
        qt = pickle.loads(ckpt["qt_bytes"])
    return qt


def wrap_dl_for_art(model, scaler, clip_values, num_classes, input_dim, device="cpu",
                    raw=True, preprocessing_defences=None):
    """Wrap DL model as ART PyTorchClassifier.

    raw=True: for BB attacks / evaluation on raw data. Embeds QT into model
              (breaks autograd — BB only).
    raw=False: direct model for WB attack generation in QT-transformed space
              (autograd intact).
    """
    from art.estimators.classification import PyTorchClassifier

    if raw:
        wrapped = QTScaledModel(model, scaler, device=device)
    else:
        wrapped = model

    return PyTorchClassifier(
        model=wrapped,
        loss=nn.CrossEntropyLoss(),
        input_shape=(input_dim,),
        nb_classes=num_classes,
        clip_values=clip_values,
        device_type=device,
        preprocessing_defences=preprocessing_defences,
    )
