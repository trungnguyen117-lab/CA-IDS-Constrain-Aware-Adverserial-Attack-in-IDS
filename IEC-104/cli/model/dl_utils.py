"""Shared utilities for DL model training (LSTM, ResDNN)."""

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ── Focal CE Loss ────────────────────────────────────────────────────────────


class FocalCE(nn.Module):
    def __init__(self, weight=None, gamma=1.7, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Data preparation ─────────────────────────────────────────────────────────


def prepare_raw_data(X_train, y_train, X_val=None, y_val=None):
    """Split val if not provided. No scaling — InputNorm handles it.

    Returns (X_tr, y_tr, X_va, y_va) — all float32.
    """
    X_tr = X_train.astype(np.float32)

    if X_val is None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tr_idx, va_idx = next(sss.split(X_tr, y_train))
        X_va = X_tr[va_idx]
        y_va = y_train[va_idx]
        X_tr = X_tr[tr_idx]
        y_tr = y_train[tr_idx]
    else:
        y_tr = y_train
        X_va = X_val.astype(np.float32)
        y_va = y_val

    return X_tr, y_tr, X_va, y_va


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


def make_cosine_scheduler(optimizer, max_epochs, steps_per_epoch, warmup_epochs=19,
                          min_lr_ratio=0.0):
    """Warmup + cosine annealing LR scheduler (per-step)."""
    total_steps = max_epochs * max(1, steps_per_epoch)
    warmup_steps = warmup_epochs * max(1, steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Mixup ───────────────────────────────────────────────────────────────────


def _mixup_pair(x, y, alpha=0.10):
    """Mixup augmentation. Returns (x_mixed, y_a, lam, y_b)."""
    if alpha <= 0:
        return x, y, 1.0, None
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1 - lam) * x[perm]
    return x_m, y, lam, y[perm]


# ── EMA ─────────────────────────────────────────────────────────────────────


def _ema_init(model, device):
    """Initialize EMA state from model (float params only)."""
    ema = {}
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            ema[k] = v.detach().clone().to(device=v.device, dtype=v.dtype)
    return ema


def _ema_update(model, ema, decay=0.995):
    """Update EMA state in-place."""
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if not v.is_floating_point():
                continue
            if ema[k].device != v.device or ema[k].dtype != v.dtype:
                ema[k] = ema[k].to(device=v.device, dtype=v.dtype)
            ema[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)


def _ema_load(model, ema):
    """Load EMA state into model."""
    current = model.state_dict()
    merged = {}
    for k, v in current.items():
        merged[k] = ema.get(k, v).to(device=v.device, dtype=v.dtype)
    model.load_state_dict(merged, strict=True)


# ── Training loop ───────────────────────────────────────────────────────────


def train_loop(model, train_dl, val_dl, y_va, optimizer, scheduler, criterion,
               max_epochs, patience, device, model_name="DL",
               mixup_alpha=0.10, ema_decay=0.995):
    """Training loop with EMA + Mixup + early-stop. Returns (best_state_dict, best_f1)."""
    best_f1, best_ema_snapshot, wait = -1.0, None, 0
    ema = _ema_init(model, device)

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            # Mixup
            xb_m, ya, lam, yb_ = _mixup_pair(xb, yb, alpha=mixup_alpha)
            logits = model(xb_m)
            if yb_ is not None:
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb_)
            else:
                loss = criterion(logits, ya)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            scheduler.step()
            _ema_update(model, ema, ema_decay)

        # Validate with EMA weights
        saved_live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _ema_load(model, ema)
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val_dl:
                preds.extend(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
        vf1 = f1_score(y_va, preds, average="macro", zero_division=0)

        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            best_ema_snapshot = {k: v.detach().cpu().clone() for k, v in ema.items()}
        else:
            wait += 1

        # Restore live weights for next training epoch
        model.load_state_dict(saved_live, strict=True)

        if wait >= patience:
            logger.info(f"{model_name} early stop epoch {epoch+1}, val F1={best_f1:.4f}")
            break
        if (epoch + 1) % 20 == 0:
            logger.debug(f"{model_name} epoch {epoch+1}, val F1={vf1:.4f}")

    # Return best EMA snapshot as the final state
    return best_ema_snapshot, best_f1


# ── ART wrapper ──────────────────────────────────────────────────────────────


def wrap_dl_for_art(model, clip_values, num_classes, input_dim, device="cpu",
                    preprocessing_defences=None):
    """Wrap DL model as ART PyTorchClassifier.

    InputNorm is embedded in the model — no external ScaledModel needed.
    Model accepts raw input directly.
    """
    from art.estimators.classification import PyTorchClassifier

    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(input_dim,),
        nb_classes=num_classes,
        clip_values=clip_values,
        device_type=device,
        preprocessing_defences=preprocessing_defences,
    )
