"""Surrogate ResDNN: near-identical architecture to ResDNN for adversarial training.

Only difference from the real ResDNN: random weight initialization (different seed).
Everything else matches: width=512, 3 residual blocks, FocalCE loss, class-balanced
sampling, cosine+warmup scheduler, early stopping.

This maximizes gradient landscape similarity -> highest transferability of adversarial
examples for defense via adversarial training.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

from .base import DLModel
from .input_norm import InputNorm
from .dl_utils import (
    prepare_raw_data, create_weighted_loaders,
    make_cosine_scheduler, train_loop, wrap_dl_for_art,
    _ema_load,
)

logger = logging.getLogger(__name__)


# ── Architecture (identical to ResDNN) ───────────────────────────────────────


class _SurrogateResBlock(nn.Module):
    """Same as ResDNN _ResidualBlock."""

    def __init__(self, d_in, d_hid, p=0.25):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hid)
        self.bn1 = nn.BatchNorm1d(d_hid)
        self.lin2 = nn.Linear(d_hid, d_in)
        self.ln2 = nn.LayerNorm(d_in)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.drop(torch.relu(self.bn1(self.lin1(x))))
        h = self.lin2(h)
        return torch.relu(self.ln2(x + h))


class SurrogateResDNN(nn.Module):
    """Same topology as ResDNN: stem(512) -> 3 residual blocks -> head."""

    def __init__(self, in_dim=66, n_classes=12, width=512):
        super().__init__()
        self.input_norm = InputNorm(in_dim)
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width), nn.BatchNorm1d(width),
            nn.ReLU(), nn.Dropout(0.30),
        )
        self.block1 = _SurrogateResBlock(width, width // 2, p=0.30)
        self.block2 = _SurrogateResBlock(width, width // 2, p=0.25)
        self.block3 = _SurrogateResBlock(width, width // 2, p=0.20)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        x = x.float()
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


# ── Focal CE Loss (same as ResDNN) ──────────────────────────────────────────


class _FocalCE(nn.Module):
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


# ── OOP Wrapper ──────────────────────────────────────────────────────────────


class SurrogateResDNNModel(DLModel):
    """Surrogate ResDNN — same training recipe as ResDNN, different random init."""

    DEFAULT_CFG = dict(
        width=512, lr=1e-3, weight_decay=2e-4,
        batch_size=1024, max_epochs=120, patience=20,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        c = {**self.DEFAULT_CFG}
        if cfg:
            c.update(cfg)
        self._device = device

        num_classes = int(len(np.unique(y_train)))
        input_dim = X_train.shape[1]

        input_norm_state = c.pop("input_norm_state", None)

        X_tr, y_tr, X_va, y_va = prepare_raw_data(X_train, y_train, X_val, y_val)
        train_dl, val_dl, counts = create_weighted_loaders(
            X_tr, y_tr, X_va, y_va, c["batch_size"], num_classes,
        )

        model = SurrogateResDNN(
            in_dim=input_dim, n_classes=num_classes, width=c["width"],
        ).to(device)

        if input_norm_state is not None:
            model.input_norm.load_state_dict(input_norm_state)
        else:
            model.input_norm.fit(X_train)

        # Class-balanced focal loss (same as ResDNN)
        beta = 0.999
        eff = (1 - beta) / (1 - beta ** counts)
        eff /= eff.mean()
        criterion = _FocalCE(
            weight=torch.tensor(eff, dtype=torch.float32).to(device),
            gamma=1.7, label_smoothing=0.05,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"],
        )
        scheduler = make_cosine_scheduler(
            optimizer, c["max_epochs"], len(train_dl),
        )

        best_state, best_f1 = train_loop(
            model, train_dl, val_dl, y_va, optimizer, scheduler, criterion,
            c["max_epochs"], c["patience"], device, model_name="SurrogateResDNN",
        )

        if best_state:
            ema_on_device = {k: v.to(device) for k, v in best_state.items()}
            _ema_load(model, ema_on_device)
        model.eval()

        logger.info(f"SurrogateResDNN trained — best val F1={best_f1:.4f}")
        self._net = model
        self._cfg = c
        return self

    def save(self, path):
        num_classes = self._net.head.out_features
        input_dim = self._net.input_norm.num_features
        torch.save({
            "state_dict": self._net.state_dict(),
            "in_dim": input_dim,
            "n_classes": num_classes,
            "width": self._cfg["width"],
        }, path)
        logger.info(f"Saved SurrogateResDNN -> {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        in_dim = int(ckpt.get("in_dim", 66))
        n_classes = int(ckpt.get("n_classes", 12))
        width = int(ckpt.get("width", 512))

        model = SurrogateResDNN(in_dim=in_dim, n_classes=n_classes, width=width)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)

        inst = cls()
        inst._net = model
        inst._cfg = {"in_dim": in_dim, "n_classes": n_classes, "width": width}
        inst._device = device
        logger.info(f"Loaded SurrogateResDNN <- {path}")
        return inst

    def wrap_for_art(self, X_ref, device="cpu", **kwargs):
        num_classes = self._net.head.out_features
        input_dim = X_ref.shape[1]
        clip_values = (0.0, float(X_ref.max()))
        return wrap_dl_for_art(
            self._net, clip_values, num_classes, input_dim,
            device=device,
        )
