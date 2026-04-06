"""Surrogate simple DNN for black-box adversarial generation.

A shallow 3-layer MLP — intentionally different architecture from FOAMI models
(ResDNN uses residual blocks, LSTM uses BiLSTM+Attention). This tests whether
adversarial examples transfer across architecturally dissimilar models.

Architecture: Linear(256) → ReLU → Linear(128) → ReLU → Linear(n_classes)
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


# ── Architecture ─────────────────────────────────────────────────────────────


class SimpleDNN(nn.Module):
    """Shallow 3-layer MLP."""

    def __init__(self, in_dim=66, n_classes=12, hidden1=256, hidden2=128,
                 dropout=0.2):
        super().__init__()
        self.input_norm = InputNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        x = x.float()
        x = self.input_norm(x)
        return self.head(self.net(x))


# ── OOP Wrapper ──────────────────────────────────────────────────────────────


class SurrogateDNNModel(DLModel):
    """Simple DNN surrogate — shallow, fast to train."""

    DEFAULT_CFG = dict(
        hidden1=256, hidden2=128, dropout=0.2,
        lr=1e-3, weight_decay=1e-4,
        batch_size=512, max_epochs=80, patience=15,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None,
              device="cpu"):
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

        model = SimpleDNN(
            in_dim=input_dim, n_classes=num_classes,
            hidden1=c["hidden1"], hidden2=c["hidden2"],
            dropout=c["dropout"],
        ).to(device)

        if input_norm_state is not None:
            model.input_norm.load_state_dict(input_norm_state)
        else:
            model.input_norm.fit(X_train)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                1.0 / (counts / counts.sum()), dtype=torch.float32
            ).to(device),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"],
        )
        scheduler = make_cosine_scheduler(
            optimizer, c["max_epochs"], len(train_dl),
        )

        best_state, best_f1 = train_loop(
            model, train_dl, val_dl, y_va, optimizer, scheduler, criterion,
            c["max_epochs"], c["patience"], device, model_name="SurrogateDNN",
        )

        if best_state:
            ema_on_device = {k: v.to(device) for k, v in best_state.items()}
            _ema_load(model, ema_on_device)
        model.eval()

        logger.info(f"SurrogateDNN trained — best val F1={best_f1:.4f}")
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
            "hidden1": self._cfg["hidden1"],
            "hidden2": self._cfg["hidden2"],
            "dropout": self._cfg["dropout"],
        }, path)
        logger.info(f"Saved SurrogateDNN -> {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        in_dim = int(ckpt.get("in_dim", 66))
        n_classes = int(ckpt.get("n_classes", 12))
        hidden1 = int(ckpt.get("hidden1", 256))
        hidden2 = int(ckpt.get("hidden2", 128))
        dropout = float(ckpt.get("dropout", 0.2))

        model = SimpleDNN(
            in_dim=in_dim, n_classes=n_classes,
            hidden1=hidden1, hidden2=hidden2, dropout=dropout,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)

        inst = cls()
        inst._net = model
        inst._cfg = {
            "in_dim": in_dim, "n_classes": n_classes,
            "hidden1": hidden1, "hidden2": hidden2, "dropout": dropout,
        }
        inst._device = device
        logger.info(f"Loaded SurrogateDNN <- {path}")
        return inst

    def wrap_for_art(self, X_ref, device="cpu", **kwargs):
        num_classes = self._net.head.out_features
        input_dim = X_ref.shape[1]
        clip_values = (0.0, float(X_ref.max()))
        return wrap_dl_for_art(
            self._net, clip_values, num_classes, input_dim,
            device=device,
        )
