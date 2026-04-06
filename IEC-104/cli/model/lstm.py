"""LSTM model: architecture, train, save, load, ART wrappers."""

import logging
import math

import numpy as np
import torch
import torch.nn as nn

from .base import DLModel
from .input_norm import InputNorm
from .dl_utils import (
    prepare_raw_data, create_weighted_loaders,
    make_cosine_scheduler, train_loop, wrap_dl_for_art,
    _ema_load, FocalCE,
)

logger = logging.getLogger(__name__)


# ── Architecture ─────────────────────────────────────────────────────────────


class _AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, 1)

    def forward(self, H):
        w = torch.softmax(self.proj(H).squeeze(-1), dim=1)
        return (H * w.unsqueeze(-1)).sum(1)


class LSTMTabular(nn.Module):
    def __init__(self, in_dim=66, step_dim=16, hidden=256, layers=2, n_classes=12,
                 dropout=0.15, bidir=True):
        super().__init__()
        self.step_dim = step_dim
        self.input_norm = InputNorm(in_dim)
        self.in_norm = nn.LayerNorm(step_dim)
        self.lstm = nn.LSTM(
            input_size=step_dim, hidden_size=hidden, num_layers=layers,
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

    def forward(self, x):
        x = x.float()
        x = self.input_norm(x)
        B, F = x.shape
        S = math.ceil(F / self.step_dim)
        pad = S * self.step_dim - F
        if pad > 0:
            x = nn.functional.pad(x, (0, pad), value=0.0)
        x = x.view(B, S, self.step_dim)
        x = self.in_norm(x)
        H, _ = self.lstm(x)
        return self.head(self.pool(H))


# ── OOP Wrapper ──────────────────────────────────────────────────────────────


class LSTMModel(DLModel):
    """LSTM classifier for FOAMI+ pipeline."""

    DEFAULT_CFG = dict(
        step_dim=16, hidden=256, layers=2, dropout=0.15, bidir=True,
        lr=1e-3, weight_decay=2e-4, batch_size=2048, max_epochs=120, patience=20,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        c = {**self.DEFAULT_CFG}
        if cfg:
            c.update(cfg)
        self._device = device

        num_classes = int(len(np.unique(y_train)))
        input_dim = X_train.shape[1]

        # InputNorm: use provided state or fit from training data
        input_norm_state = c.pop("input_norm_state", None)

        X_tr, y_tr, X_va, y_va = prepare_raw_data(X_train, y_train, X_val, y_val)
        train_dl, val_dl, counts = create_weighted_loaders(
            X_tr, y_tr, X_va, y_va, c["batch_size"], num_classes,
        )

        model = LSTMTabular(
            in_dim=input_dim,
            step_dim=c["step_dim"], hidden=c["hidden"], layers=c["layers"],
            n_classes=num_classes, dropout=c["dropout"], bidir=c["bidir"],
        ).to(device)

        if input_norm_state is not None:
            model.input_norm.load_state_dict(input_norm_state)
            logger.info("Reusing baseline InputNorm for AT")
        else:
            model.input_norm.fit(X_train)

        beta = 0.999
        eff = (1 - beta) / (1 - beta ** counts)
        eff /= eff.mean()
        criterion = FocalCE(
            weight=torch.tensor(eff, dtype=torch.float32).to(device),
            gamma=1.7, label_smoothing=0.05,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"],
        )
        warmup_ep = c.get("warmup_epochs", 19)
        scheduler = make_cosine_scheduler(optimizer, c["max_epochs"], len(train_dl),
                                          warmup_epochs=warmup_ep)

        best_state, best_f1 = train_loop(
            model, train_dl, val_dl, y_va, optimizer, scheduler, criterion,
            c["max_epochs"], c["patience"], device, model_name="LSTM",
        )

        if best_state:
            ema_on_device = {k: v.to(device) for k, v in best_state.items()}
            _ema_load(model, ema_on_device)
        model.eval()

        logger.info(f"LSTM trained — best val F1={best_f1:.4f}")
        self._net = model
        self._cfg = c
        return self

    def save(self, path):
        cfg = self._cfg
        torch.save({
            "state_dict": self._net.state_dict(),
            "step_dim": cfg["step_dim"],
            "hidden": cfg["hidden"],
            "layers": cfg["layers"],
            "dropout": cfg["dropout"],
            "bidir": cfg["bidir"],
            "num_classes": self._net.head[-1].out_features,
            "in_dim": self._net.input_norm.num_features,
        }, path)
        logger.info(f"Saved LSTM -> {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = {
            "step_dim": int(ckpt.get("step_dim", 16)),
            "hidden": int(ckpt.get("hidden", 256)),
            "layers": int(ckpt.get("layers", 2)),
            "dropout": float(ckpt.get("dropout", 0.15)),
            "bidir": bool(ckpt.get("bidir", True)),
        }
        num_classes = int(ckpt.get("num_classes", ckpt.get("n_classes", 12)))
        in_dim = int(ckpt.get("in_dim", 66))

        model = LSTMTabular(
            in_dim=in_dim,
            step_dim=cfg["step_dim"], hidden=cfg["hidden"], layers=cfg["layers"],
            n_classes=num_classes, dropout=cfg["dropout"], bidir=cfg["bidir"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)

        inst = cls()
        inst._net = model
        inst._cfg = cfg
        inst._device = device
        logger.info(f"Loaded LSTM <- {path}")
        return inst

    def wrap_for_art(self, X_ref, device="cpu", preprocessing_defences=None, **kwargs):
        num_classes = self._net.head[-1].out_features
        input_dim = X_ref.shape[1]
        clip_values = (0.0, float(X_ref.max()))
        return wrap_dl_for_art(
            self._net, clip_values, num_classes, input_dim,
            device=device, preprocessing_defences=preprocessing_defences,
        )
