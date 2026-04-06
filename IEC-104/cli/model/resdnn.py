"""ResDNN model: architecture, train, save, load, ART wrappers."""

import logging

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


class _ResidualBlock(nn.Module):
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


class ResDNN(nn.Module):
    def __init__(self, in_dim=66, n_classes=12, width=512):
        super().__init__()
        self.input_norm = InputNorm(in_dim)
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width), nn.BatchNorm1d(width),
            nn.ReLU(), nn.Dropout(0.30),
        )
        self.block1 = _ResidualBlock(width, width // 2, p=0.30)
        self.block2 = _ResidualBlock(width, width // 2, p=0.25)
        self.block3 = _ResidualBlock(width, width // 2, p=0.20)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        x = x.float()
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


# ── OOP Wrapper ──────────────────────────────────────────────────────────────


class ResDNNModel(DLModel):
    """ResDNN classifier for FOAMI+ pipeline."""

    DEFAULT_CFG = dict(
        lr=1e-3, weight_decay=2e-4, batch_size=1024, max_epochs=120, patience=20,
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

        model = ResDNN(in_dim=input_dim, n_classes=num_classes).to(device)

        if input_norm_state is not None:
            model.input_norm.load_state_dict(input_norm_state)
            logger.info("Reusing baseline InputNorm for AT")
        else:
            model.input_norm.fit(X_train)

        # Class-balanced focal loss
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
            c["max_epochs"], c["patience"], device, model_name="ResDNN",
        )


        if best_state:
            ema_on_device = {k: v.to(device) for k, v in best_state.items()}
            _ema_load(model, ema_on_device)
        model.eval()


        logger.info(f"ResDNN trained — best val F1={best_f1:.4f}")
        self._net = model
        self._cfg = c
        return self

    def save(self, path):
        num_classes = self._net.head.out_features
        input_dim = self._net.stem[0].in_features
        torch.save({
            "state_dict": self._net.state_dict(),
            
            "in_dim": input_dim,
            "n_classes": num_classes,
        }, path)
        logger.info(f"Saved ResDNN -> {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        in_dim = int(ckpt.get("in_dim", 66))
        n_classes = int(ckpt.get("n_classes", 12))

        model = ResDNN(in_dim=in_dim, n_classes=n_classes)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)

        inst = cls()
        inst._net = model
        inst._cfg = {"in_dim": in_dim, "n_classes": n_classes}
        inst._device = device
        logger.info(f"Loaded ResDNN <- {path}")
        return inst

    def wrap_for_art(self, X_ref, device="cpu", preprocessing_defences=None, **kwargs):
        num_classes = self._net.head.out_features
        input_dim = X_ref.shape[1]
        clip_values = (0.0, float(X_ref.max()))
        return wrap_dl_for_art(
            self._net, clip_values, num_classes, input_dim,
            device=device, preprocessing_defences=preprocessing_defences,
        )
