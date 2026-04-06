"""FT-Transformer model wrapper (from MODBUS notebook 6_MI)."""

import logging
import math
import pickle

import numpy as np
import torch
import torch.nn as nn

from .base import DLModel
from .dl_utils import (
    prepare_qt_data, create_weighted_loaders,
    train_loop, restore_qt, wrap_dl_for_art,
)

logger = logging.getLogger(__name__)


# ── Architecture ──────────────────────────────────────────────────────────────


class NumericalTokenizer(nn.Module):
    """Per-feature linear tokenizer: x_i -> x_i * w_i + b_i."""

    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(d_token)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (B, F) -> (B, F, d_token)
        return x.unsqueeze(-1) * self.weight[None] + self.bias[None]


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer Encoder."""

    def __init__(self, n_features, num_classes, d_token=192, n_heads=8,
                 n_layers=3, ffn_factor=4 / 3, dropout=0.1):
        super().__init__()
        self.tokenizer = NumericalTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        d_ffn = int(d_token * ffn_factor)
        d_ffn = max(d_ffn, d_token)
        d_ffn = (d_ffn + 7) // 8 * 8  # align to 8

        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2), nn.ReLU(),
            nn.Linear(d_token // 2, num_classes),
        )

    def forward(self, x):
        x = x.float()
        tokens = self.tokenizer(x)                        # (B, F, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)    # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, F+1, d)
        out = self.encoder(tokens)
        return self.head(self.norm(out[:, 0]))              # CLS -> logits


class FocalLoss(nn.Module):
    """Focal loss with label smoothing for class-imbalanced training."""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=weight, reduction='none', label_smoothing=label_smoothing,
        )

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.softmax(logits, 1).gather(1, targets.unsqueeze(1)).squeeze(1)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ── Model Wrapper ─────────────────────────────────────────────────────────────


class FTTransformerModel(DLModel):
    """Multi-seed FT-Transformer model for MODBUS-2023."""

    _nets = None  # list of FTTransformer nets (one per seed)
    _device = "cpu"

    DEFAULT_CFG = dict(
        d_token=192, n_heads=8, n_layers=3, ffn_factor=4 / 3, dropout=0.1,
        lr=1e-4, weight_decay=1e-5, batch_size=256,
        max_epochs=150, patience=30, n_seeds=3,
    )

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        self._device = device
        c = {**self.DEFAULT_CFG}
        if cfg:
            c.update(cfg)
        self._cfg = c

        n_features = X_train.shape[1]
        num_classes = int(len(np.unique(y_train)))
        n_seeds = c.get("n_seeds", 3)

        # Prepare data with QuantileTransformer
        qt, X_tr, y_tr, X_va, y_va = prepare_qt_data(X_train, y_train, X_val, y_val)
        self._scaler = qt

        train_dl, val_dl, counts = create_weighted_loaders(
            X_tr, y_tr, X_va, y_va, c["batch_size"], num_classes,
        )

        # Class weights for FocalLoss
        inv = counts.sum() / (counts + 1e-9)
        inv /= inv.mean()
        class_w = torch.tensor(inv, dtype=torch.float32).to(device)

        self._nets = []
        for s in range(n_seeds):
            seed = s * 137 + 7  # seeds: [7, 144, 281] matching notebook
            torch.manual_seed(seed)
            np.random.seed(seed)

            net = FTTransformer(
                n_features, num_classes,
                d_token=c["d_token"], n_heads=c["n_heads"],
                n_layers=c["n_layers"], ffn_factor=c.get("ffn_factor", 4 / 3),
                dropout=c["dropout"],
            ).to(device)

            optimizer = torch.optim.AdamW(
                net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=120, eta_min=1e-6,
            )
            criterion = FocalLoss(weight=class_w)

            best_state, best_f1 = train_loop(
                net, train_dl, val_dl, y_va, optimizer, scheduler, criterion,
                c["max_epochs"], c["patience"], device, model_name=f"FTT-seed{s}",
            )
            net.load_state_dict(best_state)
            net.eval()
            self._nets.append(net)
            logger.info(f"FTT seed {seed}: val F1={best_f1:.4f}")

        # Use first net as primary _net for base class predict methods
        self._net = self._nets[0]
        return self

    def predict(self, X):
        """Average logits across seeds, then argmax."""
        X_sc = self._scaler.transform(X).astype(np.float32)
        xt = torch.from_numpy(X_sc).to(self._device)
        all_logits = []
        with torch.no_grad():
            for net in self._nets:
                all_logits.append(net(xt))
        avg = torch.stack(all_logits).mean(dim=0)
        return torch.argmax(avg, dim=1).cpu().numpy()

    def predict_proba(self, X):
        """Average softmax across seeds."""
        X_sc = self._scaler.transform(X).astype(np.float32)
        xt = torch.from_numpy(X_sc).to(self._device)
        all_proba = []
        with torch.no_grad():
            for net in self._nets:
                all_proba.append(torch.softmax(net(xt), dim=1))
        avg = torch.stack(all_proba).mean(dim=0)
        return avg.cpu().numpy().astype(np.float64)

    def save(self, path):
        ckpt = {
            "state_dicts": [n.state_dict() for n in self._nets],
            "qt_bytes": pickle.dumps(self._scaler),
            "n_features": self._nets[0].tokenizer.weight.shape[0],
            "num_classes": self._nets[0].head[-1].out_features,
            "config": self._cfg,
        }
        torch.save(ckpt, path)
        logger.info(f"Saved FTT ({len(self._nets)} seeds) → {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        inst = cls()
        inst._device = device
        ckpt = torch.load(path, map_location=device, weights_only=False)

        inst._scaler = restore_qt(ckpt)
        inst._cfg = ckpt.get("config", {})
        n_features = ckpt["n_features"]
        num_classes = ckpt["num_classes"]
        c = inst._cfg or cls.DEFAULT_CFG

        inst._nets = []
        for sd in ckpt["state_dicts"]:
            net = FTTransformer(
                n_features, num_classes,
                d_token=c.get("d_token", 192), n_heads=c.get("n_heads", 8),
                n_layers=c.get("n_layers", 3),
                ffn_factor=c.get("ffn_factor", 4 / 3),
                dropout=c.get("dropout", 0.1),
            ).to(device)
            net.load_state_dict(sd)
            net.eval()
            inst._nets.append(net)

        inst._net = inst._nets[0]
        logger.info(f"Loaded FTT ({len(inst._nets)} seeds) ← {path}")
        return inst

    def wrap_for_art(self, X_ref, clip_values=None, raw=True, device="cpu",
                     preprocessing_defences=None, **kwargs):
        """Wrap first seed for ART.

        raw=True: for evaluation on raw data (BB attacks).
        raw=False: for WB attack generation in transformed space.
        clip_values: per-feature (min, max) from training data. If None, computed from X_ref.
        """
        net = self._nets[0] if self._nets else self._net
        n_features = net.tokenizer.weight.shape[0]
        num_classes = net.head[-1].out_features

        if clip_values is None:
            if raw:
                clip_values = (X_ref.min(axis=0).astype(np.float32), X_ref.max(axis=0).astype(np.float32))
            else:
                X_sc = self._scaler.transform(X_ref).astype(np.float32)
                clip_values = (float(X_sc.min()), float(X_sc.max()))

        return wrap_dl_for_art(
            net, self._scaler, clip_values, num_classes, n_features,
            device=device, raw=raw,
            preprocessing_defences=preprocessing_defences,
        )
