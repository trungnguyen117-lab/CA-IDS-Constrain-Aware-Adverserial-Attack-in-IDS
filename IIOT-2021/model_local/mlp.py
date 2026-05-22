"""Plain MLP — InputNorm + Linear stack (raw input, internal normalization)."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.core.models import DLModel
from src.core.paths import InputNorm

logger = logging.getLogger(__name__)


class PlainMLP(nn.Module):
    def __init__(self, in_dim, n_classes, h1=256, h2=128, h3=64, dropout=0.1):
        super().__init__()
        self.input_norm = InputNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h3, n_classes),
        )

    def forward(self, x):
        return self.net(self.input_norm(x.float()))


class MLPModel(DLModel):
    DEFAULT_CFG = dict(
        h1=256, h2=128, h3=64, dropout=0.1,
        lr=1e-3, weight_decay=1e-4,
        batch_size=512, max_epochs=60, patience=10, val_size=0.1,
    )

    @classmethod
    def build_net(cls, in_dim, n_classes, cfg, X_train_raw=None) -> nn.Module:
        c = {**cls.DEFAULT_CFG, **(cfg or {})}
        net = PlainMLP(in_dim, n_classes, c["h1"], c["h2"], c["h3"], c["dropout"])
        if X_train_raw is not None:
            net.input_norm.fit(X_train_raw.astype(np.float32))
        return net

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        c = {**self.DEFAULT_CFG, **(cfg or {})}
        self._device = device
        in_dim = X_train.shape[1]
        n_classes = int(len(np.unique(y_train)))

        net = self.build_net(in_dim, n_classes, c, X_train_raw=X_train).to(device)

        X_train = X_train.astype(np.float32)
        if X_val is not None:
            X_val = X_val.astype(np.float32)

        if X_val is not None and y_val is not None:
            X_tr, y_tr = X_train, y_train
            X_va, y_va = X_val, y_val
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=c["val_size"], random_state=42)
            tr_idx, va_idx = next(sss.split(X_train, y_train))
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

        counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
        inv = counts.sum() / (counts + 1e-9)
        inv /= inv.mean()
        sampler = WeightedRandomSampler(inv[y_tr], num_samples=len(y_tr), replacement=True)
        train_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
            batch_size=c["batch_size"], sampler=sampler,
        )
        val_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).long()),
            batch_size=4096, shuffle=False,
        )
        class_w = torch.tensor(1.0 / (counts / counts.sum()), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        optimizer = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=c["max_epochs"] * len(train_dl),
        )

        best_state, best_loss, patience = None, float("inf"), 0
        for epoch in range(c["max_epochs"]):
            net.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(net(xb), yb).backward()
                optimizer.step()
                scheduler.step()
            net.eval()
            tot_loss, tot_n = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    bs = yb.size(0)
                    tot_loss += float(criterion(net(xb), yb).item()) * bs
                    tot_n += bs
            val_loss = tot_loss / max(1, tot_n)
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= c["patience"]:
                    logger.info("Early stop epoch %d, best val_loss=%.4f", epoch + 1, best_loss)
                    break

        if best_state:
            net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        net.eval()
        self._net = net; self._cfg = c
        logger.info("MLP trained — best val_loss=%.4f", best_loss)
        return self

    def save(self, path):
        torch.save({
            "state_dict": self._net.state_dict(),
            "cfg": self._cfg,
            "in_dim": self._net.net[0].in_features,
            "n_classes": self._net.net[-1].out_features,
        }, path)
        logger.info("Saved MLP → %s", path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        c = ckpt["cfg"]
        net = PlainMLP(ckpt["in_dim"], ckpt["n_classes"],
                       c["h1"], c["h2"], c["h3"], c["dropout"])
    
        net.load_state_dict(ckpt["state_dict"], strict=False)
        net.eval().to(device)
        inst = cls()
        inst._net = net; inst._cfg = c; inst._device = device
        return inst
