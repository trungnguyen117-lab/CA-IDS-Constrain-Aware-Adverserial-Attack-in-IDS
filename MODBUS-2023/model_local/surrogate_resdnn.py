"""SurrogateResDNN — InputNorm + 2 residual blocks (transferability eval)."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.core.models import DLModel
from src.core.paths import InputNorm

logger = logging.getLogger(__name__)


class _ResidualBlock(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.2):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hid)
        self.bn1 = nn.BatchNorm1d(d_hid)
        self.lin2 = nn.Linear(d_hid, d_in)
        self.ln2 = nn.LayerNorm(d_in)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(torch.relu(self.bn1(self.lin1(x))))
        h = self.lin2(h)
        return torch.relu(self.ln2(x + h))


class SimpleResDNN(nn.Module):
    def __init__(self, in_dim, n_classes, width=256, dropout=0.2):
        super().__init__()
        self.input_norm = InputNorm(in_dim)
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width), nn.BatchNorm1d(width),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.block1 = _ResidualBlock(width, width // 2, dropout)
        self.block2 = _ResidualBlock(width, width // 2, dropout)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        x = self.input_norm(x.float())
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


class SurrogateResDNN(DLModel):
    DEFAULT_CFG = dict(
        width=256, dropout=0.2,
        lr=1e-3, weight_decay=1e-4,
        batch_size=512, max_epochs=80, patience=15, val_size=0.15,
    )

    @classmethod
    def build_net(cls, in_dim, n_classes, cfg, X_train_raw=None) -> nn.Module:
        c = {**cls.DEFAULT_CFG, **(cfg or {})}
        net = SimpleResDNN(in_dim, n_classes, c["width"], c["dropout"])
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
            X_tr, y_tr, X_va, y_va = X_train, y_train, X_val, y_val
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=c["val_size"], random_state=42)
            tr, va = next(sss.split(X_train, y_train))
            X_tr, y_tr = X_train[tr], y_train[tr]
            X_va, y_va = X_train[va], y_train[va]

        counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
        inv = counts.sum() / (counts + 1e-9); inv /= inv.mean()
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
                    break

        if best_state:
            net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        net.eval()
        self._net = net; self._cfg = c
        logger.info("SurrogateResDNN trained — best val_loss=%.4f", best_loss)
        return self

    def save(self, path):
        torch.save({
            "state_dict": self._net.state_dict(),
            "cfg": self._cfg,
            "in_dim": self._net.stem[0].in_features,
            "n_classes": self._net.head.out_features,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        c = ckpt["cfg"]
        net = SimpleResDNN(ckpt["in_dim"], ckpt["n_classes"],
                           c["width"], c["dropout"])
        net.load_state_dict(ckpt["state_dict"], strict=False)
        net.eval().to(device)
        inst = cls()
        inst._net = net; inst._cfg = c; inst._device = device
        return inst
