"""Plain DNN — InputNorm + LN/GELU/Linear/Dropout."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.core.models import DLModel
from src.core.paths import InputNorm

logger = logging.getLogger(__name__)


def mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class PlainDNN(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dims=(384, 256, 128, 64),
                 dropout=0.15):
        super().__init__()
        self.input_norm = InputNorm(in_dim)
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers.append(mlp_block(prev, h, dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.input_norm(x.float()))


class DNNModel(DLModel):
    DEFAULT_CFG = dict(
        hidden_dims=(384, 256, 128, 64), dropout=0.15,
        max_lr=5e-4, weight_decay=5e-4, label_smoothing=0.05,
        batch_size=512, max_epochs=250, patience=35, val_size=0.1,
        warmup_pct=0.05, seed=42,
    )

    @classmethod
    def build_net(cls, in_dim, n_classes, cfg, X_train_raw=None) -> nn.Module:
        c = {**cls.DEFAULT_CFG, **(cfg or {})}
        net = PlainDNN(in_dim, n_classes,
                       hidden_dims=tuple(c["hidden_dims"]), dropout=c["dropout"])
        if X_train_raw is not None:
            net.input_norm.fit(X_train_raw.astype(np.float32))
        return net

    def train(self, X_train, y_train, X_val=None, y_val=None, cfg=None, device="cpu"):
        c = {**self.DEFAULT_CFG, **(cfg or {})}
        torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
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
        criterion = nn.CrossEntropyLoss(weight=class_w,
                                        label_smoothing=c["label_smoothing"])
        optimizer = torch.optim.AdamW(net.parameters(), lr=c["max_lr"],
                                      weight_decay=c["weight_decay"])
        total_steps = c["max_epochs"] * max(1, len(train_dl))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=c["max_lr"], total_steps=total_steps,
            pct_start=c["warmup_pct"], anneal_strategy="cos",
            div_factor=25.0, final_div_factor=1e4,
        )

        best_state, best_f1, patience = None, -1.0, 0
        for epoch in range(c["max_epochs"]):
            net.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(net(xb), yb).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            net.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in val_dl:
                    preds.append(net(xb.to(device)).argmax(dim=1).cpu().numpy())
            val_f1 = f1_score(y_va, np.concatenate(preds), average="macro", zero_division=0)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= c["patience"]:
                    logger.info("Early stop epoch %d, best F1=%.4f", epoch + 1, best_f1)
                    break

        if best_state:
            net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        net.eval()
        self._net = net; self._cfg = c
        logger.info("DNN trained — best val F1=%.4f", best_f1)
        return self

    def save(self, path):
        torch.save({
            "state_dict": self._net.state_dict(),
            "cfg": self._cfg,
            "in_dim": self._net.net[0][0].in_features,
            "n_classes": self._net.net[-1].out_features,
        }, path)
        logger.info("Saved DNN → %s", path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        c = ckpt["cfg"]
        net = PlainDNN(
            ckpt["in_dim"], ckpt["n_classes"],
            hidden_dims=tuple(c["hidden_dims"]), dropout=c["dropout"],
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval().to(device)
        inst = cls()
        inst._net = net; inst._cfg = c; inst._device = device
        return inst
