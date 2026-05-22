"""Defenses: MagNet AE, Mahalanobis OOD, distillation, ART preprocessing wrapper."""

from __future__ import annotations

import pickle
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import os
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F


# ── dae_model ──

class VanillaDAE(nn.Module):
    def __init__(self, data_dim, hidden_dim=256, bottleneck_dim=64, dropout=0.1, **_):
        super().__init__()
        mid = hidden_dim // 2
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid), nn.LayerNorm(mid), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(mid, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, mid), nn.LayerNorm(mid), nn.SiLU(),
            nn.Linear(mid, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DAEResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(self.norm(x))


class ResidualDAE(nn.Module):
    def __init__(self, data_dim, hidden_dim=256, bottleneck_dim=64,
                 n_res_blocks=2, dropout=0.1, **_):
        super().__init__()
        h1, h2 = hidden_dim, hidden_dim // 2
        self.enc1 = nn.Sequential(nn.Linear(data_dim, h1), nn.LayerNorm(h1), nn.SiLU(), nn.Dropout(dropout))
        self.enc1_res = nn.ModuleList([DAEResBlock(h1) for _ in range(n_res_blocks)])
        self.enc2 = nn.Sequential(nn.Linear(h1, h2), nn.LayerNorm(h2), nn.SiLU(), nn.Dropout(dropout))
        self.enc2_res = nn.ModuleList([DAEResBlock(h2) for _ in range(n_res_blocks)])
        self.enc_bottleneck = nn.Sequential(nn.Linear(h2, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.SiLU())
        self.dec_bottleneck = nn.Sequential(nn.Linear(bottleneck_dim, h2), nn.LayerNorm(h2), nn.SiLU())
        self.dec2 = nn.Sequential(nn.Linear(h2 * 2, h1), nn.LayerNorm(h1), nn.SiLU())
        self.dec2_res = nn.ModuleList([DAEResBlock(h1) for _ in range(n_res_blocks)])
        self.dec1 = nn.Sequential(nn.Linear(h1 * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.dec1_res = nn.ModuleList([DAEResBlock(hidden_dim) for _ in range(n_res_blocks)])
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, data_dim), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        for blk in self.enc1_res:
            e1 = blk(e1)
        e2 = self.enc2(e1)
        for blk in self.enc2_res:
            e2 = blk(e2)
        z = self.enc_bottleneck(e2)
        d2 = self.dec_bottleneck(z)
        d2 = self.dec2(torch.cat([d2, e2], dim=-1))
        for blk in self.dec2_res:
            d2 = blk(d2)
        d1 = self.dec1(torch.cat([d2, e1], dim=-1))
        for blk in self.dec1_res:
            d1 = blk(d1)
        return self.output_proj(d1)


DAE_ARCH_REGISTRY = {"vanilla": VanillaDAE, "residual": ResidualDAE}


def build_dae(arch, data_dim, hidden_dim=256, bottleneck_dim=64, device="cpu", **kwargs):
    return DAE_ARCH_REGISTRY[arch](
        data_dim=data_dim, hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim, **kwargs,
    ).to(device)


def train_dae(model, x_clean, epochs=500, lr=1e-3, noise_factor=0.3,
              batch_size=512, **_):
    optimizer = optim.AdamW(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_clean), batch_size=batch_size, shuffle=True)
    losses = []
    pbar = tqdm(range(epochs), desc="DAE")
    for _ in pbar:
        model.train()
        ep = []
        for (batch,) in loader:
            noise = torch.randn_like(batch)
            x_noisy = torch.clamp(batch + noise_factor * noise, 0.0, 1.0)
            recon = model(x_noisy)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep.append(loss.item())
        avg = float(np.mean(ep))
        losses.append(avg)
        pbar.set_postfix(MSE=f"{avg:.6f}")
    return losses


def save_checkpoint(path, model, config, scaler):
    torch.save({"state_dict": model.state_dict(), "config": config,
                "scaler": pickle.dumps(scaler)}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", ckpt.get("cfg"))
    if cfg is None:
        raise KeyError(f"Checkpoint at {path} missing 'config'/'cfg'")
    model = build_dae(
        arch=cfg["arch"], data_dim=cfg["data_dim"],
        hidden_dim=cfg["hidden_dim"], bottleneck_dim=cfg["bottleneck_dim"],
        device=device, n_res_blocks=cfg.get("n_res_blocks", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = pickle.loads(ckpt["scaler"])
    return model, scaler, cfg


# ── magnet ──

logger = logging.getLogger(__name__)

CLIP_MIN, CLIP_MAX, EPS = 0.0, 1.0, 1e-10


class TabularAEDetector:
    def __init__(self, model, scaler, name, p=2, device="cpu"):
        self.model, self.scaler, self.name = model, scaler, name
        self.p, self.device = p, device

    def mark(self, X_raw):
        X_sc = self.scaler.transform(X_raw).astype(np.float32)
        with torch.no_grad():
            recon = self.model(torch.from_numpy(X_sc).to(self.device)).cpu().numpy()
        return np.mean(np.power(np.abs(X_sc - recon), self.p), axis=1)


class TabularReformer:
    def __init__(self, model, scaler, device="cpu"):
        self.model, self.scaler, self.device = model, scaler, device

    def heal(self, X_raw):
        X_sc = self.scaler.transform(X_raw).astype(np.float32)
        with torch.no_grad():
            recon = self.model(torch.from_numpy(X_sc).to(self.device))
        recon = torch.clamp(recon, CLIP_MIN, CLIP_MAX).cpu().numpy()
        return self.scaler.inverse_transform(recon).astype(np.float32)


class MagNetOperator:
    def __init__(self, detectors: dict, reformer: TabularReformer):
        self.detectors = detectors
        self.reformer = reformer
        self.thresholds: dict = {}

    def calibrate(self, X_val_raw, drop_rate):
        for name, det in self.detectors.items():
            marks = det.mark(X_val_raw)
            n_drop = max(1, int(len(marks) * drop_rate))
            thr = float(np.sort(marks)[-n_drop])
            self.thresholds[name] = thr
            logger.info("Det %s thr=%.6f drop=%d/%d", name, thr, n_drop, len(marks))
        return self.thresholds

    def calibrate_optimal(self, X_clean, X_adv, fpr_budget=0.01):
        for name, det in self.detectors.items():
            cm = det.mark(X_clean); am = det.mark(X_adv)
            cands = np.unique(np.percentile(np.concatenate([cm, am]), np.linspace(0, 100, 500)))
            best_thr, best_tpr = None, -1.0
            for t in cands:
                if (cm >= t).mean() > fpr_budget:
                    continue
                tpr = float((am >= t).mean())
                if tpr > best_tpr:
                    best_tpr, best_thr = tpr, t
            if best_thr is None:
                best_thr = float(cm.max()) + EPS
            self.thresholds[name] = best_thr
            fpr = float((cm >= best_thr).mean()) * 100
            tpr = float((am >= best_thr).mean()) * 100
            logger.info("Det %s opt thr=%.6f FPR=%.2f%% TPR=%.2f%%", name, best_thr, fpr, tpr)
        return self.thresholds

    def filter(self, X_raw):
        n = len(X_raw)
        passing = np.arange(n)
        rejects = {}
        for name, det in self.detectors.items():
            marks = det.mark(X_raw)
            ok = np.where(marks < self.thresholds[name])[0]
            before = len(passing)
            passing = np.intersect1d(passing, ok)
            rejects[name] = before - len(passing)
        return passing, rejects

    def defend(self, X_raw, mode="det_only"):
        if mode == "ref_only":
            return self.reformer.heal(X_raw), np.arange(len(X_raw)), 0
        passing, _ = self.filter(X_raw)
        n_rej = len(X_raw) - len(passing)
        if len(passing) == 0:
            return np.empty((0, X_raw.shape[1]), dtype=np.float32), passing, n_rej
        X_pass = X_raw[passing]
        if mode == "det_only":
            return X_pass, passing, n_rej
        return self.reformer.heal(X_pass), passing, n_rej


def _save_ae(save_dir, fname, model, ae_cfg, n_feat, scaler):
    cfg = {"arch": ae_cfg["arch"], "data_dim": n_feat,
           "hidden_dim": int(ae_cfg["hidden_dim"]),
           "bottleneck_dim": int(ae_cfg["bottleneck_dim"]),
           "dropout": float(ae_cfg["dropout"]),
           "noise_factor": float(ae_cfg["noise_factor"])}
    if "n_res_blocks" in ae_cfg:
        cfg["n_res_blocks"] = int(ae_cfg["n_res_blocks"])
    save_checkpoint(os.path.join(save_dir, fname), model, cfg, scaler)


def train_magnet_aes(X_train_raw, config, device="cpu", save_dir=None):
    """Train AE-I (vanilla) + AE-II (residual). Returns (det_i, det_ii, reformer, scaler)."""
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X_train_raw).astype(np.float32)
    x_t = torch.from_numpy(X_sc).to(device)
    n_feat = X_train_raw.shape[1]
    lr, bs = float(config["lr"]), int(config["batch_size"])

    cfg_i, cfg_ii = config["ae_i"], config["ae_ii"]
    logger.info("Training AE-I (%s)", cfg_i["arch"])
    m_i = build_dae(arch=cfg_i["arch"], data_dim=n_feat,
                    hidden_dim=int(cfg_i["hidden_dim"]),
                    bottleneck_dim=int(cfg_i["bottleneck_dim"]),
                    dropout=float(cfg_i["dropout"]), device=device)
    train_dae(m_i, x_t, epochs=int(cfg_i["epochs"]), lr=lr,
              noise_factor=float(cfg_i["noise_factor"]), batch_size=bs)
    m_i.eval()

    logger.info("Training AE-II (%s)", cfg_ii["arch"])
    m_ii = build_dae(arch=cfg_ii["arch"], data_dim=n_feat,
                     hidden_dim=int(cfg_ii["hidden_dim"]),
                     bottleneck_dim=int(cfg_ii["bottleneck_dim"]),
                     n_res_blocks=int(cfg_ii.get("n_res_blocks", 2)),
                     dropout=float(cfg_ii["dropout"]), device=device)
    train_dae(m_ii, x_t, epochs=int(cfg_ii["epochs"]), lr=lr,
              noise_factor=float(cfg_ii["noise_factor"]), batch_size=bs)
    m_ii.eval()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _save_ae(save_dir, "magnet_ae_i.pth", m_i, cfg_i, n_feat, scaler)
        _save_ae(save_dir, "magnet_ae_ii.pth", m_ii, cfg_ii, n_feat, scaler)

    p_i = int(config.get("detector_i_p", 2))
    p_ii = int(config.get("detector_ii_p", 1))
    det_i = TabularAEDetector(m_i, scaler, "I", p=p_i, device=device)
    det_ii = TabularAEDetector(m_ii, scaler, "II", p=p_ii, device=device)
    reformer = TabularReformer(m_ii, scaler, device=device)
    return det_i, det_ii, reformer, scaler


def load_magnet_aes(save_dir, config, device="cpu"):
    p_i = int(config.get("detector_i_p", 2))
    p_ii = int(config.get("detector_ii_p", 1))
    m_i, sc_i, _ = load_checkpoint(os.path.join(save_dir, "magnet_ae_i.pth"), device=device)
    m_ii, sc_ii, _ = load_checkpoint(os.path.join(save_dir, "magnet_ae_ii.pth"), device=device)
    det_i = TabularAEDetector(m_i, sc_i, "I", p=p_i, device=device)
    det_ii = TabularAEDetector(m_ii, sc_ii, "II", p=p_ii, device=device)
    reformer = TabularReformer(m_ii, sc_ii, device=device)
    return det_i, det_ii, reformer


# ── dl_defenses ──

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 20.0):
        super().__init__()
        self.T = float(temperature)

    def forward(self, student_logits, teacher_soft):
        log_p = F.log_softmax(student_logits / self.T, dim=-1)
        return -(teacher_soft * log_p).sum(dim=-1).mean() * (self.T ** 2)


def _wrap(model_cls, net, cfg, device):
    if hasattr(model_cls, "from_distilled_net"):
        return model_cls.from_distilled_net(net, cfg, device)
    out = model_cls()
    out._net = net
    out._cfg = cfg
    out._device = device
    return out


def _teacher_forward(teacher, device):
    """Build a callable that maps raw x → averaged logits.

    Handles models whose underlying net was trained on scaled inputs
    (e.g. FTT with QuantileTransformer): applies ``teacher._scaler`` if
    present, and averages over ``teacher._nets`` if it's a multi-seed
    ensemble. Falls back to ``teacher._net`` directly.
    """
    scaler = getattr(teacher, "_scaler", None)
    nets = getattr(teacher, "_nets", None) or [teacher._net]
    for n in nets:
        n.eval().to(device)

    @torch.no_grad()
    def forward(x):
        if scaler is not None:
            x_np = x.detach().cpu().numpy().astype(np.float32)
            x = torch.from_numpy(scaler.transform(x_np).astype(np.float32)).to(device)
        return torch.stack([n(x) for n in nets]).mean(0)

    return forward


def _teacher_soft_labels(teacher_fwd, X_raw, T, device, batch=4096):
    soft = []
    x_t = torch.from_numpy(X_raw).to(device)
    for i in range(0, len(x_t), batch):
        logits = teacher_fwd(x_t[i:i + batch])
        soft.append(F.softmax(logits / T, dim=-1).cpu().numpy())
    return np.concatenate(soft).astype(np.float32)


class DistilledTrainer:
    """B4 Defensive Distillation."""

    def __init__(self, temperature=20.0, epochs=30, lr=1e-3,
                 batch_size=512, weight_decay=2e-4):
        self.T = float(temperature)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)

    def fit(self, model_cls, net_factory, X_train, y_train, X_val, y_val,
            device="cpu", teacher_cfg=None):
        logger.info("[Distill] Training teacher")
        teacher = model_cls()
        teacher.train(X_train, y_train, X_val=X_val, y_val=y_val,
                      cfg=teacher_cfg, device=device)
        X_raw = X_train.astype(np.float32)
        teacher_fwd = _teacher_forward(teacher, device)
        soft = _teacher_soft_labels(teacher_fwd, X_raw, self.T, device)
        logger.info("[Distill] Soft labels: %s, T=%g", soft.shape, self.T)

        in_dim = X_raw.shape[1]
        n_classes = int(len(np.unique(y_train)))
        cfg = teacher._cfg
        student = net_factory(in_dim, n_classes, cfg, X_raw).to(device)
        criterion = DistillationLoss(temperature=self.T)
        opt = torch.optim.AdamW(student.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_raw), torch.from_numpy(soft)),
            batch_size=self.batch_size, shuffle=True,
        )
        for epoch in range(self.epochs):
            student.train()
            running = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(student(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 3.0)
                opt.step()
                running += float(loss.detach()) * len(xb)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("[Distill] %d/%d loss=%.4f",
                            epoch + 1, self.epochs, running / len(X_raw))
        student.eval()
        return _wrap(model_cls, student, cfg, device)
