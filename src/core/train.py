"""Training logic: baseline + adversarial training + distillation (no CLI)."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import yaml

from .data import load_dataset
from .defenses import DistilledTrainer
from .metrics import assemble_at_data, report_metrics
from .models import build_model

logger = logging.getLogger(__name__)


# ── Baseline ─────────────────────────────────────────────────────────────────


def train_baseline(cfg, target, X_tr, y_tr, X_te, y_te, device, out_dir):
    logger.info("=" * 50)
    logger.info("Training: %s", target.upper())
    logger.info("=" * 50)
    train_yaml = cfg.cfg_yaml("training", target)
    if train_yaml:
        logger.info("Config: %s", train_yaml)
    m = build_model(cfg, target)
    m.train(X_tr, y_tr, X_te, y_te, cfg=train_yaml or None, device=device)
    preds = m.predict(X_te)
    report_metrics(f"{target.upper()} baseline", y_te, preds)
    path = cfg.model_path(target, base_dir=out_dir)
    m.save(str(path))
    logger.info("Saved → %s", path)


def run_baseline_training(cfg, models, device, out_dir, train_path, test_path):
    """Train each baseline model from config, save to out_dir."""
    _, X_tr, y_tr, _ = load_dataset(cfg, train_path or "train")
    _, X_te, y_te, _ = load_dataset(cfg, test_path or "test")
    logger.info("Train: %s, Test: %s", X_tr.shape, X_te.shape)
    out = out_dir or cfg.paths.models
    os.makedirs(out, exist_ok=True)
    for name in models:
        train_baseline(cfg, name, X_tr, y_tr, X_te, y_te, device, out)


# ── Adversarial training (AT / PGD-AT) ───────────────────────────────────────


def load_attack_weights(cfg, target):
    """Per-attack mix weights for AT — reads ``config/at_weights/<target>.yaml``,
    falls back to ``cfg.at_weights[target]``."""
    path = cfg.root / "config" / "at_weights" / f"{target}.yaml"
    if path.is_file():
        with open(path) as f:
            w = (yaml.safe_load(f) or {}).get("attack_weights")
        if w:
            logger.info("Loaded weights for %s from %s: %s", target, path, w)
            return w
    return cfg.at_weights.get(target)


def train_adv_target(cfg, target, train_path, test_path, out_dir, device,
                     defense="at",
                     adv_base=None, balance_adv=True, attack_weights=None,
                     config_variant=None, per_attack_cap=None,
                     adv_clean_ratio=1.0, attack_filter=None):
    """Assemble (clean + adv) set and train one target → save to out_dir."""
    _, X_clean, y_clean, feats = load_dataset(cfg, train_path)
    _, X_te, y_te, _ = load_dataset(cfg, test_path)

    df_at = assemble_at_data(
        cfg, target, X_clean, y_clean, feats,
        adv_base=adv_base, balance_adv=balance_adv,
        attack_filter=attack_filter,
        attack_weights=attack_weights,
        per_attack_cap=per_attack_cap,
        adv_clean_ratio=adv_clean_ratio,
    )
    Xa = df_at[feats].values.astype(np.float32)
    ya = df_at[cfg.label_col].values.astype(int)

    m = build_model(cfg, target)
    train_yaml = cfg.cfg_yaml("training", target if not config_variant
                                          else f"{target}_{config_variant}")
    if train_yaml:
        logger.info("Config %s: %s", target, train_yaml)

    t0 = time.time()
    m.train(Xa, ya, cfg=train_yaml or None, device=device)
    logger.info("Fit time: %.1fs", time.time() - t0)

    preds = m.predict(X_te)
    metrics = report_metrics(f"{target.upper()} {defense.upper()}", y_te, preds)

    os.makedirs(out_dir, exist_ok=True)
    out = cfg.model_path(target, defense=defense, base_dir=out_dir)
    m.save(str(out))
    return metrics


def net_factory(cfg, target):
    """Return classmethod ``build_net(in_dim, n_classes, cfg, X_raw) → nn.Module``
    declared on the dataset's DL model class."""
    m = build_model(cfg, target)
    cls = type(m)
    if not hasattr(cls, "build_net"):
        raise AttributeError(
            f"{cls.__name__} needs a classmethod build_net(in_dim, n_classes, cfg, X_raw)"
        )
    return cls.build_net


def train_distill(cfg, target, train_path, test_path, out_dir, device,
                  temperature=20.0, epochs=30):
    """Train one DL target with defensive distillation → save to out_dir."""
    m_cls = type(build_model(cfg, target))

    _, X_tr, y_tr, _ = load_dataset(cfg, train_path or "train")
    _, X_te, y_te, _ = load_dataset(cfg, test_path or "test")

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr, va = next(sss.split(X_tr, y_tr))
    trainer = DistilledTrainer(temperature=temperature, epochs=epochs)
    out_model = trainer.fit(
        m_cls, net_factory(cfg, target),
        X_tr[tr], y_tr[tr], X_tr[va], y_tr[va], device=device,
    )
    preds = out_model.predict(X_te)
    report_metrics(f"{target.upper()} DISTILL", y_te, preds)
    os.makedirs(out_dir, exist_ok=True)
    out_path = cfg.model_path(target, defense="distill", base_dir=out_dir)
    out_model.save(str(out_path))
