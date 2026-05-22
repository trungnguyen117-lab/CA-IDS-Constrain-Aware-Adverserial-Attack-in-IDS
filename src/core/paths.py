"""Paths, device resolution, input normalization, logging."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn


def resolve_device(arg: str) -> str:
    """``auto`` → cuda > mps > cpu; pass through otherwise."""
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_arg(cfg, value, default=None, *, as_str: bool = False):
    """Resolve an optional CLI path against ``cfg.root``."""
    path = cfg.resolve(value) if value else default
    return str(path) if as_str and path is not None else path


# ── input_norm ──

class InputNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mu", torch.zeros(num_features))
        self.register_buffer("sigma", torch.ones(num_features))
        self._fitted = False

    def fit(self, X):
        if isinstance(X, np.ndarray):
            mu = torch.from_numpy(X.mean(axis=0).astype(np.float32))
            sigma = torch.from_numpy(X.std(axis=0).astype(np.float32))
        else:
            mu = X.float().mean(dim=0).cpu()
            sigma = X.float().std(dim=0, unbiased=False).cpu()
        sigma = torch.where(sigma < self.eps, torch.ones_like(sigma), sigma)
        self.mu.copy_(mu.to(self.mu.device))
        self.sigma.copy_(sigma.to(self.sigma.device))
        self._fitted = True
        return self

    def forward(self, x):
        return (x - self.mu) / self.sigma


# ── logging ──

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


