"""InputNorm: per-feature normalization embedded in model architecture.

Adapted from APELID+ (apelid/src/training/dnn.py). Key differences:
- Fits mu/sigma from full training data via explicit fit() call
- No learnable affine (gamma/beta) — FOAMI+ models have BatchNorm after
- Params saved/loaded automatically via state_dict (register_buffer)
"""

import torch
import torch.nn as nn


class InputNorm(nn.Module):
    """Per-feature input normalization as an nn.Module layer.

    mu/sigma are registered as buffers → saved in state_dict, moved with .to(device),
    but not treated as trainable parameters.
    """

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("mu", torch.zeros(num_features))
        self.register_buffer("sigma", torch.ones(num_features))

    def fit(self, X):
        """Fit mu/sigma from training data. Call before training loop.

        Args:
            X: numpy array or tensor of shape (n_samples, num_features)
        """
        X_t = torch.as_tensor(X, dtype=torch.float32)
        self.mu.copy_(X_t.mean(dim=0))
        sigma = X_t.std(dim=0, unbiased=False)
        sigma = torch.where(sigma < self.eps, torch.ones_like(sigma), sigma)
        self.sigma.copy_(sigma)

    def forward(self, x):
        return (x.float() - self.mu) / (self.sigma + self.eps)
