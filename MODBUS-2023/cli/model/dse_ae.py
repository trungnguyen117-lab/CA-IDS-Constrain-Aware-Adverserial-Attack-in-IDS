"""Autoencoder-based DSE for adversarial sample detection.

Trains an autoencoder on clean data only. At inference, adversarial samples
have higher reconstruction error than clean samples -> detected as anomalous.
"""

import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class TabularAE(nn.Module):
    """Symmetric autoencoder for tabular features."""

    def __init__(self, n_features, bottleneck_dim=8, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Encoder
        enc_layers = []
        in_dim = n_features
        for h in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ])
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        in_dim = bottleneck_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class DSEAutoencoder:
    """Autoencoder-based anomaly detector for adversarial samples.

    Not a classifier -- does not extend BaseModel.
    """

    def __init__(self, n_features=58, bottleneck_dim=8, hidden_dims=None,
                 device="cpu"):
        self.n_features = n_features
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.ae = TabularAE(n_features, bottleneck_dim, hidden_dims).to(device)
        self._scaler = None

    def _scale(self, X):
        """Scale input using fitted StandardScaler."""
        return self._scaler.transform(X).astype(np.float32)

    def train(self, X_train, cfg=None):
        """Train autoencoder on clean data with MSE reconstruction loss."""
        if cfg is None:
            cfg = {}

        lr = cfg.get("lr", 1e-3)
        weight_decay = cfg.get("weight_decay", 1e-4)
        batch_size = cfg.get("batch_size", 256)
        max_epochs = cfg.get("max_epochs", 100)
        patience = cfg.get("patience", 20)

        # Fit scaler on training data
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train).astype(np.float32)

        dataset = TensorDataset(
            torch.from_numpy(X_scaled),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=lr, weight_decay=weight_decay,
        )

        best_loss, best_state, wait = float("inf"), None, 0

        for epoch in range(max_epochs):
            self.ae.train()
            total_loss = 0.0
            n_batches = 0

            for (x,) in loader:
                x = x.to(self.device)
                x_hat = self.ae(x)
                loss = criterion(x_hat, x)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                wait = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.ae.state_dict().items()
                }
            else:
                wait += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"AE epoch {epoch+1}/{max_epochs}, "
                    f"loss={avg_loss:.6f}, best={best_loss:.6f}"
                )

            if wait >= patience:
                logger.info(
                    f"AE early stop at epoch {epoch+1}, "
                    f"best loss={best_loss:.6f}"
                )
                break

        if best_state is not None:
            self.ae.load_state_dict(best_state)

        self.ae.eval()
        logger.info(f"AE trained. Best MSE={best_loss:.6f}")

    @torch.no_grad()
    def reconstruction_error(self, X):
        """Per-sample MSE reconstruction error in scaled space. Returns (N,) ndarray."""
        self.ae.eval()
        X_scaled = self._scale(X)
        errors = []
        bs = 4096
        for i in range(0, len(X_scaled), bs):
            x = torch.from_numpy(X_scaled[i:i + bs]).to(self.device)
            x_hat = self.ae(x)
            err = (x - x_hat).pow(2).mean(dim=1).cpu().numpy()
            errors.append(err)
        return np.concatenate(errors, axis=0)

    @torch.no_grad()
    def purify(self, X):
        """Reconstruct input through AE and inverse-scale back to raw space.

        raw X → scale → AE reconstruct → inverse scale → purified raw X
        """
        self.ae.eval()
        X_scaled = self._scale(X)
        parts = []
        bs = 4096
        for i in range(0, len(X_scaled), bs):
            x = torch.from_numpy(X_scaled[i:i + bs]).to(self.device)
            x_hat = self.ae(x).cpu().numpy()
            parts.append(x_hat)
        X_hat_scaled = np.concatenate(parts, axis=0)
        return self._scaler.inverse_transform(X_hat_scaled).astype(np.float32)

    def save(self, path):
        ckpt = {
            "n_features": self.n_features,
            "bottleneck_dim": self.bottleneck_dim,
            "hidden_dims": self.hidden_dims,
            "state_dict": self.ae.state_dict(),
            "scaler_bytes": pickle.dumps(self._scaler),
        }
        torch.save(ckpt, path)
        logger.info(f"AE saved to {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            n_features=ckpt["n_features"],
            bottleneck_dim=ckpt["bottleneck_dim"],
            hidden_dims=ckpt["hidden_dims"],
            device=device,
        )
        model.ae.load_state_dict(ckpt["state_dict"])
        model.ae.eval()
        model._scaler = pickle.loads(ckpt["scaler_bytes"])
        logger.info(f"AE loaded from {path}")
        return model
