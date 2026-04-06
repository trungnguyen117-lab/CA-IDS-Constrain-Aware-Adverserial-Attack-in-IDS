"""Deep Similarity Encoder (DSE) for adversarial query detection.

Based on Tiki-Taka (Zhang et al., ToN 2022). Trains an encoder with
contrastive loss to map inputs into a low-dimensional embedding space
where similar queries cluster together, enabling detection of iterative
black-box adversarial attacks.
"""

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ── Encoder architectures ──────────────────────────────────────────────────


class MLPEncoder(nn.Module):
    """MLP-based DSE encoder: tabular features -> low-dim embedding."""

    def __init__(self, n_features, embedding_dim=3, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        e = self.net(x)
        return nn.functional.normalize(e, p=2, dim=1)


class Conv1DEncoder(nn.Module):
    """1D CNN DSE encoder: treat tabular features as 1D signal."""

    def __init__(self, n_features, embedding_dim=3, channels=None,
                 kernel_size=3):
        super().__init__()
        if channels is None:
            channels = [32, 64, 32]

        conv_layers = []
        in_ch = 1
        for out_ch in channels:
            pad = kernel_size // 2
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Linear(channels[-1] * n_features, embedding_dim)

    def forward(self, x):
        # x: (B, F) -> (B, 1, F)
        h = self.conv(x.unsqueeze(1))
        h = h.flatten(1)
        e = self.head(h)
        return nn.functional.normalize(e, p=2, dim=1)


# ── Contrastive loss ───────────────────────────────────────────────────────


class ContrastiveLoss(nn.Module):
    """Contrastive loss from Tiki-Taka paper.

    L = ||e_i - e_tilde||^2 + max(0, margin^2 - ||e_m - e_n||^2)
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin_sq = margin ** 2

    def forward(self, e_sim_a, e_sim_b, e_dis_a, e_dis_b):
        # Similar pair: minimize distance
        loss_sim = (e_sim_a - e_sim_b).pow(2).sum(dim=1).mean()
        # Dissimilar pair: maximize distance (up to margin)
        d_dis = (e_dis_a - e_dis_b).pow(2).sum(dim=1)
        loss_dis = torch.clamp(self.margin_sq - d_dis, min=0).mean()
        return loss_sim + loss_dis


# ── Training dataset ───────────────────────────────────────────────────────


class ContrastiveDataset(torch.utils.data.Dataset):
    """Generate contrastive pairs on-the-fly.

    Similar pair: (x_i, x_i + alpha * noise)
    Dissimilar pair: (x_i, x_j) where j != i
    """

    def __init__(self, X, noise_scale=0.15):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.noise_scale = noise_scale
        self.n = len(X)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.X[idx]
        # Similar: x + Gaussian noise
        noise = torch.randn_like(x) * self.noise_scale * (x.abs() + 1e-8)
        x_sim = x + noise
        # Dissimilar: random different sample
        j = torch.randint(0, self.n - 1, (1,)).item()
        if j >= idx:
            j += 1
        x_dis = self.X[j]
        return x, x_sim, x_dis


# ── DSE model wrapper ──────────────────────────────────────────────────────


class DSEModel:
    """High-level DSE wrapper with train/encode/save/load.

    Not a classifier — does not extend BaseModel.
    """

    def __init__(self, encoder_type="mlp", n_features=58, embedding_dim=3,
                 hidden_dims=None, cnn_channels=None, cnn_kernel_size=3,
                 device="cpu"):
        self.encoder_type = encoder_type
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.device = device

        if encoder_type == "mlp":
            self.encoder = MLPEncoder(
                n_features, embedding_dim, hidden_dims,
            ).to(device)
        elif encoder_type == "1dcnn":
            self.encoder = Conv1DEncoder(
                n_features, embedding_dim, cnn_channels, cnn_kernel_size,
            ).to(device)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def train(self, X_train, cfg=None):
        """Train the DSE encoder with contrastive loss."""
        if cfg is None:
            cfg = {}

        margin = cfg.get("margin", 0.5)
        noise_scale = cfg.get("noise_scale", 0.15)
        lr = cfg.get("lr", 0.001)
        weight_decay = cfg.get("weight_decay", 0.0001)
        batch_size = cfg.get("batch_size", 256)
        max_epochs = cfg.get("max_epochs", 100)
        patience = cfg.get("patience", 20)

        dataset = ContrastiveDataset(X_train, noise_scale=noise_scale)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)

        criterion = ContrastiveLoss(margin=margin)
        optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay,
        )

        best_loss, best_state, wait = float("inf"), None, 0

        for epoch in range(max_epochs):
            self.encoder.train()
            total_loss = 0.0
            n_batches = 0

            for x, x_sim, x_dis in loader:
                x = x.to(self.device)
                x_sim = x_sim.to(self.device)
                x_dis = x_dis.to(self.device)

                e_x = self.encoder(x)
                e_sim = self.encoder(x_sim)
                e_dis = self.encoder(x_dis)

                loss = criterion(e_x, e_sim, e_x, e_dis)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                wait = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.encoder.state_dict().items()
                }
            else:
                wait += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"DSE epoch {epoch+1}/{max_epochs}, "
                    f"loss={avg_loss:.6f}, best={best_loss:.6f}"
                )

            if wait >= patience:
                logger.info(
                    f"DSE early stop at epoch {epoch+1}, "
                    f"best loss={best_loss:.6f}"
                )
                break

        if best_state is not None:
            self.encoder.load_state_dict(best_state)

        # Log embedding stats
        self.encoder.eval()
        with torch.no_grad():
            sample = torch.from_numpy(
                X_train[:min(2000, len(X_train))].astype(np.float32)
            ).to(self.device)
            emb = self.encoder(sample).cpu().numpy()
            logger.info(
                f"DSE trained. Embedding stats: "
                f"mean_norm={np.linalg.norm(emb, axis=1).mean():.4f}, "
                f"mean_dist={np.mean(np.linalg.norm(emb[:-1] - emb[1:], axis=1)):.6f}"
            )

    @torch.no_grad()
    def encode(self, X):
        """Encode input to embedding space. Returns (N, embedding_dim) ndarray."""
        self.encoder.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        embeddings = []
        # Process in batches to avoid OOM
        bs = 4096
        for i in range(0, len(X_t), bs):
            batch = X_t[i:i + bs]
            emb = self.encoder(batch).cpu().numpy()
            embeddings.append(emb)
        return np.concatenate(embeddings, axis=0)

    def save(self, path):
        """Save DSE encoder checkpoint."""
        ckpt = {
            "encoder_type": self.encoder_type,
            "n_features": self.n_features,
            "embedding_dim": self.embedding_dim,
            "state_dict": self.encoder.state_dict(),
        }
        torch.save(ckpt, path)
        logger.info(f"DSE saved to {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        """Load DSE encoder from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            encoder_type=ckpt["encoder_type"],
            n_features=ckpt["n_features"],
            embedding_dim=ckpt["embedding_dim"],
            device=device,
        )
        model.encoder.load_state_dict(ckpt["state_dict"])
        model.encoder.eval()
        logger.info(f"DSE loaded from {path}")
        return model
