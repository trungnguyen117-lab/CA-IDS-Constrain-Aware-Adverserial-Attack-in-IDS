"""Denoising Autoencoder (DAE) for adversarial purification.

Trains on MinMax-scaled [0,1] clean data with Gaussian noise corruption.
At inference, adversarial inputs pass through the trained encoder-decoder
in a single forward pass to remove adversarial perturbations.

Architectures:
  - VanillaDAE: Plain encoder-decoder with bottleneck (baseline)
  - ResidualDAE: U-Net style with skip connections (preserves fine-grained features)
"""

import pickle
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ── Architecture 1: Vanilla DAE (baseline) ──────────────────────────────────


class VanillaDAE(nn.Module):
    """Plain encoder-decoder with bottleneck.

    Encoder: data_dim → hidden → hidden//2 → bottleneck
    Decoder: bottleneck → hidden//2 → hidden → data_dim (Sigmoid)
    """

    def __init__(self, data_dim, hidden_dim=256, bottleneck_dim=64, dropout=0.1,
                 **kwargs):
        super().__init__()
        mid = hidden_dim // 2

        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid),
            nn.LayerNorm(mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, mid),
            nn.LayerNorm(mid),
            nn.SiLU(),
            nn.Linear(mid, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Residual block (shared) ─────────────────────────────────────────────────


class _DAEResBlock(nn.Module):
    """Residual block: LayerNorm → Linear → SiLU → Linear + skip."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


# ── Architecture 2: Residual DAE (U-Net style) ──────────────────────────────


class ResidualDAE(nn.Module):
    """Encoder-decoder with skip connections for tabular data.

    Encoder: data_dim → h1 → h2 → bottleneck (with residual blocks)
    Decoder: bottleneck → cat(h2, skip) → cat(h1, skip) → data_dim
    Skip connections preserve fine-grained feature information.
    """

    def __init__(self, data_dim, hidden_dim=256, bottleneck_dim=64,
                 n_res_blocks=2, dropout=0.1, **kwargs):
        super().__init__()
        h1 = hidden_dim
        h2 = hidden_dim // 2

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Linear(data_dim, h1), nn.LayerNorm(h1), nn.SiLU(), nn.Dropout(dropout),
        )
        self.enc1_res = nn.ModuleList([_DAEResBlock(h1) for _ in range(n_res_blocks)])

        self.enc2 = nn.Sequential(
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.SiLU(), nn.Dropout(dropout),
        )
        self.enc2_res = nn.ModuleList([_DAEResBlock(h2) for _ in range(n_res_blocks)])

        self.enc_bottleneck = nn.Sequential(
            nn.Linear(h2, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.SiLU(),
        )

        # Decoder (skip connections double the input width)
        self.dec_bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, h2), nn.LayerNorm(h2), nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Linear(h2 * 2, h1), nn.LayerNorm(h1), nn.SiLU(),
        )
        self.dec2_res = nn.ModuleList([_DAEResBlock(h1) for _ in range(n_res_blocks)])

        self.dec1 = nn.Sequential(
            nn.Linear(h1 * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.dec1_res = nn.ModuleList([_DAEResBlock(hidden_dim) for _ in range(n_res_blocks)])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        for blk in self.enc1_res:
            e1 = blk(e1)

        e2 = self.enc2(e1)
        for blk in self.enc2_res:
            e2 = blk(e2)

        z = self.enc_bottleneck(e2)

        # Decode with skip connections
        d2 = self.dec_bottleneck(z)
        d2 = self.dec2(torch.cat([d2, e2], dim=-1))
        for blk in self.dec2_res:
            d2 = blk(d2)

        d1 = self.dec1(torch.cat([d2, e1], dim=-1))
        for blk in self.dec1_res:
            d1 = blk(d1)

        return self.output_proj(d1)


# ── Model Registry ──────────────────────────────────────────────────────────

DAE_ARCH_REGISTRY = {
    "vanilla": VanillaDAE,
    "residual": ResidualDAE,
}


def build_dae(arch, data_dim, hidden_dim=256, bottleneck_dim=64, device="cpu",
              **kwargs):
    """Build DAE model by architecture name."""
    if arch not in DAE_ARCH_REGISTRY:
        raise ValueError(
            f"Unknown arch: {arch!r}. Available: {list(DAE_ARCH_REGISTRY.keys())}"
        )
    cls = DAE_ARCH_REGISTRY[arch]
    return cls(data_dim=data_dim, hidden_dim=hidden_dim,
               bottleneck_dim=bottleneck_dim, **kwargs).to(device)


# ── Training ────────────────────────────────────────────────────────────────


def train_dae(model, x_clean, epochs=2000, lr=1e-3, noise_factor=0.3,
              noise_schedule=None, batch_size=512, **kwargs):
    """Train DAE on clean data with Gaussian noise corruption.

    For each batch:
      1. Sample noise_factor from noise_schedule (if provided) or use fixed
      2. x_noisy = clamp(x_clean + nf * N(0,1), 0, 1)
      3. x_recon = model(x_noisy)
      4. loss = MSE(x_recon, x_clean)

    Returns list of per-epoch average losses.
    """
    optimizer = optim.AdamW(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(x_clean)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    train_loss = []
    pbar = tqdm(range(epochs))
    for _ in pbar:
        model.train()
        epoch_losses = []

        # Pick noise level for this epoch
        if noise_schedule:
            nf = random.choice(noise_schedule)
        else:
            nf = noise_factor

        for (batch,) in loader:
            noise = torch.randn_like(batch)
            x_noisy = torch.clamp(batch + nf * noise, 0.0, 1.0)
            x_recon = model(x_noisy)
            loss = loss_fn(x_recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        train_loss.append(avg_loss)
        pbar.set_postfix(MSE=f"{avg_loss:.6f}", nf=f"{nf:.2f}")

    return train_loss


# ── Checkpointing ───────────────────────────────────────────────────────────


def save_checkpoint(path, model, config, train_loss, scaler):
    """Save DAE model + MinMaxScaler + config to a single .pth file."""
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "train_loss": train_loss,
        "scaler": pickle.dumps(scaler),
    }, path)


def load_checkpoint(path, device="cpu"):
    """Load checkpoint. Returns (model, scaler, config)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = build_dae(
        arch=cfg["arch"],
        data_dim=cfg["data_dim"],
        hidden_dim=cfg["hidden_dim"],
        bottleneck_dim=cfg["bottleneck_dim"],
        device=device,
        n_res_blocks=cfg.get("n_res_blocks", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = pickle.loads(ckpt["scaler"])
    return model, scaler, cfg
