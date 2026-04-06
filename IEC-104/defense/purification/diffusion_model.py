"""DDPM diffusion model for adversarial purification.

Adapted from adversarial-purification/diffusion.py for tabular IDS data.
Trains on MinMax-scaled [0,1] clean data, purifies adversarial examples
by adding noise then iteratively denoising via learned reverse process.

Architectures:
  - MLP: Original 10-layer plain MLP (baseline)
  - ResidualMLP: Residual blocks + LayerNorm (better gradient flow)
  - UNetMLP: Encoder-decoder with skip connections (preserves input structure)
"""

import pickle

import torch
from torch import nn, optim
from tqdm import tqdm


# ── Diffusion Process (shared across all architectures) ──────────────────────


class Diffusion:
    """DDPM forward/reverse process with linear beta schedule."""

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = torch.linspace(float(beta_start), float(beta_end), noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_data(self, x, t):
        """Forward process: add noise at timestep t."""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def reconstruct(self, model, x, t, progress_bar=True):
        """Reverse process: denoise from timestep t back to 0 (DDPM Algorithm 2)."""
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(1, t), position=0, disable=not progress_bar):
                j = t - i
                ts = (torch.ones(x.shape[0]) * j).int().to(self.device)
                predicted_noise = model(x, ts)
                alpha = self.alpha[ts][:, None]
                alpha_hat = self.alpha_hat[ts][:, None]
                beta = self.beta[ts][:, None]
                noise = torch.randn_like(x) if j > 1 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha)) * \
                    (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + \
                    torch.sqrt(beta) * noise
            model.train()
            return x

    def purify(self, model, x, t, progress_bar=True):
        """Add noise at timestep t then denoise. Clips output to [0,1]."""
        ts = (torch.ones(x.shape[0]) * (t - 1)).long().to(self.device)
        x_t, _ = self.noise_data(x, ts)
        x_purified = self.reconstruct(model, x_t, t, progress_bar=progress_bar)
        return torch.clamp(x_purified, 0.0, 1.0)


# ── Timestep Embedding (shared) ─────────────────────────────────────────────


class TimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding + projection for timestep conditioning."""

    def __init__(self, emb_dim, hidden_dim, device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, hidden_dim))

    def forward(self, t):
        t = t.unsqueeze(-1).float()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.emb_dim, 2, device=self.device).float()
                      / self.emb_dim)
        )
        pe_sin = torch.sin(t.repeat(1, self.emb_dim // 2) * inv_freq)
        pe_cos = torch.cos(t.repeat(1, self.emb_dim // 2) * inv_freq)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return self.proj(pe)


# ── Architecture 1: Original MLP (baseline) ─────────────────────────────────


class MLP(nn.Module):
    """10-layer noise estimation network with sinusoidal timestep embedding."""

    def __init__(self, data_dim, hidden_dim=512, emb_dim=256, device="cuda"):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device

        self.l1 = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l6 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l7 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l8 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l9 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l10 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.l11 = nn.Linear(hidden_dim, data_dim)

        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, hidden_dim))

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.emb_dim)
        t = self.emb_layer(t)
        a1 = self.l1(x) + t
        a2 = self.l2(a1) + t
        a3 = self.l3(a2) + t
        a4 = self.l4(a3) + t
        a5 = self.l5(a4) + t
        a6 = self.l6(a5) + t
        a7 = self.l7(a6) + t
        a8 = self.l8(a7) + t
        a9 = self.l9(a8) + t
        a10 = self.l10(a9) + t
        return self.l11(a10)


# ── Architecture 2: Residual MLP ────���───────────────────────────────────────


class _ResBlock(nn.Module):
    """Residual block: LayerNorm → Linear → SiLU → Linear + skip + time emb."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, t_emb):
        return x + self.net(self.norm(x)) + t_emb


class ResidualMLP(nn.Module):
    """Residual MLP with LayerNorm — better gradient flow for tabular data.

    Architecture: input_proj → N residual blocks → output_proj
    Each block has a skip connection + timestep injection.
    """

    def __init__(self, data_dim, hidden_dim=256, emb_dim=128, n_blocks=8,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.t_emb = TimestepEmbedding(emb_dim, hidden_dim, device)
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList([_ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        t_emb = self.t_emb(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(self.output_norm(h))


# ── Architecture 3: U-Net MLP ─────��──────────────────────────────��──────────


class UNetMLP(nn.Module):
    """U-Net style encoder-decoder MLP with skip connections.

    Encoder compresses features, decoder expands back.
    Skip connections at each level preserve input structure — critical for
    tabular data where small feature changes matter.

    Levels: data_dim → h1 → h2 → h3 (bottleneck) → h2 → h1 → data_dim
    """

    def __init__(self, data_dim, hidden_dim=256, emb_dim=128, device="cpu"):
        super().__init__()
        self.device = device
        h1, h2, h3 = hidden_dim, hidden_dim * 2, hidden_dim * 4

        self.t_emb = TimestepEmbedding(emb_dim, h1, device)
        # Time projection for each level
        self.t_proj2 = nn.Linear(h1, h2)
        self.t_proj3 = nn.Linear(h1, h3)

        # Encoder
        self.enc1 = nn.Sequential(nn.Linear(data_dim, h1), nn.LayerNorm(h1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Linear(h1, h2), nn.LayerNorm(h2), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Linear(h2, h3), nn.LayerNorm(h3), nn.SiLU())

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(h3, h3), nn.LayerNorm(h3), nn.SiLU(),
            nn.Linear(h3, h3), nn.LayerNorm(h3), nn.SiLU(),
        )

        # Decoder (input is concat of skip + decoded, so double width)
        self.dec3 = nn.Sequential(nn.Linear(h3 * 2, h2), nn.LayerNorm(h2), nn.SiLU())
        self.dec2 = nn.Sequential(nn.Linear(h2 * 2, h1), nn.LayerNorm(h1), nn.SiLU())
        self.dec1 = nn.Linear(h1 * 2, data_dim)

    def forward(self, x, t):
        t1 = self.t_emb(t)           # (B, h1)
        t2 = self.t_proj2(t1)        # (B, h2)
        t3 = self.t_proj3(t1)        # (B, h3)

        # Encode
        e1 = self.enc1(x) + t1       # (B, h1)
        e2 = self.enc2(e1) + t2      # (B, h2)
        e3 = self.enc3(e2) + t3      # (B, h3)

        # Bottleneck
        b = self.bottleneck(e3) + t3  # (B, h3)

        # Decode with skip connections
        d3 = self.dec3(torch.cat([b, e3], dim=-1))    # (B, h2)
        d2 = self.dec2(torch.cat([d3, e2], dim=-1))   # (B, h1)
        d1 = self.dec1(torch.cat([d2, e1], dim=-1))   # (B, data_dim)
        return d1


# ── Model Registry ──────────────────────────────────────────────────────────

ARCH_REGISTRY = {
    "mlp": MLP,
    "residual": ResidualMLP,
    "unet": UNetMLP,
}


def build_model(arch, data_dim, hidden_dim=256, emb_dim=128, device="cpu", **kwargs):
    """Build noise estimation model by architecture name."""
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch: {arch!r}. Available: {list(ARCH_REGISTRY.keys())}")
    cls = ARCH_REGISTRY[arch]
    return cls(data_dim=data_dim, hidden_dim=hidden_dim, emb_dim=emb_dim,
               device=device, **kwargs).to(device)


# ── Training ────────────────────────────────────────────��───────────────────


def train_diffusion(model, diffusion, x_train, epochs=5000, lr=1e-3, device="cuda",
                    log_interval=500):
    """Train diffusion model on clean data. Returns list of per-epoch MSE losses."""
    optimizer = optim.AdamW(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    train_loss = []

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        t = diffusion.sample_timesteps(x_train.shape[0]).to(device)
        x_t, noise = diffusion.noise_data(x_train, t)
        predicted_noise = model(x_t, t)
        loss = loss_fn(noise, predicted_noise)

        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(MSE=train_loss[-1])

    return train_loss


# ── Checkpointing ─────────────────────────────────────��─────────────────────


def save_checkpoint(path, model, config, train_loss, scaler):
    """Save diffusion model + MinMaxScaler + config to a single .pth file."""
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "train_loss": train_loss,
        "scaler": pickle.dumps(scaler),
    }, path)


def load_checkpoint(path, device="cpu"):
    """Load checkpoint. Returns (model, diffusion, scaler, config)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    arch = cfg.get("arch", "mlp")
    extra = {}
    if arch == "residual" and "n_blocks" in cfg:
        extra["n_blocks"] = cfg["n_blocks"]

    model = build_model(
        arch=arch,
        data_dim=cfg["data_dim"],
        hidden_dim=cfg["hidden_dim"],
        emb_dim=cfg["emb_dim"],
        device=device,
        **extra,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    diffusion = Diffusion(
        noise_steps=cfg["noise_steps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        device=device,
    )

    scaler = pickle.loads(ckpt["scaler"])
    return model, diffusion, scaler, cfg
