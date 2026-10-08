import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Encode diffusion timestep t into a continuous embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        time_bias = self.relu(self.time_mlp(t_emb))[..., None, None]
        h = h + time_bias
        return self.relu(self.conv2(h))


class MiniUNet(nn.Module):
    """Lightweight U-Net used to predict DDPM noise epsilon_theta(x_t, t)."""

    def __init__(self, in_channels=3, base_dim=32, time_emb_dim=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.down1 = Block(in_channels, base_dim, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.down2 = Block(base_dim, base_dim * 2, time_emb_dim)

        self.bot = Block(base_dim * 2, base_dim * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(
            base_dim * 2, base_dim, kernel_size=2, stride=2
        )
        self.up_block = Block(base_dim * 2, base_dim, time_emb_dim)
        self.out_conv = nn.Conv2d(base_dim, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)

        h = self.bot(x2, t_emb)

        h = self.up1(h)
        h = torch.cat([h, x1], dim=1)
        h = self.up_block(h, t_emb)

        return self.out_conv(h)


class DDPM:
    """DDPM forward process and noise-prediction training objective."""

    def __init__(
        self,
        timesteps=200,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
    ):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def _extract(self, values, t, x_shape):
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(
            batch_size,
            *((1,) * (len(x_shape) - 1)),
        )

    def q_sample(self, x_0, t, noise=None):
        """Add Gaussian noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_hat = self._extract(
            torch.sqrt(self.alpha_hat), t, x_0.shape
        )
        sqrt_one_minus_alpha_hat = self._extract(
            torch.sqrt(1.0 - self.alpha_hat), t, x_0.shape
        )

        x_t = (
            sqrt_alpha_hat * x_0
            + sqrt_one_minus_alpha_hat * noise
        )
        return x_t, noise

    def compute_loss(self, model, x_0):
        """DDPM objective: MSE between real and predicted noise."""
        batch_size = x_0.shape[0]

        t = torch.randint(
            0,
            self.timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        x_t, real_noise = self.q_sample(x_0, t)
        predicted_noise = model(x_t, t)

        return F.mse_loss(predicted_noise, real_noise)
