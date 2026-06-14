"""Denoising autoencoder for learning robust MNIST representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DenoisingAutoencoder(nn.Module):
    """Autoencoder trained to reconstruct clean images from corrupted inputs."""

    def __init__(self, latent_dim: int = 32, noise_type: str = "gaussian") -> None:
        super().__init__()
        if noise_type not in {"gaussian", "salt_pepper"}:
            raise ValueError("noise_type must be either 'gaussian' or 'salt_pepper'.")

        self.latent_dim = latent_dim
        self.noise_type = noise_type
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def add_noise(self, x: Tensor, noise_factor: float = 0.3) -> Tensor:
        """Apply Gaussian or salt-and-pepper noise to a batch of images."""
        if not 0.0 <= noise_factor <= 1.0:
            raise ValueError("noise_factor must be in the interval [0, 1].")

        if self.noise_type == "gaussian":
            return torch.clamp(x + noise_factor * torch.randn_like(x), 0.0, 1.0)

        noisy = x.clone()
        mask = torch.rand_like(noisy)
        noisy[mask < noise_factor / 2] = 0.0
        noisy[mask > 1 - noise_factor / 2] = 1.0
        return noisy

    def encode(self, x: Tensor) -> Tensor:
        """Return the latent representation for a clean or noisy batch."""
        x = x.view(-1, 784)
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        """Corrupt the input image and reconstruct the original target shape."""
        x_noisy = self.add_noise(x.clone())
        x_noisy = x_noisy.view(-1, 784)
        z = self.encoder(x_noisy)
        return self.decoder(z)

