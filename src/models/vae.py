"""Variational autoencoder model and loss for MNIST."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VAE(nn.Module):
    """A fully connected variational autoencoder for 28x28 MNIST digits."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return latent distribution parameters ``mu`` and ``log_var``."""
        x = x.view(-1, 784)
        hidden = self.encoder(x)
        return self.mu(hidden), self.log_var(hidden)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample latent vectors using the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vectors into flattened MNIST reconstructions."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return reconstruction, mean, and log variance for an input batch."""
        x = x.view(-1, 784)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, n: int, device: torch.device | str) -> Tensor:
        """Generate ``n`` digit-like samples from a standard normal prior."""
        z = torch.randn(n, self.latent_dim).to(device)
        return self.decode(z)


def vae_loss(recon_x: Tensor, x: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """Return binary cross-entropy plus KL divergence for a VAE batch."""
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld

