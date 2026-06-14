"""Fully connected autoencoder models for MNIST reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BasicAutoencoder(nn.Module):
    """A dense autoencoder that compresses MNIST images into a latent vector.

    The network flattens each 28x28 input image, maps it through a three-layer
    encoder with batch normalization, and reconstructs the image with a
    symmetric decoder ending in a sigmoid so outputs stay in the [0, 1] range.
    """

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
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

    def encode(self, x: Tensor) -> Tensor:
        """Return the latent representation for a batch of images."""
        x = x.view(-1, 784)
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct a batch of MNIST images."""
        x = x.view(-1, 784)
        z = self.encoder(x)
        return self.decoder(z)


class RegularizedAutoencoder(nn.Module):
    """A dense autoencoder with dropout regularization in the encoder.

    This model keeps the same encoder-decoder dimensions as
    :class:`BasicAutoencoder` and inserts dropout after each encoder ReLU to
    reduce overfitting during training.
    """

    def __init__(self, latent_dim: int = 32, dropout_p: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in the interval [0, 1).")

        self.latent_dim = latent_dim
        self.dropout_p = dropout_p
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
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

    def encode(self, x: Tensor) -> Tensor:
        """Return the latent representation for a batch of images."""
        x = x.view(-1, 784)
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct a batch of MNIST images."""
        x = x.view(-1, 784)
        z = self.encoder(x)
        return self.decoder(z)

