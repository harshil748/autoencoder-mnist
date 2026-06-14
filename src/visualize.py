"""Visualization helpers for reconstruction and latent-space analysis."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader

try:
    from .models import DenoisingAutoencoder, VAE
except ImportError:
    from models import DenoisingAutoencoder, VAE


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _safe_title(title: str, default: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.strip()).strip("_")
    return slug or default


def _ensure_path(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_reconstruction(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def plot_reconstructions(
    model: nn.Module,
    test_loader: DataLoader,
    device: str | torch.device,
    n: int = 10,
    title: str = "",
    save_path: str | Path = "results/plots/",
) -> Path:
    """Save a two-row plot of original and reconstructed MNIST images."""
    device_obj = _resolve_device(device)
    model.to(device_obj)
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:n].to(device_obj)

    with torch.no_grad():
        recon = _as_reconstruction(model(images)).view(-1, 1, 28, 28)

    n = min(n, images.size(0))
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.2))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for idx in range(n):
        axes[0, idx].imshow(images[idx].cpu().squeeze(), cmap="gray")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(recon[idx].cpu().squeeze(), cmap="gray")
        axes[1, idx].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Recon", fontsize=10)
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    output_path = _ensure_path(save_path) / f"{_safe_title(title, 'model')}_reconstructions.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_noisy_comparison(
    denoising_model: DenoisingAutoencoder,
    test_loader: DataLoader,
    device: str | torch.device,
    n: int = 8,
    save_path: str | Path = "results/plots/",
) -> Path:
    """Save original, noisy, and reconstructed images for a denoising model."""
    device_obj = _resolve_device(device)
    denoising_model.to(device_obj)
    denoising_model.eval()
    images, _ = next(iter(test_loader))
    images = images[:n].to(device_obj)

    with torch.no_grad():
        noisy = denoising_model.add_noise(images.clone())
        latent = denoising_model.encoder(noisy.view(-1, 784))
        recon = denoising_model.decoder(latent).view(-1, 1, 28, 28)

    n = min(n, images.size(0))
    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 4.8))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    rows = [("Original", images), ("Noisy", noisy), ("Recon", recon)]
    for row_idx, (label, batch) in enumerate(rows):
        for col_idx in range(n):
            axes[row_idx, col_idx].imshow(batch[col_idx].cpu().squeeze(), cmap="gray")
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(label, fontsize=10)

    fig.tight_layout()
    output_path = _ensure_path(save_path) / "denoising_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_curves(
    history: dict,
    title: str = "",
    save_path: str | Path = "results/plots/",
) -> Path:
    """Save train and validation loss curves from a training history."""
    epochs = np.arange(1, len(history.get("train_losses", [])) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history.get("train_losses", []), color="tab:blue", label="Train loss")
    ax.plot(epochs, history.get("val_losses", []), color="tab:orange", label="Val loss")

    if history.get("stopped_early") and history.get("best_epoch"):
        ax.axvline(history["best_epoch"], color="gray", linestyle="--", label="Best epoch")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Training Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output_path = _ensure_path(save_path) / f"{_safe_title(title, 'training')}_loss_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_latent_tsne(
    model: nn.Module,
    test_loader: DataLoader,
    device: str | torch.device,
    save_path: str | Path = "results/plots/",
) -> Path:
    """Encode the test set, reduce latents with t-SNE, and save a scatter plot."""
    device_obj = _resolve_device(device)
    model.to(device_obj)
    model.eval()

    latents: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_labels in test_loader:
            images = images.to(device_obj)
            if isinstance(model, VAE):
                mu, _ = model.encode(images)
                z = mu
            elif hasattr(model, "encode"):
                z = model.encode(images)
            else:
                raise TypeError("model must provide an encode method for latent visualization.")
            latents.append(z.cpu().numpy())
            labels.append(batch_labels.numpy())

    latent_matrix = np.concatenate(latents, axis=0)
    label_vector = np.concatenate(labels, axis=0)
    embedded = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(
        latent_matrix
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=label_vector,
        cmap="tab10",
        s=8,
        alpha=0.75,
    )
    ax.set_title("t-SNE of Autoencoder Latent Space")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    colorbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    colorbar.set_label("Digit")
    fig.tight_layout()

    output_path = _ensure_path(save_path) / "tsne_latent.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
