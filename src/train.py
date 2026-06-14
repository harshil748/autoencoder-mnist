"""Training utilities and MNIST dataloaders for autoencoder experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm.auto import tqdm

try:
    from .models.vae import vae_loss
except ImportError:  # Allows `python src/train.py` during local experiments.
    from models.vae import vae_loss


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 5,
    is_vae: bool = False,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Train an autoencoder with Adam, validation tracking, and early stopping."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")

    device_obj = _resolve_device(device)
    model.to(device_obj)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = vae_loss if is_vae else nn.MSELoss()

    history: dict[str, Any] = {
        "train_losses": [],
        "val_losses": [],
        "best_epoch": 0,
        "stopped_early": False,
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for images, _ in train_loader:
            images = images.to(device_obj)
            optimizer.zero_grad(set_to_none=True)

            if is_vae:
                recon, mu, log_var = model(images)
                loss = criterion(recon, images, mu, log_var) / images.size(0)
            else:
                recon = model(images)
                loss = criterion(recon, images.view(-1, 784))

            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device_obj)
                if is_vae:
                    recon, mu, log_var = model(images)
                    loss = criterion(recon, images, mu, log_var) / images.size(0)
                else:
                    recon = model(images)
                    loss = criterion(recon, images.view(-1, 784))
                val_loss_sum += float(loss.item())
                val_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                history["stopped_early"] = True
                break

    return history


def get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST and return train, validation, and test dataloaders."""
    transform = transforms.ToTensor()
    data_root = Path("./data")

    full_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_set, val_set = random_split(
        full_train,
        [50000, 10000],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

