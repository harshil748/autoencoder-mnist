"""Evaluation routines for the MNIST autoencoder comparison study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .models import BasicAutoencoder, DenoisingAutoencoder, RegularizedAutoencoder
    from .train import get_dataloaders, train_model
except ImportError:  # Allows `python src/evaluate.py` from the project root.
    from models import BasicAutoencoder, DenoisingAutoencoder, RegularizedAutoencoder
    from train import get_dataloaders, train_model


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_reconstruction(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def compute_mse(model: nn.Module, test_loader: DataLoader, device: str | torch.device) -> float:
    """Compute average pixel-wise mean squared reconstruction error."""
    device_obj = _resolve_device(device)
    model.to(device_obj)
    model.eval()

    total_squared_error = 0.0
    total_pixels = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device_obj)
            recon = _as_reconstruction(model(images))
            target = images.view(-1, 784)
            total_squared_error += F.mse_loss(recon, target, reduction="sum").item()
            total_pixels += target.numel()

    return float(total_squared_error / max(total_pixels, 1))


def _loader_to_numpy(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_images, batch_labels in loader:
        images.append(batch_images.view(batch_images.size(0), -1).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def _pca_baseline(train_loader: DataLoader, test_loader: DataLoader, n_components: int = 32) -> float:
    train_x, _ = _loader_to_numpy(train_loader)
    test_x, _ = _loader_to_numpy(test_loader)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_x)
    reconstructed = pca.inverse_transform(pca.transform(test_x))
    return float(np.mean((test_x - reconstructed) ** 2))


def run_full_comparison(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str | torch.device,
) -> pd.DataFrame:
    """Train all requested model/regularization combinations and save MSE results."""
    device_obj = _resolve_device(device)
    results: list[dict[str, object]] = []
    latent_dims = [16, 32, 64]

    for latent_dim in tqdm(latent_dims, desc="Latent dimensions"):
        experiments = [
            ("BasicAutoencoder", BasicAutoencoder(latent_dim), "None", 0.0),
            ("BasicAutoencoder", BasicAutoencoder(latent_dim), "L2 weight_decay=1e-4", 1e-4),
            (
                "RegularizedAutoencoder",
                RegularizedAutoencoder(latent_dim=latent_dim, dropout_p=0.2),
                "Dropout p=0.2",
                0.0,
            ),
            (
                "DenoisingAutoencoder",
                DenoisingAutoencoder(latent_dim=latent_dim, noise_type="gaussian"),
                "Gaussian noise",
                0.0,
            ),
            (
                "DenoisingAutoencoder",
                DenoisingAutoencoder(latent_dim=latent_dim, noise_type="gaussian"),
                "Gaussian noise + L2 weight_decay=1e-4",
                1e-4,
            ),
            (
                "DenoisingAutoencoder",
                DenoisingAutoencoder(latent_dim=latent_dim, noise_type="salt_pepper"),
                "Salt-pepper noise",
                0.0,
            ),
            (
                "DenoisingAutoencoder",
                DenoisingAutoencoder(latent_dim=latent_dim, noise_type="salt_pepper"),
                "Salt-pepper noise + L2 weight_decay=1e-4",
                1e-4,
            ),
        ]

        for model_name, model, regularization, weight_decay in experiments:
            train_model(
                model,
                train_loader,
                val_loader,
                epochs=50,
                lr=1e-3,
                weight_decay=weight_decay,
                patience=5,
                device=device_obj,
            )
            results.append(
                {
                    "Model": model_name,
                    "Latent_Dim": latent_dim,
                    "Regularization": regularization,
                    "Test_MSE": compute_mse(model, test_loader, device_obj),
                }
            )

    results.append(
        {
            "Model": "PCA",
            "Latent_Dim": 32,
            "Regularization": "Linear PCA baseline",
            "Test_MSE": _pca_baseline(train_loader, test_loader, n_components=32),
        }
    )

    df = pd.DataFrame(results, columns=["Model", "Latent_Dim", "Regularization", "Test_MSE"])
    output_path = Path("results/metrics_table.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    return df


def main() -> None:
    """Run the full comparison experiment and print the metrics table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()
    df = run_full_comparison(train_loader, val_loader, test_loader, device)
    markdown = df.to_markdown(index=False)
    print(markdown)
    Path("results/metrics_table.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()

