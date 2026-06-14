# Autoencoder Implementation & Regularization Study on MNIST

> A deep learning study comparing Basic Autoencoder, Denoising Autoencoder, and Variational Autoencoder on MNIST with regularization analysis across latent space sizes.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)

---

## Project Overview

This is a structured deep learning study implementing and comparing autoencoder architectures on the MNIST handwritten digit dataset using PyTorch.

The study covers three key areas:

1. **Model comparison** — Basic Autoencoder vs Denoising Autoencoder vs PCA baseline
2. **Regularization analysis** — L2 weight decay, Dropout, and Early Stopping
3. **Latent space study** — Effect of latent dimensions (16 vs 32 vs 64) on reconstruction quality

All results are quantified using MSE and visualized through reconstruction plots and training curves.

---

## Features

### Core Features

- **Basic Autoencoder** — Fully-connected encoder-decoder (784 → latent → 784) with MSE loss and Adam optimizer
- **Denoising Autoencoder** — Trains on Gaussian or salt-and-pepper corrupted inputs to reconstruct clean digits
- **Regularization Techniques** — L2 (weight_decay), Dropout layers in encoder/decoder, Early Stopping on val loss plateau
- **Latent Space Analysis** — Systematic comparison across latent dims 16, 32, and 64
- **Visualizations** — Side-by-side original vs reconstructed images, training/validation loss curves
- **MSE Comparison Table** — Full results table across all model × latent size combinations

### Bonus Features

- **Variational Autoencoder (VAE)** — Reparameterization trick with ELBO loss; sample new digit images from latent space
- **Anomaly Detection Demo** — Train on digits 0–8, detect digit 9 using high reconstruction error
- **Latent Space Visualization** — t-SNE / UMAP plot of latent vectors colored by digit class
- **Convolutional Autoencoder** — Conv2d / ConvTranspose2d architecture vs FC comparison

---

## Tech Stack

| Area                    | Technology                      |
| ----------------------- | ------------------------------- |
| Deep Learning Framework | PyTorch 2.0+                    |
| Baseline                | scikit-learn PCA                |
| Data Loading            | torchvision.datasets.MNIST      |
| Visualization           | matplotlib, seaborn             |
| Notebook                | Jupyter Notebook / Google Colab |
| Version Control         | Git / GitHub                    |

---

## Prerequisites

- Python 3.10 or higher
- pip
- Git
- (Optional) CUDA-enabled GPU for faster training

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/autoencoder-mnist.git
cd autoencoder-mnist
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** MNIST downloads automatically via torchvision on first run. No manual dataset setup needed.

---

## Usage

### Running the Notebook (recommended)

```bash
jupyter notebook notebooks/autoencoder_study.ipynb
```

Open in your browser and run cells top-to-bottom. Each section has markdown explaining what it does.

### Running Individual Scripts

```bash
# Train the basic autoencoder
python src/train.py --model basic --latent_dim 32 --epochs 50

# Evaluate and generate the MSE comparison table
python src/evaluate.py

# Generate all plots
python src/visualize.py
```

### On Google Colab

Upload the notebook to [Google Colab](https://colab.research.google.com/) and enable GPU runtime (`Runtime → Change runtime type → T4 GPU`). All dependencies install in the first cell.

---

## Project Structure

```
autoencoder-mnist/
├── notebooks/
│   └── autoencoder_study.ipynb   # Main Jupyter notebook — full study
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── basic_ae.py           # Basic Autoencoder (+ Regularized variant)
│   │   ├── denoising_ae.py       # Denoising Autoencoder with noise injection
│   │   └── vae.py                # Variational Autoencoder (bonus)
│   ├── __init__.py
│   ├── train.py                  # Training loop with early stopping
│   ├── evaluate.py               # MSE evaluation across all model combinations
│   └── visualize.py              # Reconstruction plots and loss curves
├── results/
│   ├── plots/                    # All saved matplotlib figures
│   └── metrics_table.md          # Auto-generated MSE comparison table
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Results

### MSE Comparison Table

| Model              | Latent 16 | Latent 32 | Latent 64 |
| ------------------ | --------- | --------- | --------- |
| Basic AE (no reg)  | 0.007962  | 0.005049  | 0.004273  |
| Basic AE + L2      | 0.018866  | 0.019756  | 0.018760  |
| Basic AE + Dropout | 0.008916  | 0.006107  | 0.005751  |
| Denoising AE       | 0.009917  | 0.007599  | 0.006959  |
| Denoising AE + L2  | 0.020824  | 0.021403  | 0.021165  |
| PCA Baseline       | 0.016833  | 0.016833  | 0.016833  |
| VAE (bonus)        | —         | —         | —         |

> Results will be populated after training. See `results/metrics_table.md` for the full auto-generated table.

### Sample Reconstructions

| Original                    | Reconstructed (Basic AE) | Reconstructed (Denoising AE) |
| --------------------------- | ------------------------ | ---------------------------- |
| _(saved to results/plots/)_ |                          |                              |

### Key Analysis Questions Answered

1. **Which model achieves better reconstruction?** — Addressed in Section 8 of the notebook with quantitative MSE scores and visual comparison.
2. **How does each regularization technique affect training?** — L2 reduces overfitting; Dropout improves generalization; Early Stopping prevents overtraining.
3. **Effect of latent space size?** — Larger latent dim → lower MSE but less compression; smaller dim → higher compression, coarser reconstruction.

---

## 🧠 Model Architecture

### Basic Autoencoder

```
Encoder: 784 → 512 → ReLU → 256 → ReLU → latent_dim
Decoder: latent_dim → 256 → ReLU → 512 → ReLU → 784 → Sigmoid
Loss: MSELoss | Optimizer: Adam
```

### Denoising Autoencoder

```
Input: x_clean → add_noise(x) → x_noisy
Same encoder-decoder as BasicAE
Target: reconstruct x_clean from x_noisy
```

### Variational Autoencoder (Bonus)

```
Encoder: 784 → 512 → (mu, log_var) branches → latent sample via reparameterization
Decoder: same as BasicAE
Loss: ELBO = BCE(reconstruction) + KL(N(mu,sigma) || N(0,1))
```

---

## 📦 requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
tqdm>=4.65.0
umap-learn>=0.5.3
```

---

## 🙏 Acknowledgements

- [PyTorch Documentation](https://pytorch.org/docs/) — deep learning framework
- [torchvision MNIST](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST) — dataset
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — baseline

---