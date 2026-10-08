# Diffusion Model on CIFAR-10

A lightweight Denoising Diffusion Probabilistic Model (DDPM) implemented with PyTorch.

The project is organized around the actual machine-learning workflow:

- `model.py`: model architecture and DDPM diffusion utilities only.
- `train.py`: downloads the CIFAR-10 training set and trains the model.
- `test.py`: evaluates the trained model on the official CIFAR-10 test set.
- `diffusion_model_cifar10.pth`: generated checkpoint (ignored by Git).

## Model

The noise-prediction network is a small U-Net with:

- sinusoidal timestep embeddings
- encoder blocks
- bottleneck
- decoder with skip connection
- noise prediction output

The DDPM training objective is mean squared error (MSE) between the sampled Gaussian noise and the model's predicted noise.

## Dataset

CIFAR-10 is downloaded automatically by `train.py` and `test.py`.

- Training set: 50,000 images
- Test set: 10,000 images
- Image size: 32x32
- Channels: RGB
- Classes: 10

The dataset is not committed to this repository.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --epochs 10 --batch-size 64 --learning-rate 0.001 --timesteps 200
```

The checkpoint is saved as:

```text
diffusion_model_cifar10.pth
```

## Testing

After training:

```bash
python test.py
```

The test script evaluates noise-prediction MSE on the CIFAR-10 test split.

This is a generative diffusion model, not a classifier, so classification accuracy is not the primary metric. Lower noise-prediction MSE indicates better agreement between the predicted and sampled diffusion noise.

## Kaggle

The repository can be cloned into a Kaggle Notebook and trained with a GPU:

```python
!git clone https://github.com/khanhhuyen-ops/diffusion-model-cifar10.git
%cd diffusion-model-cifar10
!pip install -r requirements.txt
!python train.py --epochs 10 --batch-size 128 --timesteps 200
!python test.py
```

Enable a GPU accelerator in the Kaggle Notebook settings before training.

## Project structure

```text
diffusion-model-cifar10/
├── model.py
├── train.py
├── test.py
├── README.md
├── requirements.txt
├── .gitignore
└── data/                       # downloaded locally, not committed
```
