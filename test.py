from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DDPM, MiniUNet


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_PATH = PROJECT_DIR / "diffusion_model_cifar10.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    print(f"Test images: {len(test_dataset)}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = MiniUNet(
        in_channels=checkpoint["in_channels"],
        base_dim=checkpoint["base_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = DDPM(
        timesteps=checkpoint["timesteps"],
        device=device,
    )

    total_squared_error = 0.0
    total_elements = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            batch_size = images.size(0)

            t = torch.randint(
                0,
                diffusion.timesteps,
                (batch_size,),
                device=device,
            ).long()

            noisy_images, real_noise = diffusion.q_sample(images, t)
            predicted_noise = model(noisy_images, t)

            squared_error = F.mse_loss(
                predicted_noise,
                real_noise,
                reduction="sum",
            )

            total_squared_error += squared_error.item()
            total_elements += real_noise.numel()

    test_mse = total_squared_error / total_elements

    print("\n==============================")
    print("DDPM TEST RESULTS")
    print("==============================")
    print(f"Noise Prediction MSE: {test_mse:.6f}")
    print("==============================")


if __name__ == "__main__":
    main()
