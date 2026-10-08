import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DDPM, MiniUNet


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_PATH = PROJECT_DIR / "diffusion_model_cifar10.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.timesteps < 2:
        raise ValueError("epochs/batch-size must be positive and timesteps >= 2.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Batches per epoch: {len(loader)}")

    model = MiniUNet(in_channels=3, base_dim=32).to(device)
    diffusion = DDPM(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_index, (images, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            loss = diffusion.compute_loss(model, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_index % 100 == 0 or batch_index == len(loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"| Batch {batch_index}/{len(loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / len(loader)
        print(
            f"Epoch {epoch + 1} complete "
            f"| Average Loss: {average_loss:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 3,
            "base_dim": 32,
            "timesteps": args.timesteps,
            "epochs": args.epochs,
        },
        CHECKPOINT_PATH,
    )

    print(f"Model saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
