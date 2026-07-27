import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from .data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD


def denormalize(images):
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def save_sample_grid(data_dir, output_path, count=32):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = datasets.CIFAR10(
        root=data_dir, train=True, transform=transform, download=True
    )
    images = torch.stack([dataset[i][0] for i in range(count)])
    grid = make_grid(denormalize(images), nrow=8, padding=2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_history(history_path, output_path):
    if not os.path.exists(history_path):
        return

    rows = []
    with open(history_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(row)
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    test_loss = [float(row["test_loss"]) for row in rows]
    train_acc = [float(row["train_acc"]) for row in rows]
    test_acc = [float(row["test_acc"]) for row in rows]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, test_loss, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, test_acc, label="test")
    plt.axhline(90.0, color="red", linestyle="--", linewidth=1, label="90%")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_confusion_matrix(preds, targets, output_path):
    matrix = torch.zeros(10, 10, dtype=torch.int64)
    for target, pred in zip(targets, preds):
        matrix[target.long(), pred.long()] += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix.numpy(), cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10), CIFAR10_CLASSES, rotation=45, ha="right")
    plt.yticks(range(10), CIFAR10_CLASSES)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("CIFAR-10 Confusion Matrix")

    for i in range(10):
        for j in range(10):
            value = matrix[i, j].item()
            if value > 0:
                plt.text(j, i, str(value), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize CIFAR-10 samples.")
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument(
        "--output",
        default="./results/cifar10_100_finetune/cifar10_samples.png",
        type=str,
    )
    parser.add_argument("--count", default=32, type=int)
    args = parser.parse_args()

    save_sample_grid(args.data_dir, args.output, args.count)
    print(f"saved sample grid to {args.output}")


if __name__ == "__main__":
    main()
