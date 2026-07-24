import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cifar10_classifier.engine import evaluate, train_one_epoch
from cifar10_classifier.model import WideResNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.randn(64, 3, 32, 32)
    targets = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(images, targets), batch_size=16, shuffle=True)

    args = argparse.Namespace(
        amp=device.type == "cuda",
        num_classes=10,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        label_smoothing=0.1,
        grad_clip=5.0,
        channels_last=False,
        tta=False,
        limit_train_batches=1,
        limit_test_batches=1,
    )

    model = WideResNet(depth=16, widen_factor=1, drop_rate=0.0, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    train_loss, train_acc = train_one_epoch(model, loader, optimizer, scaler, device, args)
    test_loss, test_acc = evaluate(model, loader, device, args)

    print(f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}%")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%")
    print("smoke test passed")


if __name__ == "__main__":
    main()
