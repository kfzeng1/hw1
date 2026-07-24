import os
import time

import torch
import torch.optim as optim

from .config import parse_args
from .data import build_loaders
from .engine import evaluate, load_checkpoint, save_checkpoint, train_one_epoch
from .model import WideResNet
from .scheduler import CosineWarmupLR
from .utils import CSVLogger, count_parameters, set_seed
from .visualize import plot_history


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = build_loaders(args)
    model = WideResNet(
        depth=args.depth,
        widen_factor=args.widen_factor,
        drop_rate=args.drop_rate,
        num_classes=args.num_classes,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile_model:
        model = torch.compile(model)
    print(f"model: WideResNet-{args.depth}-{args.widen_factor}")
    print(f"trainable parameters: {count_parameters(model):,}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineWarmupLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, scaler, device)
        print(
            f"resumed from {args.resume}, start_epoch={start_epoch}, "
            f"best_acc={best_acc:.2f}"
        )

    last_path = os.path.join(args.output_dir, "cifar10_wrn_last.pt")
    best_path = os.path.join(args.output_dir, "cifar10_wrn_best.pt")
    history_path = os.path.join(args.output_dir, args.history_file)
    curve_path = os.path.join(args.output_dir, "training_curves.png")
    logger = CSVLogger(
        history_path,
        [
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "best_acc",
            "seconds",
        ],
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            lr = scheduler.step(epoch)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, device, args
            )
            test_loss, test_acc = evaluate(model, test_loader, device, args)
            elapsed = time.time() - epoch_start

            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(best_path, model, optimizer, scaler, epoch, best_acc, args)
            save_checkpoint(last_path, model, optimizer, scaler, epoch, best_acc, args)

            logger.write(
                {
                    "epoch": epoch + 1,
                    "lr": lr,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "best_acc": best_acc,
                    "seconds": elapsed,
                }
            )
            plot_history(history_path, curve_path)

            print(
                f"epoch {epoch + 1:03d}/{args.epochs} "
                f"lr={lr:.6f} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% "
                f"best={best_acc:.2f}% "
                f"time={elapsed:.1f}s"
            )
            if args.stop_at_target and best_acc >= args.target_acc:
                print(f"stopping because target accuracy {args.target_acc:.2f}% was reached")
                break
    finally:
        logger.close()

    print(f"best accuracy: {best_acc:.2f}%")
    print(f"best checkpoint: {best_path}")
    print(f"history: {history_path}")
    print(f"training curves: {curve_path}")
    if best_acc >= args.target_acc:
        print(f"target reached: {best_acc:.2f}% >= {args.target_acc:.2f}%")
    else:
        print(f"target not reached yet: {best_acc:.2f}% < {args.target_acc:.2f}%")


if __name__ == "__main__":
    main()
