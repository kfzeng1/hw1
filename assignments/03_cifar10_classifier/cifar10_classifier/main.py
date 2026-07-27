import csv
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


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return device


def build_model(args, device):
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
    return model


def build_optimizer(model, args):
    return optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def build_scheduler(optimizer, args):
    return CosineWarmupLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )


def checkpoint_paths(output_dir, history_file):
    return {
        "last": os.path.join(output_dir, "cifar10_wrn_last.pt"),
        "best": os.path.join(output_dir, "cifar10_wrn_best.pt"),
        "history": os.path.join(output_dir, history_file),
        "curves": os.path.join(output_dir, "training_curves.png"),
    }


def prepare_history_file(paths, args, start_epoch, fieldnames):
    if not args.resume or start_epoch <= 0:
        return

    history_path = paths["history"]
    resume_history = os.path.join(os.path.dirname(args.resume), args.history_file)
    source_path = history_path if os.path.exists(history_path) else resume_history

    if not os.path.exists(source_path):
        return

    rows = []
    with open(source_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(row["epoch"]) <= start_epoch:
                rows.append(row)

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_run_header(args, device, model):
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"model: WideResNet-{args.depth}-{args.widen_factor}")
    print(f"trainable parameters: {count_parameters(model):,}")
    print(f"epochs: {args.epochs}")
    print(f"output_dir: {args.output_dir}")


def train(args):
    set_seed(args.seed)
    device = get_device()
    args.amp = args.amp and device.type == "cuda"

    train_loader, test_loader = build_loaders(args)
    model = build_model(args, device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    paths = checkpoint_paths(args.output_dir, args.history_file)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, scaler, device)
        print(
            f"resumed from {args.resume}, start_epoch={start_epoch}, "
            f"best_acc={best_acc:.2f}"
        )

    print_run_header(args, device, model)
    history_fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "best_acc",
        "seconds",
    ]
    prepare_history_file(paths, args, start_epoch, history_fields)
    logger = CSVLogger(
        paths["history"],
        history_fields,
        append=bool(args.resume),
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            best_acc = run_epoch(
                epoch,
                best_acc,
                model,
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                args,
                paths,
                logger,
            )
            if args.stop_at_target and best_acc >= args.target_acc:
                print(f"stopping because target accuracy {args.target_acc:.2f}% was reached")
                break
    finally:
        logger.close()

    print_summary(best_acc, args, paths)


def run_epoch(
    epoch,
    best_acc,
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
    paths,
    logger,
):
    epoch_start = time.time()
    lr = scheduler.step(epoch)
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, scaler, device, args
    )
    test_loss, test_acc = evaluate(model, test_loader, device, args)
    elapsed = time.time() - epoch_start

    if test_acc > best_acc:
        best_acc = test_acc
        save_checkpoint(paths["best"], model, optimizer, scaler, epoch, best_acc, args)
    save_checkpoint(paths["last"], model, optimizer, scaler, epoch, best_acc, args)

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
    plot_history(paths["history"], paths["curves"])

    print(
        f"epoch {epoch + 1:03d}/{args.epochs} "
        f"lr={lr:.6f} "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% "
        f"best={best_acc:.2f}% "
        f"time={elapsed:.1f}s"
    )
    return best_acc


def print_summary(best_acc, args, paths):
    print(f"best accuracy: {best_acc:.2f}%")
    print(f"best checkpoint: {paths['best']}")
    print(f"last checkpoint: {paths['last']}")
    print(f"history: {paths['history']}")
    print(f"training curves: {paths['curves']}")
    if best_acc >= args.target_acc:
        print(f"target reached: {best_acc:.2f}% >= {args.target_acc:.2f}%")
    else:
        print(f"target not reached yet: {best_acc:.2f}% < {args.target_acc:.2f}%")


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
