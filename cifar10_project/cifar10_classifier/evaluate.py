import argparse
import os

import torch

from .config import parse_args as parse_train_args
from .data import build_loaders
from .engine import evaluate
from .inference import get_device, load_model_from_checkpoint
from .visualize import save_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint.")
    parser.add_argument("--checkpoint", default="./checkpoints_50/cifar10_wrn_best.pt", type=str)
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output-dir", default="./checkpoints_50", type=str)
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device, amp, tta):
    all_preds = []
    all_targets = []
    model.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            if tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
        all_preds.append(logits.argmax(dim=1).cpu())
        all_targets.append(targets)
    return torch.cat(all_preds), torch.cat(all_targets)


def main():
    args = parse_args()
    device = get_device()
    model, checkpoint, device = load_model_from_checkpoint(args.checkpoint, device)
    train_args_dict = vars(parse_train_args([]))
    train_args_dict.update(checkpoint.get("args", {}))
    train_args = argparse.Namespace(**train_args_dict)
    train_args.data_dir = args.data_dir
    train_args.test_batch_size = args.batch_size
    train_args.workers = args.workers
    train_args.tta = args.tta
    train_args.amp = device.type == "cuda" and not getattr(train_args, "no_amp", False)
    train_args.limit_train_samples = 0
    train_args.limit_test_samples = 0
    train_args.limit_train_batches = 0
    train_args.limit_test_batches = 0

    _, test_loader = build_loaders(train_args)
    test_loss, test_acc = evaluate(model, test_loader, device, train_args)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%")

    preds, targets = collect_predictions(
        model, test_loader, device, train_args.amp, train_args.tta
    )
    matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")
    save_confusion_matrix(preds, targets, matrix_path)
    print(f"confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()
