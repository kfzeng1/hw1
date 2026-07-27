import argparse
import os

import torch

from .config import parse_args as parse_train_args
from .data import build_test_loader
from .engine import evaluate
from .inference import get_device, load_model_from_checkpoint
from .metrics import classification_metrics, save_metrics
from .visualize import save_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints_100_finetune/cifar10_wrn_best.pt",
        type=str,
    )
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output-dir", default="./results/cifar10_100_finetune", type=str)
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
        all_targets.append(targets.cpu())
    return torch.cat(all_preds), torch.cat(all_targets)


def build_eval_args(cli_args, checkpoint, device):
    train_args_dict = vars(parse_train_args([]))
    train_args_dict.update(checkpoint.get("args", {}))
    train_args = argparse.Namespace(**train_args_dict)
    train_args.data_dir = cli_args.data_dir
    train_args.test_batch_size = cli_args.batch_size
    train_args.workers = cli_args.workers
    train_args.tta = cli_args.tta
    train_args.amp = device.type == "cuda" and not getattr(train_args, "no_amp", False)
    train_args.limit_train_samples = 0
    train_args.limit_test_samples = 0
    train_args.limit_train_batches = 0
    train_args.limit_test_batches = 0
    return train_args


def main():
    args = parse_args()
    device = get_device()
    model, checkpoint, device = load_model_from_checkpoint(args.checkpoint, device)
    eval_args = build_eval_args(args, checkpoint, device)

    test_loader = build_test_loader(eval_args)
    test_loss, test_acc = evaluate(model, test_loader, device, eval_args)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%")

    preds, targets = collect_predictions(
        model, test_loader, device, eval_args.amp, eval_args.tta
    )
    matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")
    save_confusion_matrix(preds, targets, matrix_path)

    metrics = classification_metrics(preds, targets)
    metrics["test_loss"] = test_loss
    metrics["test_acc_percent"] = test_acc
    metrics_json, report_csv = save_metrics(metrics, args.output_dir)

    print(f"accuracy={metrics['accuracy'] * 100:.2f}%")
    print(f"macro_f1={metrics['macro_avg']['f1'] * 100:.2f}%")
    print(f"weighted_f1={metrics['weighted_avg']['f1'] * 100:.2f}%")
    print(f"confusion matrix: {matrix_path}")
    print(f"metrics json: {metrics_json}")
    print(f"classification report: {report_csv}")


if __name__ == "__main__":
    main()
