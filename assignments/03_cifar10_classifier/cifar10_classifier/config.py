import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    mirror: str = "official"
    download_url: str = ""
    output_dir: str = "./checkpoints_100_finetune"
    history_file: str = "history.csv"
    epochs: int = 100
    batch_size: int = 128
    test_batch_size: int = 512
    workers: int = 4
    lr: float = 0.1
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    momentum: float = 0.9
    weight_decay: float = 5e-4
    depth: int = 28
    widen_factor: int = 10
    drop_rate: float = 0.3
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    seed: int = 42
    num_classes: int = 10
    resume: str = ""
    no_amp: bool = False
    channels_last: bool = False
    compile_model: bool = False
    tta: bool = False
    target_acc: float = 90.0
    stop_at_target: bool = False
    limit_train_samples: int = 0
    limit_test_samples: int = 0
    limit_train_batches: int = 0
    limit_test_batches: int = 0
    amp: bool = True


def build_parser():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier.")
    parser.add_argument("--data-dir", default=TrainConfig.data_dir, type=str)
    parser.add_argument(
        "--mirror",
        default=TrainConfig.mirror,
        choices=("official", "sjtu", "oneflow", "baidu", "brainchip"),
        type=str,
    )
    parser.add_argument("--download-url", default=TrainConfig.download_url, type=str)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir, type=str)
    parser.add_argument("--history-file", default=TrainConfig.history_file, type=str)
    parser.add_argument("--epochs", default=TrainConfig.epochs, type=int)
    parser.add_argument("--batch-size", default=TrainConfig.batch_size, type=int)
    parser.add_argument("--test-batch-size", default=TrainConfig.test_batch_size, type=int)
    parser.add_argument("--workers", default=TrainConfig.workers, type=int)
    parser.add_argument("--lr", default=TrainConfig.lr, type=float)
    parser.add_argument("--min-lr", default=TrainConfig.min_lr, type=float)
    parser.add_argument("--warmup-epochs", default=TrainConfig.warmup_epochs, type=int)
    parser.add_argument("--momentum", default=TrainConfig.momentum, type=float)
    parser.add_argument("--weight-decay", default=TrainConfig.weight_decay, type=float)
    parser.add_argument("--depth", default=TrainConfig.depth, type=int)
    parser.add_argument("--widen-factor", default=TrainConfig.widen_factor, type=int)
    parser.add_argument("--drop-rate", default=TrainConfig.drop_rate, type=float)
    parser.add_argument("--mixup-alpha", default=TrainConfig.mixup_alpha, type=float)
    parser.add_argument("--cutmix-alpha", default=TrainConfig.cutmix_alpha, type=float)
    parser.add_argument("--label-smoothing", default=TrainConfig.label_smoothing, type=float)
    parser.add_argument("--grad-clip", default=TrainConfig.grad_clip, type=float)
    parser.add_argument("--seed", default=TrainConfig.seed, type=int)
    parser.add_argument("--num-classes", default=TrainConfig.num_classes, type=int)
    parser.add_argument("--resume", default=TrainConfig.resume, type=str)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip TTA.")
    parser.add_argument("--target-acc", default=TrainConfig.target_acc, type=float)
    parser.add_argument("--stop-at-target", action="store_true")
    parser.add_argument("--limit-train-samples", default=0, type=int)
    parser.add_argument("--limit-test-samples", default=0, type=int)
    parser.add_argument("--limit-train-batches", default=0, type=int)
    parser.add_argument("--limit-test-batches", default=0, type=int)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    args.amp = not args.no_amp
    return args
