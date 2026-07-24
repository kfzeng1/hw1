import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 classifier for 50 epochs on CIFAR-10."
    )
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument(
        "--mirror",
        default="official",
        choices=("official", "sjtu", "oneflow", "baidu", "brainchip"),
        type=str,
    )
    parser.add_argument("--download-url", default="", type=str)
    parser.add_argument("--output-dir", default="./checkpoints_50", type=str)
    parser.add_argument("--history-file", default="history.csv", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--test-batch-size", default=512, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--depth", default=28, type=int)
    parser.add_argument("--widen-factor", default=10, type=int)
    parser.add_argument("--drop-rate", default=0.3, type=float)
    parser.add_argument("--mixup-alpha", default=0.2, type=float)
    parser.add_argument("--cutmix-alpha", default=1.0, type=float)
    parser.add_argument("--label-smoothing", default=0.1, type=float)
    parser.add_argument("--grad-clip", default=5.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip TTA.")
    parser.add_argument("--target-acc", default=90.0, type=float)
    parser.add_argument("--stop-at-target", action="store_true")

    parser.add_argument("--limit-train-samples", default=0, type=int)
    parser.add_argument("--limit-test-samples", default=0, type=int)
    parser.add_argument("--limit-train-batches", default=0, type=int)
    parser.add_argument("--limit-test-batches", default=0, type=int)

    args = parser.parse_args(argv)
    args.amp = not args.no_amp
    return args
