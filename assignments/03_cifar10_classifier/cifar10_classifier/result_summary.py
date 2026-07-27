import argparse
import csv
import json
from pathlib import Path


def read_history(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_metrics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def best_epoch(history):
    if not history:
        raise ValueError("history is empty")
    return max(history, key=lambda row: float(row["test_acc"]))


def print_summary(result_dir):
    result_dir = Path(result_dir)
    history = read_history(result_dir / "history.csv")
    metrics = read_metrics(result_dir / "metrics.json")
    best = best_epoch(history)
    last = history[-1]

    print("CIFAR-10 result summary")
    print(f"result_dir: {result_dir}")
    print(f"epochs: {len(history)}")
    print(f"best_epoch: {best['epoch']}")
    print(f"best_test_acc: {float(best['test_acc']):.2f}%")
    print(f"final_test_acc: {float(last['test_acc']):.2f}%")
    print(f"accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"macro_precision: {metrics['macro_avg']['precision'] * 100:.2f}%")
    print(f"macro_recall: {metrics['macro_avg']['recall'] * 100:.2f}%")
    print(f"macro_f1: {metrics['macro_avg']['f1'] * 100:.2f}%")
    print(f"weighted_f1: {metrics['weighted_avg']['f1'] * 100:.2f}%")
    print(f"correct: {metrics['correct']} / {metrics['total']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Print CIFAR-10 training and test summary.")
    parser.add_argument("--result-dir", default="./results/cifar10_50_epochs", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    print_summary(args.result_dir)


if __name__ == "__main__":
    main()
