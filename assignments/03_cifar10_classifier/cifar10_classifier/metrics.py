import csv
import json
import os

import torch

from .data import CIFAR10_CLASSES


def confusion_matrix(preds, targets, num_classes=10):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for target, pred in zip(targets, preds):
        matrix[target.long(), pred.long()] += 1
    return matrix


def classification_metrics(preds, targets, class_names=CIFAR10_CLASSES):
    num_classes = len(class_names)
    matrix = confusion_matrix(preds, targets, num_classes)
    total = int(matrix.sum().item())
    correct = int(matrix.diag().sum().item())
    accuracy = correct / max(1, total)

    per_class = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    for index, name in enumerate(class_names):
        true_positive = matrix[index, index].item()
        false_positive = matrix[:, index].sum().item() - true_positive
        false_negative = matrix[index, :].sum().item() - true_positive
        support = matrix[index, :].sum().item()

        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        per_class.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    macro = {
        "precision": sum(precisions) / num_classes,
        "recall": sum(recalls) / num_classes,
        "f1": sum(f1_scores) / num_classes,
    }
    weighted = {
        "precision": _weighted_average(precisions, supports),
        "recall": _weighted_average(recalls, supports),
        "f1": _weighted_average(f1_scores, supports),
    }
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "per_class": per_class,
    }


def _weighted_average(values, weights):
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total


def save_metrics(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "metrics.json")
    csv_path = os.path.join(output_dir, "classification_report.csv")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["class", "precision", "recall", "f1", "support"]
        )
        writer.writeheader()
        writer.writerows(metrics["per_class"])

    return json_path, csv_path
