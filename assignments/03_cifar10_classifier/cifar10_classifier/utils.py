import random
import csv
import os
from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    @property
    def avg(self):
        return self.total / max(1, self.count)

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n


def accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100.0


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if self.file.tell() == 0:
            self.writer.writeheader()
            self.file.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
