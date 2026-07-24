import math


class CosineWarmupLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
