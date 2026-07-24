import math
import random

import torch
import torch.nn.functional as F


def one_hot(targets, num_classes, smoothing):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y = torch.full(
        (targets.size(0), num_classes),
        off_value,
        device=targets.device,
        dtype=torch.float32,
    )
    y.scatter_(1, targets.unsqueeze(1), on_value)
    return y


def rand_bbox(width, height, lam):
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def mix_batch(images, targets, num_classes, mixup_alpha, cutmix_alpha, smoothing):
    target_onehot = one_hot(targets, num_classes, smoothing)

    use_mixup = mixup_alpha > 0
    use_cutmix = cutmix_alpha > 0
    if not use_mixup and not use_cutmix:
        return images, target_onehot

    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    if use_mixup and use_cutmix:
        use_cutmix_now = random.random() < 0.5
    else:
        use_cutmix_now = use_cutmix

    if use_cutmix_now:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        x1, y1, x2, y2 = rand_bbox(images.size(3), images.size(2), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(images.size(2) * images.size(3)))
    else:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        images = images * lam + images[perm] * (1.0 - lam)

    mixed_targets = target_onehot * lam + target_onehot[perm] * (1.0 - lam)
    return images, mixed_targets


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()
