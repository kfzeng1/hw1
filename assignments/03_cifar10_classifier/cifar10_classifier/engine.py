import os
import random

import torch

from .augment import mix_batch, one_hot, soft_cross_entropy
from .inference import unwrap_model
from .utils import AverageMeter, accuracy


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if getattr(args, "channels_last", False):
            images = images.contiguous(memory_format=torch.channels_last)
        images, soft_targets = mix_batch(
            images,
            targets,
            args.num_classes,
            args.mixup_alpha,
            args.cutmix_alpha,
            args.label_smoothing,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp):
            logits = model(images)
            loss = soft_cross_entropy(logits, soft_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(accuracy(logits.detach(), targets), images.size(0))

        if args.limit_train_batches > 0 and batch_idx + 1 >= args.limit_train_batches:
            break

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if getattr(args, "channels_last", False):
            images = images.contiguous(memory_format=torch.channels_last)
        hard_targets = one_hot(targets, args.num_classes, smoothing=0.0)

        with torch.autocast(device_type=device.type, enabled=args.amp):
            logits = model(images)
            if args.tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
            loss = soft_cross_entropy(logits, hard_targets)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(accuracy(logits, targets), images.size(0))

        if args.limit_test_batches > 0 and batch_idx + 1 >= args.limit_test_batches:
            break

    return loss_meter.avg, acc_meter.avg


def save_checkpoint(path, model, optimizer, scaler, epoch, best_acc, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(path, model, optimizer, scaler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )
    return checkpoint["epoch"] + 1, checkpoint.get("best_acc", 0.0)
