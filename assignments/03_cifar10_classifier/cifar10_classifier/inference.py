import torch

from .model import WideResNet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def load_model_from_checkpoint(checkpoint_path, device=None):
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = WideResNet(
        depth=ckpt_args.get("depth", 28),
        widen_factor=ckpt_args.get("widen_factor", 10),
        drop_rate=ckpt_args.get("drop_rate", 0.3),
        num_classes=ckpt_args.get("num_classes", 10),
    ).to(device)

    state_dict = checkpoint["model"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint, device
