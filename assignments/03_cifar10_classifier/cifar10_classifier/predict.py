import argparse

import torch
from PIL import Image

from .data import CIFAR10_CLASSES, build_predict_transform
from .inference import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one CIFAR-10 image.")
    parser.add_argument("image", type=str)
    parser.add_argument("--checkpoint", default="./checkpoints_50/cifar10_wrn_best.pt", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    model, _, device = load_model_from_checkpoint(args.checkpoint)

    transform = build_predict_transform()
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_prob, top_idx = probs.max(dim=0)

    print(f"class: {CIFAR10_CLASSES[top_idx.item()]}")
    print(f"probability: {top_prob.item():.4f}")


if __name__ == "__main__":
    main()
