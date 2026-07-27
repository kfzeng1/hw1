import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch


@dataclass
class FitResult:
    degree: int
    scaled_weights: torch.Tensor
    feature_scales: torch.Tensor
    losses: list[float]

    @property
    def final_loss(self):
        return self.losses[-1]

    @property
    def raw_coefficients(self):
        return (self.scaled_weights / self.feature_scales.t()).flatten()


def polynomial_features(x, degree):
    return torch.cat([x**power for power in range(degree + 1)], dim=1)


def scale_features(features):
    scales = features.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
    return features / scales, scales


def fit_polynomial(x_train, y_train, degree, epochs, lr, log_interval):
    features, scales = scale_features(polynomial_features(x_train, degree))
    weights = torch.zeros(degree + 1, 1, dtype=torch.float64, requires_grad=True)
    losses = []

    for epoch in range(1, epochs + 1):
        prediction = features @ weights
        loss = torch.mean((prediction - y_train) ** 2)
        loss.backward()

        with torch.no_grad():
            weights -= lr * weights.grad
            weights.grad.zero_()

        losses.append(float(loss.item()))
        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(f"degree={degree}, epoch={epoch:5d}, mse={loss.item():.8f}")

    return FitResult(
        degree=degree,
        scaled_weights=weights.detach(),
        feature_scales=scales.detach(),
        losses=losses,
    )


def predict(x, result):
    features = polynomial_features(x, result.degree)
    scaled_features = features / result.feature_scales
    return scaled_features @ result.scaled_weights


def polynomial_expression(result):
    parts = []
    for power, coeff in enumerate(result.raw_coefficients.tolist()):
        if power == 0:
            parts.append(f"{coeff:+.8f}")
        elif power == 1:
            parts.append(f"{coeff:+.8f}x")
        else:
            parts.append(f"{coeff:+.8f}x^{power}")
    return " ".join(parts)


def save_loss_curves(results, output_path):
    plt.figure(figsize=(10, 5))
    for result in results:
        epochs = range(1, len(result.losses) + 1)
        plt.plot(epochs, result.losses, label=f"degree {result.degree}")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Loss convergence with gradient descent")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_fit_comparison(x_train, y_train, x_plot, y_true, results, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot.numpy(), y_true.numpy(), "k-", linewidth=2, label="sin(x)")
    plt.scatter(x_train.numpy(), y_train.numpy(), s=10, alpha=0.18, label="training points")

    for result in results:
        y_plot = predict(x_plot, result).detach()
        plt.plot(
            x_plot.numpy(),
            y_plot.numpy(),
            linewidth=1.8,
            label=f"degree {result.degree}, MSE={result.final_loss:.6f}",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial approximation of sin(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_coefficients(results, output_path):
    lines = [
        "Polynomial approximation target: y = sin(x)",
        "Training method: full-batch gradient descent",
        "",
    ]
    for result in results:
        lines.append(f"degree {result.degree}: y_pred = {polynomial_expression(result)}")
        lines.append(f"final MSE = {result.final_loss:.10f}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_history(results, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["degree", "epoch", "mse"])
        for result in results:
            for epoch, loss in enumerate(result.losses, start=1):
                writer.writerow([result.degree, epoch, f"{loss:.12f}"])


def parse_degrees(text):
    degrees = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not degrees:
        raise argparse.ArgumentTypeError("degrees cannot be empty")
    return degrees


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit sin(x) with degree 3, 4 and 5 polynomial regression."
    )
    parser.add_argument("--degrees", default="3,4,5", type=parse_degrees)
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--lr", default=0.4, type=float)
    parser.add_argument("--train-points", default=160, type=int)
    parser.add_argument("--plot-points", default=500, type=int)
    parser.add_argument("--log-interval", default=1000, type=int)
    parser.add_argument("--output-dir", default="./results", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_train = torch.linspace(-math.pi, math.pi, args.train_points, dtype=torch.float64).reshape(-1, 1)
    y_train = torch.sin(x_train)
    x_plot = torch.linspace(-math.pi, math.pi, args.plot_points, dtype=torch.float64).reshape(-1, 1)
    y_true = torch.sin(x_plot)

    results = [
        fit_polynomial(
            x_train=x_train,
            y_train=y_train,
            degree=degree,
            epochs=args.epochs,
            lr=args.lr,
            log_interval=args.log_interval,
        )
        for degree in args.degrees
    ]

    loss_path = args.output_dir / "loss_convergence.png"
    fit_path = args.output_dir / "fit_comparison.png"
    coefficient_path = args.output_dir / "coefficients.txt"
    history_path = args.output_dir / "training_history.csv"

    save_loss_curves(results, loss_path)
    save_fit_comparison(x_train, y_train, x_plot, y_true, results, fit_path)
    write_coefficients(results, coefficient_path)
    write_history(results, history_path)

    print()
    print(coefficient_path.read_text(encoding="utf-8").strip())
    print(f"saved: {loss_path}")
    print(f"saved: {fit_path}")
    print(f"saved: {history_path}")


if __name__ == "__main__":
    main()
