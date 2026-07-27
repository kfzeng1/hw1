import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch


@dataclass
class FitResult:
    degree: int
    model: torch.nn.Linear
    final_loss: float
    losses: list[float]


def make_polynomial_features(x, degree):
    """Return [1, x, x^2, ..., x^degree] for each input value."""
    return torch.cat([x**i for i in range(degree + 1)], dim=1)


def train_polynomial_regression(x_train, y_train, degree, epochs, lr):
    features = make_polynomial_features(x_train, degree)
    model = torch.nn.Linear(degree + 1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    losses = []

    for epoch in range(1, epochs + 1):
        y_pred = model(features)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch == 1 or epoch % 1000 == 0 or epoch == epochs:
            print(f"degree={degree}, epoch={epoch:4d}, loss={loss.item():.8f}")

    return FitResult(degree, model, losses[-1], losses)


def polynomial_expression(model):
    coeffs = [weight.item() for weight in model.weight.detach().flatten()]
    parts = [f"{coeffs[0]:+.6f}"]
    for power, coeff in enumerate(coeffs[1:], start=1):
        suffix = "x" if power == 1 else f"x^{power}"
        parts.append(f"{coeff:+.6f}{suffix}")
    return " ".join(parts)


def save_fit_comparison(x_train, y_train, x_plot, y_true, results, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot.numpy(), y_true.numpy(), "k-", label="sin(x)")

    for result in results:
        y_plot = result.model(
            make_polynomial_features(x_plot, result.degree)
        ).detach()
        plt.plot(
            x_plot.numpy(),
            y_plot.numpy(),
            label=f"degree {result.degree}, MSE={result.final_loss:.6f}",
        )

    plt.scatter(x_train.numpy(), y_train.numpy(), s=10, alpha=0.25, label="train data")
    plt.title("Polynomial Linear Regression Approximation of sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_loss_curves(results, output_path):
    plt.figure(figsize=(10, 5))
    for result in results:
        epochs = range(1, len(result.losses) + 1)
        plt.plot(epochs, result.losses, label=f"degree {result.degree}")
    plt.yscale("log")
    plt.title("MSE Loss Convergence")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_coefficients(results, output_path):
    lines = []
    for result in results:
        lines.append(f"degree {result.degree}: y = {polynomial_expression(result.model)}")
        lines.append(f"final MSE = {result.final_loss:.8f}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use polynomial linear regression to approximate sin(x)."
    )
    parser.add_argument("--degrees", default="3,4,5", type=str)
    parser.add_argument("--epochs", default=5000, type=int)
    parser.add_argument("--lr", default=0.03, type=float)
    parser.add_argument("--train-points", default=160, type=int)
    parser.add_argument("--plot-points", default=400, type=int)
    parser.add_argument("--output-dir", default="./results", type=str)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    degrees = [int(item.strip()) for item in args.degrees.split(",") if item.strip()]
    x_train = torch.linspace(-math.pi, math.pi, args.train_points).reshape(-1, 1)
    y_train = torch.sin(x_train)
    x_plot = torch.linspace(-math.pi, math.pi, args.plot_points).reshape(-1, 1)
    y_true = torch.sin(x_plot)

    results = []
    for degree in degrees:
        print(f"\nTraining polynomial degree {degree}")
        results.append(
            train_polynomial_regression(
                x_train=x_train,
                y_train=y_train,
                degree=degree,
                epochs=args.epochs,
                lr=args.lr,
            )
        )

    fit_path = output_dir / "fit_comparison.png"
    loss_path = output_dir / "loss_convergence.png"
    coeff_path = output_dir / "coefficients.txt"
    save_fit_comparison(x_train, y_train, x_plot, y_true, results, fit_path)
    save_loss_curves(results, loss_path)
    write_coefficients(results, coeff_path)

    print("\nLearned polynomial expressions:")
    print(coeff_path.read_text(encoding="utf-8").strip())
    print(f"\nSaved fit comparison to {fit_path}")
    print(f"Saved loss curves to {loss_path}")


if __name__ == "__main__":
    main()
