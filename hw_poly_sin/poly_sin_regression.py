import math

import matplotlib.pyplot as plt
import torch


def make_polynomial_features(x, degree):
    """Return [1, x, x^2, ..., x^degree] for each x."""
    return torch.cat([x**i for i in range(degree + 1)], dim=1)


def train_polynomial_regression(x_train, y_train, degree, epochs=5000, lr=0.03):
    features = make_polynomial_features(x_train, degree)
    model = torch.nn.Linear(degree + 1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        y_pred = model(features)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"degree={degree}, epoch={epoch:4d}, loss={loss.item():.8f}")

    return model, loss.item()


def polynomial_expression(model):
    weights = model.weight.detach().flatten()
    coeffs = [w.item() for w in weights]

    parts = [f"{coeffs[0]:+.6f}"]
    for power, coeff in enumerate(coeffs[1:], start=1):
        if power == 1:
            parts.append(f"{coeff:+.6f}x")
        else:
            parts.append(f"{coeff:+.6f}x^{power}")
    return " ".join(parts)


def main():
    torch.manual_seed(0)

    x_train = torch.linspace(-math.pi, math.pi, 160).reshape(-1, 1)
    y_train = torch.sin(x_train)

    x_plot = torch.linspace(-math.pi, math.pi, 400).reshape(-1, 1)
    y_true = torch.sin(x_plot)

    degrees = [3, 4, 5]
    results = {}

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot.numpy(), y_true.numpy(), "k-", label="sin(x)")

    for degree in degrees:
        print(f"\nTraining polynomial degree {degree}")
        model, final_loss = train_polynomial_regression(x_train, y_train, degree)
        y_plot = model(make_polynomial_features(x_plot, degree)).detach()
        results[degree] = (model, final_loss)

        plt.plot(
            x_plot.numpy(),
            y_plot.numpy(),
            label=f"degree {degree}, MSE={final_loss:.6f}",
        )

    plt.scatter(x_train.numpy(), y_train.numpy(), s=10, alpha=0.25, label="train data")
    plt.title("Polynomial Linear Regression Approximation of sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("poly_sin_regression.png", dpi=160)

    print("\nLearned polynomial expressions:")
    for degree, (model, final_loss) in results.items():
        print(f"degree {degree}: y = {polynomial_expression(model)}")
        print(f"          final MSE = {final_loss:.8f}")

    print("\nSaved plot to poly_sin_regression.png")


if __name__ == "__main__":
    main()
