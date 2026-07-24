import matplotlib.pyplot as plt
import torch


def f(x, y):
    return x * x + y * y


def gradient(x, y):
    return [2 * x, 2 * y]


def hessian():
    return [[2, 0], [0, 2]]


def newton_minimize(start_x, start_y, eps=1e-10, max_iter=20):
    x = float(start_x)
    y = float(start_y)
    history = []

    for step in range(1, max_iter + 1):
        grad = gradient(x, y)
        hess = hessian()

        # Newton step: [x, y] = [x, y] - H^(-1) * grad
        dx = grad[0] / hess[0][0]
        dy = grad[1] / hess[1][1]
        next_x = x - dx
        next_y = y - dy

        history.append((step, x, y, f(x, y), grad[0], grad[1], next_x, next_y))

        if abs(next_x - x) < eps and abs(next_y - y) < eps:
            break

        x = next_x
        y = next_y

    return x, y, f(x, y), history


def main():
    start_x = 3
    start_y = 4

    min_x, min_y, min_value, history = newton_minimize(start_x, start_y)

    print("Newton method for minimizing f(x, y) = x^2 + y^2")
    print(f"start point: ({start_x}, {start_y})")
    print()
    print("step        x        y      f(x,y)      grad_x      grad_y     next_x     next_y")

    for step, x, y, value, grad_x, grad_y, next_x, next_y in history:
        print(
            f"{step:>4d}  {x:>7.3f}  {y:>7.3f}  {value:>10.6f}  "
            f"{grad_x:>10.6f}  {grad_y:>10.6f}  {next_x:>9.6f}  {next_y:>9.6f}"
        )

    print()
    print(f"minimum point: ({min_x:.10f}, {min_y:.10f})")
    print(f"minimum value: {min_value:.10f}")

    plot_newton_path(history)


def plot_newton_path(history):
    x = torch.linspace(-5, 5, 200)
    y = torch.linspace(-5, 5, 200)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    grid_z = grid_x**2 + grid_y**2

    path_x = [item[1] for item in history]
    path_y = [item[2] for item in history]
    path_x.append(history[-1][6])
    path_y.append(history[-1][7])

    plt.figure(figsize=(7, 6))
    contour = plt.contour(grid_x.numpy(), grid_y.numpy(), grid_z.numpy(), levels=20)
    plt.clabel(contour, inline=True, fontsize=8)

    plt.plot(path_x, path_y, "ro-", linewidth=2, markersize=6, label="Newton path")
    for i in range(len(path_x) - 1):
        plt.annotate(
            "",
            xy=(path_x[i + 1], path_y[i + 1]),
            xytext=(path_x[i], path_y[i]),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
        )

    plt.scatter([0], [0], color="black", s=60, label="minimum")
    plt.title("Newton Method for f(x, y) = x^2 + y^2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("newton_square_xy.png", dpi=160)
    print("Saved plot to newton_square_xy.png")


if __name__ == "__main__":
    main()
