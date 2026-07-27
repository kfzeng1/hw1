import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch


@dataclass
class NewtonStep:
    step: int
    x: float
    y: float
    value: float
    grad_x: float
    grad_y: float
    next_x: float
    next_y: float


def objective(x, y):
    return x * x + y * y


def gradient(x, y):
    return torch.tensor([2.0 * x, 2.0 * y], dtype=torch.float64)


def hessian():
    return torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)


def newton_minimize(start_x, start_y, eps=1e-10, max_iter=20):
    point = torch.tensor([float(start_x), float(start_y)], dtype=torch.float64)
    history = []

    for step in range(1, max_iter + 1):
        x, y = point.tolist()
        grad = gradient(x, y)
        delta = torch.linalg.solve(hessian(), grad)
        next_point = point - delta
        next_x, next_y = next_point.tolist()

        history.append(
            NewtonStep(
                step=step,
                x=x,
                y=y,
                value=objective(x, y),
                grad_x=grad[0].item(),
                grad_y=grad[1].item(),
                next_x=next_x,
                next_y=next_y,
            )
        )

        if torch.norm(next_point - point).item() < eps:
            point = next_point
            break
        point = next_point

    min_x, min_y = point.tolist()
    return min_x, min_y, objective(min_x, min_y), history


def save_iteration_table(history, min_x, min_y, min_value, output_path):
    headers = ["step", "x", "y", "f(x,y)", "grad_x", "grad_y", "next_x", "next_y"]
    rows = [
        [
            step.step,
            f"{step.x:.6f}",
            f"{step.y:.6f}",
            f"{step.value:.6f}",
            f"{step.grad_x:.6f}",
            f"{step.grad_y:.6f}",
            f"{step.next_x:.6f}",
            f"{step.next_y:.6f}",
        ]
        for step in history
    ]

    fig_height = max(2.8, 1.2 + 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title(
        f"Newton Iterations, minimum=({min_x:.6f}, {min_y:.6f}), f={min_value:.6f}",
        pad=16,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_newton_path(history, output_path):
    x = torch.linspace(-5, 5, 200)
    y = torch.linspace(-5, 5, 200)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    grid_z = grid_x**2 + grid_y**2

    path_x = [item.x for item in history] + [history[-1].next_x]
    path_y = [item.y for item in history] + [history[-1].next_y]

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
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_iteration_log(history, min_x, min_y, min_value, output_path):
    lines = [
        "Newton method for minimizing f(x, y) = x^2 + y^2",
        "step,x,y,f(x,y),grad_x,grad_y,next_x,next_y",
    ]
    for item in history:
        lines.append(
            f"{item.step},{item.x:.10f},{item.y:.10f},{item.value:.10f},"
            f"{item.grad_x:.10f},{item.grad_y:.10f},{item.next_x:.10f},{item.next_y:.10f}"
        )
    lines.extend(
        [
            "",
            f"minimum point: ({min_x:.10f}, {min_y:.10f})",
            f"minimum value: {min_value:.10f}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use Newton's method to minimize f(x, y)=x^2+y^2."
    )
    parser.add_argument("--start-x", default=3.0, type=float)
    parser.add_argument("--start-y", default=4.0, type=float)
    parser.add_argument("--eps", default=1e-10, type=float)
    parser.add_argument("--max-iter", default=20, type=int)
    parser.add_argument("--output-dir", default="./results", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_x, min_y, min_value, history = newton_minimize(
        args.start_x, args.start_y, eps=args.eps, max_iter=args.max_iter
    )

    print("Newton method for minimizing f(x, y) = x^2 + y^2")
    print(f"start point: ({args.start_x}, {args.start_y})")
    print("step        x        y      f(x,y)      grad_x      grad_y     next_x     next_y")
    for item in history:
        print(
            f"{item.step:>4d}  {item.x:>7.3f}  {item.y:>7.3f}  "
            f"{item.value:>10.6f}  {item.grad_x:>10.6f}  {item.grad_y:>10.6f}  "
            f"{item.next_x:>9.6f}  {item.next_y:>9.6f}"
        )
    print(f"\nminimum point: ({min_x:.10f}, {min_y:.10f})")
    print(f"minimum value: {min_value:.10f}")

    path_plot = output_dir / "newton_path.png"
    table_plot = output_dir / "iteration_process.png"
    log_path = output_dir / "iteration_log.txt"
    save_newton_path(history, path_plot)
    save_iteration_table(history, min_x, min_y, min_value, table_plot)
    write_iteration_log(history, min_x, min_y, min_value, log_path)
    print(f"Saved Newton path to {path_plot}")
    print(f"Saved iteration table to {table_plot}")


if __name__ == "__main__":
    main()
