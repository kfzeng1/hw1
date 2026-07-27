import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch


HESSIAN = torch.tensor([[2.0, 1.0], [1.0, 4.0]], dtype=torch.float64)


@dataclass
class NewtonStep:
    step: int
    x1: float
    x2: float
    value: float
    grad_x1: float
    grad_x2: float
    step_x1: float
    step_x2: float
    next_x1: float
    next_x2: float


def objective(point):
    x1, x2 = point
    return x1 * x1 + 2.0 * x2 * x2 + x1 * x2


def gradient(point):
    x1, x2 = point
    return torch.tensor([2.0 * x1 + x2, x1 + 4.0 * x2], dtype=torch.float64)


def assert_positive_definite(matrix):
    eigenvalues = torch.linalg.eigvalsh(matrix)
    if torch.any(eigenvalues <= 0):
        raise ValueError("Hessian is not positive definite.")
    return eigenvalues


def newton_minimize(start_x1, start_x2, tolerance=1e-10, max_iter=20):
    assert_positive_definite(HESSIAN)
    point = torch.tensor([float(start_x1), float(start_x2)], dtype=torch.float64)
    history = []

    for step in range(1, max_iter + 1):
        grad = gradient(point)
        newton_direction = torch.linalg.solve(HESSIAN, grad)
        next_point = point - newton_direction

        history.append(
            NewtonStep(
                step=step,
                x1=point[0].item(),
                x2=point[1].item(),
                value=objective(point).item(),
                grad_x1=grad[0].item(),
                grad_x2=grad[1].item(),
                step_x1=newton_direction[0].item(),
                step_x2=newton_direction[1].item(),
                next_x1=next_point[0].item(),
                next_x2=next_point[1].item(),
            )
        )

        if torch.linalg.norm(next_point - point).item() < tolerance:
            point = next_point
            break
        point = next_point

    return point, objective(point).item(), history


def write_iteration_log(history, point, value, output_path):
    lines = [
        "Newton method for f(x1, x2) = x1^2 + 2x2^2 + x1x2",
        "gradient = [2x1 + x2, x1 + 4x2]",
        "Hessian = [[2, 1], [1, 4]]",
        "",
    ]
    for item in history:
        lines.append(
            "step {step}: x=({x1:.10f}, {x2:.10f}), f={value:.10f}, "
            "grad=({grad_x1:.10f}, {grad_x2:.10f}), "
            "newton_step=({step_x1:.10f}, {step_x2:.10f}), "
            "next=({next_x1:.10f}, {next_x2:.10f})".format(**item.__dict__)
        )
    lines.extend(
        [
            "",
            f"minimum point: ({point[0].item():.10f}, {point[1].item():.10f})",
            f"minimum value: {value:.10f}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_iteration_csv(history, output_path):
    fieldnames = [
        "step",
        "x1",
        "x2",
        "f",
        "grad_x1",
        "grad_x2",
        "newton_step_x1",
        "newton_step_x2",
        "next_x1",
        "next_x2",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    "step": item.step,
                    "x1": f"{item.x1:.10f}",
                    "x2": f"{item.x2:.10f}",
                    "f": f"{item.value:.10f}",
                    "grad_x1": f"{item.grad_x1:.10f}",
                    "grad_x2": f"{item.grad_x2:.10f}",
                    "newton_step_x1": f"{item.step_x1:.10f}",
                    "newton_step_x2": f"{item.step_x2:.10f}",
                    "next_x1": f"{item.next_x1:.10f}",
                    "next_x2": f"{item.next_x2:.10f}",
                }
            )


def save_iteration_table(history, point, value, output_path):
    headers = [
        "step",
        "x1",
        "x2",
        "f",
        "grad x1",
        "grad x2",
        "next x1",
        "next x2",
    ]
    rows = [
        [
            item.step,
            f"{item.x1:.6f}",
            f"{item.x2:.6f}",
            f"{item.value:.6f}",
            f"{item.grad_x1:.6f}",
            f"{item.grad_x2:.6f}",
            f"{item.next_x1:.6f}",
            f"{item.next_x2:.6f}",
        ]
        for item in history
    ]

    fig_height = max(3.0, 1.2 + 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title(
        f"Newton iterations, minimum=({point[0].item():.6f}, {point[1].item():.6f}), f={value:.6f}",
        pad=16,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_newton_path(history, output_path):
    grid_x1 = torch.linspace(-5, 5, 240)
    grid_x2 = torch.linspace(-5, 5, 240)
    mesh_x1, mesh_x2 = torch.meshgrid(grid_x1, grid_x2, indexing="xy")
    values = mesh_x1**2 + 2.0 * mesh_x2**2 + mesh_x1 * mesh_x2

    path_x1 = [item.x1 for item in history] + [history[-1].next_x1]
    path_x2 = [item.x2 for item in history] + [history[-1].next_x2]

    plt.figure(figsize=(7, 6))
    contour = plt.contour(mesh_x1.numpy(), mesh_x2.numpy(), values.numpy(), levels=24)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.plot(path_x1, path_x2, "o-", color="#c43c39", linewidth=2, markersize=6, label="Newton path")

    for index in range(len(path_x1) - 1):
        plt.annotate(
            "",
            xy=(path_x1[index + 1], path_x2[index + 1]),
            xytext=(path_x1[index], path_x2[index]),
            arrowprops={"arrowstyle": "->", "color": "#c43c39", "lw": 1.8},
        )

    plt.scatter([0], [0], color="black", s=60, label="minimum")
    plt.title("Newton method for f(x1, x2) = x1^2 + 2x2^2 + x1x2")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimize f(x1,x2)=x1^2+2x2^2+x1x2 with Newton's method."
    )
    parser.add_argument("--start-x1", default=3.0, type=float)
    parser.add_argument("--start-x2", default=4.0, type=float)
    parser.add_argument("--tolerance", default=1e-10, type=float)
    parser.add_argument("--max-iter", default=20, type=int)
    parser.add_argument("--output-dir", default="./results", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    point, value, history = newton_minimize(
        start_x1=args.start_x1,
        start_x2=args.start_x2,
        tolerance=args.tolerance,
        max_iter=args.max_iter,
    )

    print("Newton method for f(x1, x2) = x1^2 + 2x2^2 + x1x2")
    print(f"start point: ({args.start_x1}, {args.start_x2})")
    print("step       x1       x2          f     grad_x1     grad_x2    next_x1    next_x2")
    for item in history:
        print(
            f"{item.step:>4d}  {item.x1:>7.3f}  {item.x2:>7.3f}  "
            f"{item.value:>9.6f}  {item.grad_x1:>10.6f}  {item.grad_x2:>10.6f}  "
            f"{item.next_x1:>9.6f}  {item.next_x2:>9.6f}"
        )
    print(f"minimum point: ({point[0].item():.10f}, {point[1].item():.10f})")
    print(f"minimum value: {value:.10f}")

    save_iteration_table(history, point, value, args.output_dir / "iteration_process.png")
    save_newton_path(history, args.output_dir / "newton_path.png")
    write_iteration_log(history, point, value, args.output_dir / "iteration_log.txt")
    write_iteration_csv(history, args.output_dir / "iteration_history.csv")


if __name__ == "__main__":
    main()
