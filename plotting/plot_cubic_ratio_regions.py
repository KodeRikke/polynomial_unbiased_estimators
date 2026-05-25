"""Plot safe ratio regions for cubic unbiased estimator analysis.

Reads outputs from scripts/analyze_cubic_ratio_regions.py and creates one panel
per epsilon showing where both MSE and variance are theoretically safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv_matrix(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return None, None, None, None

    if data.ndim == 0:
        rows = np.array([data])
    else:
        rows = data

    u_values = np.unique(rows["u"])
    v_values = np.unique(rows["v"])

    # Reshape from row list into [v, u] image grid.
    u_index = {u: i for i, u in enumerate(u_values)}
    v_index = {v: i for i, v in enumerate(v_values)}

    both = np.zeros((len(v_values), len(u_values)), dtype=float)
    mse = np.zeros_like(both)
    var = np.zeros_like(both)

    for row in rows:
        ui = u_index[row["u"]]
        vi = v_index[row["v"]]
        both[vi, ui] = row["both_safe"]
        mse[vi, ui] = row["mse_safe"]
        var[vi, ui] = row["var_safe"]

    return u_values, v_values, both, (mse, var)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cubic ratio safe regions from analysis CSVs")
    parser.add_argument("--analysis-dir", type=str, default="reports/analysis")
    parser.add_argument("--output", type=str, default="plots/monte_carlo/cubic_ratio_safe_regions.png")
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0])
    parser.add_argument("--mode", choices=["both", "mse", "var"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eps_list = args.epsilon_values
    n = len(eps_list)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.9 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])

    for idx, eps in enumerate(eps_list):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        token = str(eps).replace(".", "p")
        csv_path = analysis_dir / f"cubic_ratio_regions_eps_{token}.csv"

        if not csv_path.exists():
            ax.set_title(f"epsilon={eps} (missing)")
            ax.axis("off")
            continue

        u_vals, v_vals, both_grid, extra = load_csv_matrix(csv_path)
        if u_vals is None:
            ax.set_title(f"epsilon={eps} (empty)")
            ax.axis("off")
            continue

        mse_grid, var_grid = extra
        if args.mode == "both":
            grid = both_grid
            title_suffix = "MSE+Var safe"
        elif args.mode == "mse":
            grid = mse_grid
            title_suffix = "MSE safe"
        else:
            grid = var_grid
            title_suffix = "Var safe"

        ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[u_vals.min(), u_vals.max(), v_vals.min(), v_vals.max()],
            interpolation="nearest",
            cmap="Greens",
            vmin=0.0,
            vmax=1.0,
        )

        frac = float(np.mean(grid > 0.5))
        ax.set_title(f"epsilon={eps} | {title_suffix} | {frac:.1%}")
        ax.set_xlabel("u = b/a")
        ax.set_ylabel("v = c/a")
        ax.grid(alpha=0.15)

    total_axes = rows * cols
    for idx in range(n, total_axes):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle("Cubic Theoretical Safe Regions in Ratio Space", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
