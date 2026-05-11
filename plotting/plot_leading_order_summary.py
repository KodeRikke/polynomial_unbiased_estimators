"""Compact visualization of leading-order noise term analysis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize leading-order noise term signs")
    parser.add_argument(
        "--csv",
        type=str,
        default="reports/analysis/leading_order_noise_terms.csv",
        help="Path to leading-order CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plot",
    )
    return parser.parse_args()


def extract_degree(poly_name: str) -> int:
    """Extract polynomial degree from name."""
    if "quadratic" in poly_name:
        return 2
    if "cubic" in poly_name:
        return 3
    if "chebyshev_T3" in poly_name:
        return 3
    if "chebyshev_T9" in poly_name:
        return 9
    if "high_degree" in poly_name:
        return 10
    return 0


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Filter to multiple q values: 0, 0.01, 0.1, 1
    # Create separate visualizations or one comprehensive one
    q_values = [("q0", "sign_q0"), ("q001", "sign_q001"), ("q01", "sign_q01"), ("q02", "sign_q02"), ("q03", "sign_q03"), ("q04", "sign_q04"), ("q05", "sign_q05"), ("q1", "sign_q1")]
    q_labels = ["q=0", "q=0.001", "q=0.1", "q=0.2", "q=0.3", "q=0.4", "q=0.5", "q=1"]

    polys = df["polynomial"].unique()
    noises = df["noise"].unique()
    metrics = df["metric_diff"].unique()

    # Extract degrees
    poly_degrees = {p: extract_degree(p) for p in polys}
    polys_sorted = sorted(polys, key=lambda p: poly_degrees[p])

    # Create figure with subplots for each metric × q value (2 metrics × 4 q-values)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for metric_idx, metric in enumerate(sorted(metrics)):
        for q_idx, (q_col_name, q_sign_col) in enumerate(q_values):
            ax_idx = metric_idx * len(q_values) + q_idx
            ax = axes[ax_idx]
            
            # Collect data: rows = polynomials, cols = (noise model)
            y_pos = np.arange(len(polys_sorted))
            
            for col_idx, noise in enumerate(sorted(noises)):
                signs = []
                for poly in polys_sorted:
                    row = df[(df["polynomial"] == poly) & (df["noise"] == noise) & (df["metric_diff"] == metric)]
                    if len(row) > 0:
                        sign_str = row.iloc[0][q_sign_col]
                        if sign_str == "+":
                            signs.append(+1)
                        elif sign_str == "-":
                            signs.append(-1)
                        else:
                            signs.append(0)
                    else:
                        signs.append(0)
                
                # Plot as colored patches
                x_offset = col_idx * 1.5
                for y, sign_val in enumerate(signs):
                    if sign_val > 0:
                        color = "green"
                        label_text = "+"
                    elif sign_val < 0:
                        color = "red"
                        label_text = "−"
                    else:
                        color = "gray"
                        label_text = "?"
                    
                    rect = mpatches.FancyBboxPatch(
                        (x_offset - 0.4, y - 0.4),
                        0.8,
                        0.8,
                        boxstyle="round,pad=0.05",
                        edgecolor="black",
                        facecolor=color,
                        alpha=0.6,
                        linewidth=1.5,
                    )
                    ax.add_patch(rect)
                    ax.text(x_offset, y, label_text, ha="center", va="center", fontsize=14, fontweight="bold")
            
            # Set axes
            ax.set_xlim(-0.7, len(noises) * 1.5 - 0.8)
            ax.set_ylim(-0.7, len(polys_sorted) - 0.3)
            ax.invert_yaxis()
            ax.set_xticks([col_idx * 1.5 for col_idx in range(len(noises))])
            ax.set_xticklabels(sorted(noises), fontsize=9)
            #metric_name = metric.replace("naive_", "").replace("-unbiased_", " - unbiased ")
            #ax.set_title(f"{metric_name}\n{q_labels[q_idx]}", fontsize=11, fontweight="bold")
            ax.set_title(f"{q_labels[q_idx]}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Noise model", fontsize=10)
            # Show polynomial labels only in top-left corner (metric_idx=0, q_idx=0)
            if metric_idx == 0 and q_idx == 0:
                ax.set_ylabel("Polynomial", fontsize=10)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([f"{p} (deg {poly_degrees[p]})" for p in polys_sorted], fontsize=9)
            # the subplot whose label is q=0.3
            # OBS, hardcoded s.t. the polynomials are also shown in the bottom-left corner.
            elif q_labels[q_idx] == "q=0.3":
                ax.set_ylabel("Polynomial", fontsize=10)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([f"{p} (deg {poly_degrees[p]})" for p in polys_sorted], fontsize=9)
            else:
                ax.set_yticks(y_pos)
                ax.set_yticklabels([])
            ax.grid(True, alpha=0.2, axis="y")

    # Legend
    green_patch = mpatches.Patch(facecolor="green", alpha=0.6, edgecolor="black", label="+ (unbiased better)")
    red_patch = mpatches.Patch(facecolor="red", alpha=0.6, edgecolor="black", label="− (naive better)")
    fig.legend(
        handles=[green_patch, red_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=10,
        frameon=True,
    )

    fig.suptitle(
        "Leading-order noise term analysis at different q values. At asymptotic noise limit ε→∞, such that:\n"
        "Δ/ε→0 for Laplace expansion (fixed Δ=1) and σ→0 for Gaussian expansion.\n"
        "Each leading term sign is calculated from naive MSE - unbiased MSE",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    
    out_path = out_dir / "leading_order_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
