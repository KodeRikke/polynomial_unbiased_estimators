from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_estimators import EstimatorSystem
from noise_models import LaplaceNoiseModel, GaussianNoiseModel
from dp_calibration.SigmaFromEpsilon import SigmaFromEpsilon
from plotting.utility_plotting import add_reference_line, evaluate_on_grid


def build_reports() -> dict[str, dict]:
    q = sp.Symbol("q", real=True)
    x = sp.Symbol("x", real=True)
    sigma = sp.symbols("sigma", positive=True, real=True)
    Delta = sp.Symbol("Delta", real=True, positive=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)

    laplace_system = EstimatorSystem(
        noise_model=LaplaceNoiseModel(Delta=Delta, epsilon=epsilon),
        q=q,
        x=x,
    )

    gaussian_system = EstimatorSystem(
        noise_model=GaussianNoiseModel(sigma=sigma),
        q=q,
        x=x,
    )
    laplace_symbols = {"Delta": Delta, "epsilon": epsilon, "q": laplace_system.q, "x": laplace_system.x}
    gaussian_symbols = {"sigma": sigma, "q": gaussian_system.q, "x": gaussian_system.x}

    polynomials = {
        "quadratic": q**2 + q + 1,
        "quadratic_coeffs": 2 * q**2 - 3 * q + 1,
        "cubic": q**3 + q**2 + q + 1,
        "chebyshev_T3": sp.chebyshevt(3, q),
        "chebyshev_T4": sp.chebyshevt(4, q),
        "quartic": q**4 + q**3 + q**2 + q + 1,
        "quartic_coeffs": q**4 + 4 * q**3 + 3 * q**2 - 2 * q + 1,
        "cubic_coeffs": 4 * q**3 + 3 * q**2 - 2 * q + 1,
        "high_degree": q**10 + q**9 + q**8 + q**7 + q**6 + q**5 + q**4 + q**3 + q**2 + q + 1,
        "high_degree_coeffs": 2 * q**10 - 3 * q**9 + 5 * q**8 - 7 * q**7 + 11 * q**6 - 13 * q**5 + 17 * q**4 - 19 * q**3 + 23 * q**2 - 29 * q + 31,
        "chebyshev_T9": sp.chebyshevt(9, q),
    }
    reports = {}
    for name, poly in polynomials.items():
        laplace_report = laplace_system.compare(poly)
        gaussian_report = gaussian_system.compare(poly)
        reports[name] = {
            "laplace": laplace_report,
            "gaussian": gaussian_report,
        }

    return reports


def metric_expr(report: dict, metric: str) -> sp.Expr:
    if metric == "variance_ratio":
        return sp.simplify(report["unbiased"]["variance"] / report["naive"]["variance"])
    if metric == "mse_ratio":
        return sp.simplify(report["unbiased"]["mse"] / report["naive"]["mse"])
    raise ValueError(f"Unsupported metric: {metric}")


def render_figure_polynomial_overlay(
    *,
    reports: dict[str, dict],
    metric: str,
    q_range: tuple[float, float],
    epsilons: list[float],
    delta_value: float,
    Delta_value: float,
    out_path: Path,
    is_higher_degree: bool = False,
) -> None:
    """Render a grid with Laplace and Gaussian curves overlaid per polynomial.
    
    Each subplot shows a single polynomial with both noise models and all epsilons.
    """
    q_grid = np.linspace(q_range[0], q_range[1], 300)
    
    if is_higher_degree:
        poly_names = ["high_degree", "high_degree_coeffs", "chebyshev_T9"]
        titles = {
            "high_degree": "High-degree baseline",
            "high_degree_coeffs": "High-degree (varied coeffs)",
            "chebyshev_T9": "Chebyshev (degree 9)",
        }
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=False)
    else:
        poly_names = ["quadratic", 
                      "quadratic_coeffs",
                      "cubic", 
                      "cubic_coeffs", 
                      "quartic", 
                      "quartic_coeffs", 
                      #"chebyshev_T3", 
                      #"chebyshev_T4",
                      ]
        titles = {
            "quadratic": "Baseline quadratic",
            "quadratic_coeffs": "Quadratic (varied coeffs)",
            "cubic": "Baseline cubic",
            "cubic_coeffs": "Cubic (varied coeffs)",
            "quartic": "Baseline quartic",
            "quartic_coeffs": "Quartic (varied coeffs)",
            #"chebyshev_T3": "Chebyshev (degree 3)",
            #"chebyshev_T4": "Chebyshev (degree 4)",
        }
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=False)

    flat_axes = np.asarray(axes).ravel()
    n_polys   = len(poly_names)

    # Special layout for 5 polynomials:
    # 5 plots in a 3×2 grid with the last cell reserved for a shared legend.
    if n_polys == 5:
        plot_axes = flat_axes[:5]
        legend_ax = flat_axes[5]
        legend_ax.axis("off")
    else:
        plot_axes = flat_axes[:n_polys]
        legend_ax = None
        for empty_ax in flat_axes[n_polys:]:
            empty_ax.axis("off")

    q_symbol = sp.Symbol("q", real=True)
    epsilon_symbol = sp.Symbol("epsilon", real=True, positive=True)
    sigma_symbol = sp.Symbol("sigma", positive=True, real=True)
    Delta_symbol = sp.Symbol("Delta", real=True, positive=True)

    # Cache symbolic ratio expressions to avoid repeated simplify calls.
    metric_cache = {
        name: {
            "laplace": metric_expr(reports[name]["laplace"], metric),
            "gaussian": metric_expr(reports[name]["gaussian"], metric),
        }
        for name in poly_names
    }

    # Color map for epsilons
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(epsilons)))

    for ax, poly_name in zip(plot_axes, poly_names):
        report = reports[poly_name]

        # Plot Laplace and Gaussian together per epsilon so matching pairs stay adjacent.
        for idx, epsilon_value in enumerate(epsilons):
            expr_laplace = metric_cache[poly_name]["laplace"]
            subs_lap = {Delta_symbol: Delta_value, epsilon_symbol: epsilon_value}
            y_vals_lap = evaluate_on_grid(expr_laplace, q_symbol, q_grid, subs_lap)
            ax.plot(
                q_grid,
                y_vals_lap,
                color=colors[idx],
                linewidth=2,
                linestyle="-",
                label=f"Laplace ε={epsilon_value:g}",
            )

            expr_gaussian = metric_cache[poly_name]["gaussian"]
            sigma_value = SigmaFromEpsilon.numeric(epsilon_value, delta_value, Delta_value)
            subs_gaus = {sigma_symbol: sigma_value}
            y_vals_gaus = evaluate_on_grid(expr_gaussian, q_symbol, q_grid, subs_gaus)
            ax.plot(
                q_grid,
                y_vals_gaus,
                color=colors[idx],
                linewidth=2,
                linestyle="--",
                label=f"Gaussian ε={epsilon_value:g}",
            )

        ax.set_title(titles[poly_name], fontsize=11)
        ax.set_xlabel(r"$q$")
        ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3)
        add_reference_line("Variance Ratio" if metric == "variance_ratio" else "MSE Ratio", ax=ax)

    metric_name = "variance" if metric == "variance_ratio" else "MSE"
    degree_type = "Higher-degree" if is_higher_degree else "Normal degree"
    fig.suptitle(
        f"{degree_type} polynomials: symbolic {metric_name} ratios",
        y=0.98,
    )

    if legend_ax is not None:
        legend_ax.set_title(
            "Legends for all plots\nSolid: Laplace\nDashed: Gaussian",
            fontsize=12, pad=8,
        )
        legend_handles = [
            Line2D([0], [0], color=colors[idx], linewidth=2, linestyle="-")
            for idx in range(len(epsilons))
        ]
        legend_labels = [f"ε = {eps:g}" for eps in epsilons]
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            frameon=False,
            fontsize=10,
            handlelength=2.5,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        handles, labels = plot_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(epsilons),
            fontsize=9,
            frameon=False,
            title="Line styles: Laplace (solid), Gaussian (dashed)",
            title_fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_figure_epsilon_fixed(
    *,
    reports: dict[str, dict],
    metric: str,
    q_range: tuple[float, float],
    epsilons: list[float],
    delta_value: float,
    Delta_value: float,
    out_path: Path,
    is_higher_degree: bool = False,
) -> None:
    """Render a grid with different polynomials as curves, fixed epsilon per subplot.
    
    Each subplot is fixed to an epsilon value, and shows all polynomials as curves.
    X-axis is q, y-axis is ratio.
    """
    q_grid = np.linspace(q_range[0], q_range[1], 300)
    
    if is_higher_degree:
        poly_names = ["high_degree", "high_degree_coeffs", "chebyshev_T9"]
        poly_labels = {
            "high_degree": "High-deg basecase",
            "high_degree_coeffs": "High-deg coefficients",
            "chebyshev_T9": "Chebyshev T9",
        }
    else:
        poly_names = ["quadratic", 
                      "cubic", 
                      "quartic", 
                      "quadratic_coeffs",
                      "cubic_coeffs", 
                      "quartic_coeffs", 
                      #"chebyshev_T3", 
                      #"chebyshev_T4",
                      ]
        
        poly_labels = {
            "quadratic": "Baseline quadratic",
            "cubic": "Baseline cubic",
            "quartic": "Baseline quartic",
            "quadratic_coeffs": "Quadratic (varied coeffs)",
            "cubic_coeffs": "Cubic (varied coeffs)",
            "quartic_coeffs": "Quartic (varied coeffs)",
            #"chebyshev_T3": "Chebyshev (degree 3)",
            #"chebyshev_T4": "Chebyshev (degree 4)",
        }
    n_functions = len(poly_names)
    n_eps = len(epsilons)
    # Special layout for 5 epsilons:
    # 5 plots in a 3 x 2 grid with the last cell reserved for a shared legend
    if n_eps == 5: 
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=False)
        flat_axes = np.asarray(axes).ravel()
        plot_axes = flat_axes[:5]
        legend_ax = flat_axes[5]
    # Special layout for 3 epsilons: 
    # 3 plots in a 3 x 1 grid 
    elif n_eps == 3:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, sharey=False)
        flat_axes = np.asarray(axes).ravel()
        plot_axes = flat_axes[:3]
        legend_ax = None
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=False)
        flat_axes = np.asarray(axes).ravel()
        plot_axes = flat_axes[:n_eps]
        legend_ax = None

    for empty_ax in flat_axes[n_eps + (1 if legend_ax is not None else 0):]:
        empty_ax.axis("off")
    
    q_symbol = sp.Symbol("q", real=True)
    epsilon_symbol = sp.Symbol("epsilon", real=True, positive=True)
    sigma_symbol = sp.Symbol("sigma", positive=True, real=True)
    Delta_symbol = sp.Symbol("Delta", real=True, positive=True)
    
    # Cache symbolic ratio expressions to avoid repeated simplify calls.
    metric_cache = {
        name: {
            "laplace": metric_expr(reports[name]["laplace"], metric),
            "gaussian": metric_expr(reports[name]["gaussian"], metric),
        }
        for name in poly_names
    }

    # Color map for polynomials
    if n_functions == 6: # define colors for 6 epsilons
        #             "quadratic", "cubic", "quartic", "quadratic_coeffs", "cubic_coeffs", "quartic_coeffs"
        #poly_colors = ["#3f4ada", "#37afde", "#1fcbc2", "#db79db", "#be2da9", "#ed0b6d"]
        #poly_colors = ["#85E4EB", "#ef66df","#F07A3F", "#2364A0", "#e20c90", "#CE4506"]
        
        poly_colors =[
            "#1B4965",  # quadratic        — dark blue
            "#9D0264",  # cubic            — dark red
            "#BC5800",  # quartic          — dark orange
            "#90DCEF",  # quadratic_coeffs — light blue
            "#F792F5",  # cubic_coeffs     — light red/pink
            "#F49A77",  # quartic_coeffs   — light orange
        ]
    else:
        poly_colors = plt.cm.tab10(np.arange(len(poly_names)) / 10)

    # place legend in empty subplot if available, otherwise rely on shared legend below the grid
    if legend_ax is not None:
        legend_ax.axis("off")
        legend_ax.set_title("Legends for all plots \nSolid: Laplace\nDashed: Gaussian", fontsize=12, pad=8)
        legend_handles = [
            Line2D([0], [0], color=poly_colors[idx], linewidth=2, linestyle="-")
            for idx in range(len(poly_names))
        ]
        legend_labels = [poly_labels[poly_name] for poly_name in poly_names]
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            frameon=False,
            fontsize=10,
            handlelength=2.5,
        )
    else:
        # Add a shared legend below the grid
        legend_handles = [
            Line2D([0], [0], color=poly_colors[idx], linewidth=2, linestyle="-")
            for idx in range(len(poly_names))
        ]
        legend_labels = [poly_labels[poly_name] for poly_name in poly_names]
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",            # anchor the legend's TOP-center...
            bbox_to_anchor=(0.5, 0.01),    # ...at y=0.06 (inside the reserved strip)
            ncol=2,
            fontsize=9,
            frameon=False,
            title="Line styles: Laplace (solid), Gaussian (dashed)",
            title_fontsize=10,
        )
    
    for ax, epsilon_value in zip(plot_axes, epsilons):

        for poly_idx, poly_name in enumerate(poly_names):
            # Plot Laplace (solid line)
            expr_laplace = metric_cache[poly_name]["laplace"]
            subs_lap = {Delta_symbol: Delta_value, epsilon_symbol: epsilon_value}
            y_vals_lap = evaluate_on_grid(expr_laplace, q_symbol, q_grid, subs_lap)
            ax.plot(
                q_grid,
                y_vals_lap,
                color=poly_colors[poly_idx],
                linewidth=2,
                linestyle="-",
                label=f"{poly_labels[poly_name]} (Lap)",
            )
            
            # Plot Gaussian (dashed line)
            expr_gaussian = metric_cache[poly_name]["gaussian"]
            sigma_value = SigmaFromEpsilon.numeric(epsilon_value, delta_value, Delta_value)
            subs_gaus = {sigma_symbol: sigma_value}
            y_vals_gaus = evaluate_on_grid(expr_gaussian, q_symbol, q_grid, subs_gaus)
            ax.plot(
                q_grid,
                y_vals_gaus,
                color=poly_colors[poly_idx],
                linewidth=2,
                linestyle="--",
                label=f"{poly_labels[poly_name]} (Gau)",
            )
        
        ax.set_title(rf"$\epsilon = {epsilon_value:g}$", fontsize=12)
        ax.set_xlabel(r"$q$")
        ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3)
        add_reference_line("Variance Ratio" if metric == "variance_ratio" else "MSE Ratio", ax=ax)
    
    metric_name = "variance" if metric == "variance_ratio" else "MSE"
    degree_type = "Higher-degree" if is_higher_degree else "Lower degree"
    fig.suptitle(
        f"{degree_type} polynomials: {metric_name} ratio vs q",
        y=0.98,
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact symbolic presentation figure")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/presentation",
        help="Directory where the presentation figures are saved",
    )
    parser.add_argument(
        "--q-min",
        type=float,
        default=-10.0,
        help="Lower q bound for the presentation figure",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=10.0,
        help="Upper q bound for the presentation figure",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.5, 3.0, 5.0],

        help="Epsilon values to draw as curves in each panel",
    )
    parser.add_argument(
        "--Delta",
        type=float,
        default=1.0,
        help="Sensitivity value used in the figure",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-10,
        help="Gaussian delta parameter placeholder, included for consistency",
    )
    return parser.parse_args()