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
        "cubic": q**3 + q**2 + q + 1,
        "chebyshev_T3": sp.chebyshevt(3, q),
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
        poly_names = ["quadratic", "cubic", "cubic_coeffs", "chebyshev_T3"]
        titles = {
            "quadratic": "Baseline quadratic",
            "cubic": "Baseline cubic",
            "cubic_coeffs": "Cubic (varied coeffs)",
            "chebyshev_T3": "Chebyshev (degree 3)",
        }
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=False)
    
    axes = np.asarray(axes)
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
    
    for ax, poly_name in zip(axes.flat, poly_names):
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
    
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_cols = len(epsilons)
    # legend with added sub-title for line styles (solid vs dashed)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=legend_cols,
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
        poly_names = ["quadratic", "cubic", "cubic_coeffs", "chebyshev_T3"]
        poly_labels = {
            "quadratic": "Quadratic basecase",
            "cubic": "Cubic basecase",
            "cubic_coeffs": "Cubic coefficients",
            "chebyshev_T3": "Chebyshev T3",
        }
    
    n_eps = len(epsilons)
    if n_eps == 5:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=False)
        flat_axes = np.asarray(axes).ravel()
        plot_axes = flat_axes[:5]
        legend_ax = flat_axes[5]
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
    poly_colors = plt.cm.tab10(np.linspace(0, 1, len(poly_names)))

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
        default=-1.5,
        help="Lower q bound for the presentation figure",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        default=1.5,
        help="Upper q bound for the presentation figure",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.5, 3.0, 5.0],

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


def main() -> None:
    args = parse_args()
    reports = build_reports()
    out_dir = Path(args.output_dir)

    # Set 1: Fixed polynomial per plot (normal-degree and higher-degree)
    normal_variance_path = out_dir / "symbolic_presentation_normal_polynomial_overlay_variance.png"
    normal_mse_path = out_dir / "symbolic_presentation_normal_polynomial_overlay_mse.png"
    higher_variance_path = out_dir / "symbolic_presentation_higher_polynomial_overlay_variance.png"
    higher_mse_path = out_dir / "symbolic_presentation_higher_polynomial_overlay_mse.png"

    render_figure_polynomial_overlay(
        reports=reports,
        metric="variance_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=normal_variance_path,
        is_higher_degree=False,
    )
    render_figure_polynomial_overlay(
        reports=reports,
        metric="mse_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=normal_mse_path,
        is_higher_degree=False,
    )
    render_figure_polynomial_overlay(
        reports=reports,
        metric="variance_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=higher_variance_path,
        is_higher_degree=True,
    )
    render_figure_polynomial_overlay(
        reports=reports,
        metric="mse_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=higher_mse_path,
        is_higher_degree=True,
    )

    # Set 2: Fixed epsilon per plot (normal-degree and higher-degree)
    normal_eps_variance_path = out_dir / "symbolic_presentation_normal_polynomial_epsilon_variance.png"
    normal_eps_mse_path = out_dir / "symbolic_presentation_normal_polynomial_epsilon_mse.png"
    higher_eps_variance_path = out_dir / "symbolic_presentation_higher_polynomial_epsilon_variance.png"
    higher_eps_mse_path = out_dir / "symbolic_presentation_higher_polynomial_epsilon_mse.png"

    render_figure_epsilon_fixed(
        reports=reports,
        metric="variance_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=normal_eps_variance_path,
        is_higher_degree=False,
    )
    render_figure_epsilon_fixed(
        reports=reports,
        metric="mse_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=normal_eps_mse_path,
        is_higher_degree=False,
    )
    render_figure_epsilon_fixed(
        reports=reports,
        metric="variance_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=higher_eps_variance_path,
        is_higher_degree=True,
    )
    render_figure_epsilon_fixed(
        reports=reports,
        metric="mse_ratio",
        q_range=(args.q_min, args.q_max),
        epsilons=args.epsilons,
        delta_value=args.delta,
        Delta_value=args.Delta,
        out_path=higher_eps_mse_path,
        is_higher_degree=True,
    )

    print(f"Saved 8 presentation figures in: {out_dir}")


if __name__ == "__main__":
    main()