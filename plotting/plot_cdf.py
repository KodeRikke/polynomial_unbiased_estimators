"""
CDF plotter for P_h(Z) = h(q+Z) - f(q) and P_g(Z) = g(q+Z) - f(q)
under Laplace noise Z ~ Lap(0, s).

Each case in the input list produces one subplot panel.
Panel layout is controlled by the `arrangement` parameter of plot_cdf.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cdf import laplace_poly_coeffs, cdf_of_polynomial_in_laplace


def _draw_panel(ax, case):
    """Render one CDF panel onto ax."""
    f_sym        = case["f_sym"]
    q_val        = case["q_val"]
    s_val        = case["s_val"]
    t_min, t_max = case["t_range"]
    label        = case.get("label", f"q={q_val}, s={s_val}")
    t_n          = case.get("t_n", 400)

    t_grid = np.linspace(t_min, t_max, t_n)

    coeffs_h = laplace_poly_coeffs(f_sym, q_val, s_val, biasedness="naive")
    coeffs_g = laplace_poly_coeffs(f_sym, q_val, s_val, biasedness="unbiased")

    F_h = cdf_of_polynomial_in_laplace(coeffs_h, s_val, t_grid)
    F_g = cdf_of_polynomial_in_laplace(coeffs_g, s_val, t_grid)

    ax.plot(t_grid, F_h, label=r"naive $h$",    color="#11BBE6", lw=2)
    ax.plot(t_grid, F_g, label=r"unbiased $g$", color="#EA3364", lw=2)
    ax.axvline(0, color="gray", lw=0.6, linestyle="-")
    ax.set_title(label)
    ax.set_xlabel(r"$t$")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")


def plot_cdf(cases, *, save_path="plots/cdf", filename="cdf_comparison", arrangement=None):
    """
    Plot CDF curves for the naive and unbiased estimators under Laplace noise.

    Parameters
    ----------
    cases : list of dict
        Each dict defines one subplot panel and must contain:
            f_sym   : sympy expression for f in terms of sp.Symbol('q')
            q_val   : concrete value of q
            s_val   : Laplace noise scale  s = Delta/epsilon
            t_range : (t_min, t_max) — range for the CDF threshold axis
            label   : title string for the panel (LaTeX math supported)
        Optional key:
            t_n     : number of t-grid points (default 400)
    save_path : str
        Directory where the figure is saved (created automatically if absent).
    filename : str
        Output filename stem — saved as <save_path>/<filename>.png.
    arrangement : None or list of list of int
        Controls which panel goes where.
        None (default)    — single row, cases in order left to right.
        [[0, 3], [1, 2]]  — 2×2 grid: row 0 gets cases[0] and cases[3],
                            row 1 gets cases[1] and cases[2].
    """
    if arrangement is None:
        arrangement = [list(range(len(cases)))]

    nrows = len(arrangement)
    ncols = max(len(row) for row in arrangement)

    _, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.2 * nrows),
        sharey=True,
        squeeze=False,          # always returns a 2-D array
    )

    for r, row_indices in enumerate(arrangement):
        for c, idx in enumerate(row_indices):
            _draw_panel(axes[r, c], cases[idx])
        for c in range(len(row_indices), ncols):
            axes[r, c].set_visible(False)

    # y-label on the leftmost visible column of every row
    for r in range(nrows):
        axes[r, 0].set_ylabel(r"$\text{Pr}[\text{error} \leq t]$")

    plt.tight_layout()
    Path(save_path).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(save_path, filename + ".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
