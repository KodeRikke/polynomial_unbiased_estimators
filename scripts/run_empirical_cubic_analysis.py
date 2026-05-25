"""Run all four empirical cubic analysis figures and save to plots/cubic/.

Figures produced
----------------
1. safe_zone_boundary          — safety boundary γ=(β²−27s²)/3 in (β,γ)-space
2. safe_fraction_vs_epsilon    — fraction of (a,b,c) integer grid with K≥0 vs ε
3. gap_function_three_noise_scales — I_MSE_cubic vertical-translation demo
4. monte_carlo_validation      — empirical vs theoretical MSE gap scatter + relative error

Each figure is saved as both .png and .pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting.plot_empirical_cubic_analysis import (
    plot_safe_zone_boundary,
    plot_safe_fraction_vs_epsilon,
    plot_gap_function,
    plot_monte_carlo_validation,
)


def _save(fig: plt.Figure, stem: Path) -> None:
    """Save figure as .png and .pdf, then close it."""
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    SEED = 4243
    DELTA = 1.0
    OUT_DIR = Path("plots/cubic")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Safe zone boundary ─────────────────────────────────────────
    # Five s values whose γ=0 crossings (β = ±3√3·s) all fall within β ∈ (-10,10):
    #   s=0.3 → ±1.56,  s=0.5 → ±2.60,  s=0.75 → ±3.90,
    #   s=1.0 → ±5.20,  s=1.5 → ±7.79
    # Parabola minima (γ = −9s²): −0.81, −2.25, −5.06, −9.0, −20.25  (all ≥ −25).
    # Shade for s=0.5 (ε=2).
    fig1 = plot_safe_zone_boundary(
        noise_scales=[0.3, 0.5, 0.75, 1.0, 1.5],
        shade_s=0.5,
        Delta=DELTA,
        beta_range=(-10.0, 10.0),
        gamma_range=(-25.0, 5.0),
    )
    _save(fig1, OUT_DIR / "safe_zone_boundary")
    print("Saved: safe_zone_boundary.{png,pdf}")

    # ── Figure 2: Safe fraction vs epsilon ────────────────────────────────────
    # Integer grid: a ∈ [−15,15]\{0}, b ∈ [−15,15], c ∈ [−15,15].
    # For each ε, count fraction satisfying K = 54a²s² + 6ac − 2b² ≥ 0
    # using the closed-form directly (no grid over x).
    epsilon_grid = np.linspace(0.05, 10.0, 100)
    grid_sizes = [5, 15, 80]

    fig2 = plot_safe_fraction_vs_epsilon(
        epsilon_grid=epsilon_grid,
        grid_sizes=grid_sizes,
        Delta=DELTA,
    )
    _save(fig2, OUT_DIR / "safe_fraction_vs_epsilon")
    print("Saved: safe_fraction_vs_epsilon.{png,pdf}")

    # ── Figure 3: Gap function ─────────────────────────────────────────────────
    # f(q) = q³ + 3q²  (non-monotonic: f′ = 3q² + 6q = 3q(q+2), roots at 0, −2)
    # K = 54s² − 18 for this polynomial.
    # Three noise scales: below threshold (s=0.3, K<0), at threshold (s=1/√3, K=0),
    # above threshold (s=1, K>0).  Same linestyle for all; visual story is vertical shift.
    A3, B3, C3 = 1.0, 3.0, 0.0
    fig3 = plot_gap_function(
        a=A3,
        b=B3,
        c=C3,
        noise_scales=[0.1, 1.0 / np.sqrt(3.0), 2.0],
        x_vals=np.linspace(-5.0, 5.0, 500),
    )
    _save(fig3, OUT_DIR / "gap_function_three_noise_scales")
    print("Saved: gap_function_three_noise_scales.{png,pdf}")

    # ── Figure 4: Monte Carlo validation ─────────────────────────────────────
    # 7 tuples spanning ≥ 3 orders of magnitude on both sides of zero.
    # (a, b, c, q, s) with K = 54a²s² + 6ac − 2b²:
    #
    #   (1,  3, 0,    0,    2  )  K=+198    gap ≈ −14400   (very safe)
    #   (1,  3, 0,    0,    1  )  K=+36     gap ≈    −252   (safe)
    #   (1,  3, 0,    0,    0.5)  K=−4.5    gap ≈    −5.63  (safe at q=0)
    #   (1,  3, 0,   −1,  1/√3)  K=0       gap =       0   (marginal)
    #   (1,  3, 0,   −1,   0.5)  K=−4.5    gap ≈    +1.13  (mildly unsafe)
    #   (1, 10, 0, −10/3,  0.5)  K=−186.5  gap ≈   +46.6  (unsafe)
    #   (1, 15, 0,   −5,   1  )  K=−396    gap ≈  +1584   (deeply unsafe)
    #
    # Negative gaps: −14400, −252, −5.63  → ≈ 3.4 orders
    # Positive gaps:  +1.13,  +46.6, +1584 → ≈ 3.1 orders
    TUPLES: list[tuple[float, float, float, float, float]] = [
        (1.0,  3.0,  0.0,   0.0,            2.0),
        (1.0,  3.0,  0.0,   0.0,            1.0),
        (1.0,  3.0,  0.0,   0.0,            0.5),
        (1.0,  3.0,  0.0,  -1.0,  1.0 / np.sqrt(3.0)),
        (1.0,  3.0,  0.0,  -1.0,            0.5),
        (1.0, 10.0,  0.0, -10.0 / 3.0,      0.5),
        (1.0, 15.0,  0.0,  -5.0,            1.0),
    ]

    # plot_monte_carlo_validation raises AssertionError (non-zero exit) if any
    # tuple's empirical gap deviates from the closed form by > 3 standard errors.
    fig4, verif = plot_monte_carlo_validation(TUPLES, N=10_000, seed=SEED)
    _save(fig4, OUT_DIR / "monte_carlo_validation")
    print("Saved: monte_carlo_validation.{png,pdf}")

    # Summary
    print(f"\nAll figures saved to {OUT_DIR.resolve()}/")
    print(
        f"\n[MC] All {verif['n_tuples']} tuples passed 3-SE verification."
        f"  Symlog linthresh = {verif['linthresh']:.4g}"
    )
    print(f"\n{'tuple (a,b,c,q,s)':>40}  {'K':>8}  {'gap_emp':>12}  {'gap_theo':>12}  {'disc (SE)':>10}")
    print("-" * 90)
    for r in verif["results"]:
        a, b, c, q, s = r["tuple"]
        print(
            f"  ({a:g},{b:g},{c:g},{q:.4g},{s:.4g})".ljust(40)
            + f"  {r['K']:+8.3g}  {r['gap_emp']:+12.5g}  {r['gap_theory']:+12.5g}"
            + f"  {r['discrepancy_se']:10.2f}"
        )


if __name__ == "__main__":
    main()
