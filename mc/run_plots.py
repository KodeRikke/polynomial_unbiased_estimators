"""Run all plotting helpers and save PDFs+PNGs to figures/.

Produces
--------
figures/gaps_vs_x.{pdf,png}          -- empirical var/MSE gap vs x  (Exp. 2)
figures/gap_vs_s_at_vertex.{pdf,png} -- MSE gap at worst-case x* vs s  (Exp. 3a)
figures/unsafe_bands.{pdf,png}       -- unsafe band widths, measured vs theory  (Exp. 3b)
figures/noise_comparison.{pdf,png}   -- Laplace vs Gaussian unbiased var/MSE  (Exp. 4)
figures/Q_residuals.{pdf,png}        -- MSE-gap residuals over quadratic grid  (Exp. Q)
figures/Q_var_residuals.{pdf,png}    -- Var-gap residuals (should be ≡ 0)  (Exp. Q)
figures/Q_loglog.{pdf,png}           -- log-log |MSE-gap| vs s, slope ≈ 4  (Exp. Q)
figures/Q_collapse.{pdf,png}         -- |MSE-gap| vs a²s⁴ collapse onto slope-1 line  (Exp. Q)
figures/Q_gap_vs_a.{pdf,png}         -- signed MSE-gap vs a, theory parabola per s  (Exp. Q)
figures/S_coincidence.{pdf,png}      -- min_x I_Var = min_x I_MSE = K  (Exp. S)
figures/P_inner.{pdf,png}            -- I_MSE(x) overlay for several s  (Exp. P)
figures/G_loglog.{pdf,png}           -- Laplace vs Gaussian on log-log axes  (Exp. G)
figures/G_multidelta.{pdf,png}       -- Gaussian family for δ∈{1e-10…1e-2}  (Exp. G Part A)
figures/G_quartic_dist.{pdf,png}     -- R=MSE_Gauss/MSE_Lap over 12500 quartics  (Exp. G Part B)

Runtime: roughly 6–10 minutes at N=200_000.
Experiments G Part A and B use closed-form MSE (no Monte Carlo, near-instant).
For a faster preview reduce N (e.g. N=20_000); results will be noisier.

Run from the project root:
    python mc/run_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parents[1]   # project root
_MC_DIR = Path(__file__).resolve().parent        # mc/

for _p in (_ROOT, _MC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mc_validation import (
    experiment_G_multidelta,
    experiment_G_quartic_sweep,
    experiment_Q,
    measure_band,
    plot_G_loglog,
    plot_G_multidelta,
    plot_G_quartic_distribution,
    plot_gap_vs_s_at_vertex,
    plot_gaps_vs_x,
    plot_noise_comparison,
    plot_P_inner,
    plot_Q_collapse,
    plot_Q_gap_vs_a,
    plot_Q_loglog,
    plot_Q_residuals,
    plot_Q_var_residuals,
    plot_S_coincidence,
    plot_unsafe_bands,
    print_estimator_table,
    sweep_epsilon_noise,
    sweep_s_at_vertex,
    sweep_x,
)

# ══════════════════════════════════════════════════════════════════════════════
# Shared parameters  — edit here to change all plots at once
# ══════════════════════════════════════════════════════════════════════════════

N    = 200_000   # Monte Carlo samples per point
SEED = 0         # master seed (all child seeds derived from this)

# Non-monotonic cubic  f = x³ − 3x  (a=1, b=0, c=-3, d=0)
# s_star = sqrt(1/3) ≈ 0.577 — used for Experiments 2, 3a, 3b, S, P
NONM_COEFFS = [0.0, -3.0, 0.0, 1.0]

# Degree-4 polynomial  f = x⁴ + x² + 1  — used for Experiments 4, G
DEG4_COEFFS = [1.0, 0.0, 1.0, 0.0, 1.0]

# Privacy parameters for the matched-privacy comparison
EPSILON_GRID = np.logspace(-1.0, 1.0, 40)   # 0.1 … 10 (log-spaced for Exp. G)
DELTA        = 1e-10
DELTA_SENS   = 1.0   # ℓ₁ = ℓ₂ sensitivity for a scalar query

X_EVAL = 1.0   # evaluation point for Experiments 4, G


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — gaps vs x  (Experiment 2)
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 1/10: gaps vs x  …", flush=True)

res_sweep_x = sweep_x(
    NONM_COEFFS,
    s      = 0.5,
    x_grid = np.linspace(-3.0, 3.0, 50),
    N      = N,
    seed   = SEED,
)
fig1 = plot_gaps_vs_x(res_sweep_x)
print("  saved: figures/gaps_vs_x.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — MSE gap at vertex vs s  (Experiment 3a)
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 2/10: gap at vertex vs s  …", flush=True)

res_sweep_s = sweep_s_at_vertex(
    NONM_COEFFS,
    s_grid = np.linspace(0.1, 1.2, 30),
    N      = N,
    seed   = SEED,
)
fig2 = plot_gap_vs_s_at_vertex(res_sweep_s)
print("  saved: figures/gap_vs_s_at_vertex.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — unsafe bands  (Experiment 3b)
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 3/10: unsafe bands  …", flush=True)

res_band = measure_band(
    NONM_COEFFS,
    s      = 0.3,
    x_grid = np.linspace(-3.0, 3.0, 80),
    N      = N,
    seed   = SEED,
)
fig3 = plot_unsafe_bands(res_band)
measured = res_band["measured"]
print(f"  var band width : {measured['var_band_width']:.4f}")
print(f"  mse band width : {measured['mse_band_width']:.4f}")
print(f"  width ratio    : {measured['width_ratio']:.4f}  (theory ≈ 1.2247)")
print("  saved: figures/unsafe_bands.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Laplace vs Gaussian noise comparison  (Experiment 4, linear axes)
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 4/10: Laplace vs Gaussian comparison (linear)  …", flush=True)

res_eps = sweep_epsilon_noise(
    DEG4_COEFFS,
    x            = X_EVAL,
    epsilon_grid = np.linspace(0.5, 5.0, 20),
    delta        = DELTA,
    Delta        = DELTA_SENS,
    N            = N,
    seed         = SEED,
)
fig4 = plot_noise_comparison(res_eps)
print("  saved: figures/noise_comparison.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Figures 5-7 — Experiment Q: exhaustive quadratic validation
# ══════════════════════════════════════════════════════════════════════════════

print("Figures 5-9/12: Experiment Q — exhaustive quadratic sweep  …", flush=True)

expQ = experiment_Q(
    N    = 50_000,
    seed = SEED,
)
fig5 = plot_Q_residuals(expQ,     Delta=DELTA_SENS)
fig6 = plot_Q_var_residuals(expQ,  Delta=DELTA_SENS)
fig7 = plot_Q_loglog(              Delta=DELTA_SENS, N=50_000, seed=SEED)
fig8c = plot_Q_collapse(expQ,      Delta=DELTA_SENS)
fig8d = plot_Q_gap_vs_a(expQ,      Delta=DELTA_SENS)
print(f"  configs tested:     {len(expQ['records'])}")
print(f"  max |residual|:     {expQ['max_abs_resid_mse']:.3g}")
print(f"  max residual (SE):  {expQ['max_n_se_mse']:.2f}")
print(f"  frac within 3 SE:   {expQ['frac_pass_3se']*100:.1f}%")
print(f"  MSE-gap < 0:        {expQ['n_mse_neg']}/{len(expQ['records'])}  "
      f"({expQ['n_mse_neg']/len(expQ['records'])*100:.2f}%)")
print("  saved: figures/Q_residuals, Q_var_residuals, Q_loglog, Q_collapse, Q_gap_vs_a")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Experiment S: MSE-safe = Var-safe coincidence
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 10/12: Experiment S — MSE-safe ≡ Var-safe  …", flush=True)

fig8 = plot_S_coincidence(
    coeffs = NONM_COEFFS,
    Delta  = DELTA_SENS,
)
print("  saved: figures/S_coincidence.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Experiment P: noise pushes inner polynomial upward
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 11/12: Experiment P — I_MSE(x) for increasing noise  …", flush=True)

fig9 = plot_P_inner(
    coeffs = NONM_COEFFS,
    Delta  = DELTA_SENS,
)
print("  saved: figures/P_inner.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Experiment G: Laplace vs Gaussian on log-log axes
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 12/12: Experiment G — log-log Laplace vs Gaussian  …", flush=True)

res_eps_loglog = sweep_epsilon_noise(
    DEG4_COEFFS,
    x            = X_EVAL,
    epsilon_grid = EPSILON_GRID,
    delta        = DELTA,
    Delta        = DELTA_SENS,
    N            = N,
    seed         = SEED,
)
fig10 = plot_G_loglog(res_eps_loglog, Delta=DELTA_SENS)

print_estimator_table(
    degrees = (2, 3, 4),
    epsilon = 1.0,
    delta   = DELTA,
    Delta   = DELTA_SENS,
)
print("  saved: figures/G_loglog.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Figure 13 — Experiment G Part A: multi-δ matched-privacy curves
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 13/14: Experiment G Part A — multi-δ Laplace vs Gaussian  …", flush=True)

res_G_multidelta = experiment_G_multidelta(
    coeffs       = DEG4_COEFFS,
    epsilon_grid = EPSILON_GRID,
    delta_values = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
    Delta        = DELTA_SENS,
)
fig13 = plot_G_multidelta(res_G_multidelta)
print("  saved: figures/G_multidelta.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 14 — Experiment G Part B: exhaustive quartic sweep
# ══════════════════════════════════════════════════════════════════════════════

print("Figure 14/14: Experiment G Part B — quartic MSE-ratio sweep  …", flush=True)

res_G_quartic = experiment_G_quartic_sweep(
    epsilon_grid = EPSILON_GRID,
    delta        = DELTA,
    Delta        = DELTA_SENS,
)
fig14 = plot_G_quartic_distribution(res_G_quartic)
print(f"  quartics tested:  {res_G_quartic['n_polys']}")
print(f"  R_median at ε=0.1: {res_G_quartic['R_median'][0]:.3g}")
print(f"  R_median at ε=10:  {res_G_quartic['R_median'][-1]:.3g}")
frac_arr = res_G_quartic["frac_R_gt1"]
print(f"  Frac R>1 (Laplace wins): {min(frac_arr)*100:.0f}%–{max(frac_arr)*100:.0f}%")
print("  saved: figures/G_quartic_dist.{pdf,png}")


print("\nAll figures saved to", (_ROOT / "figures").resolve())
