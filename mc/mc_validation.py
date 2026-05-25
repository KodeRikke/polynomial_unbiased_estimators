"""Monte Carlo validation of variance/MSE-gap formulas for univariate polynomials
under Laplace and Gaussian DP noise.

Coefficient convention
----------------------
``coeffs = [c0, c1, ..., cn]`` lowest-degree-first, so
``f(x) = c0 + c1*x + ... + cn*x**n``.
This does NOT match numpy's ``np.polyval`` convention (highest-first), so we
always reverse before calling polyval.

For the closed-form formula functions the standard notation is used:
  quadratic: f = a*x**2 + b*x + c  →  a=coeffs[2], b=coeffs[1], c=coeffs[0]
  cubic:     f = a*x**3 + b*x**2 + c*x + d  →  a=coeffs[3], b=coeffs[2],
                                                  c=coeffs[1], d=coeffs[0]

Noise conventions
-----------------
Laplace: scale s = Δ/ε,  Z ~ Laplace(0, s),  Var(Z) = 2s².
Gaussian: sigma calibrated via ``calibrateAnalyticGaussianMechanism(ε, δ, Δ)``
          (Balle–Wang ICML'18).  Do NOT use the Dwork–Roth √(2 ln(1.25/δ)) formula.
For a scalar query, ℓ₁ = ℓ₂ sensitivity, so the same Δ is passed to both.

Standard errors
---------------
Means/biases: CLT SE = sample_std / √N.
Variances, MSEs, gaps: paired bootstrap with B=200 resamples of size N drawn
with replacement.  Both estimators are resampled together (same index vector)
to preserve the correlation between h(X_i) and g(X_i).

Randomness
----------
All draws use a local ``rng = np.random.default_rng(seed)``.  No global state.
"""
from __future__ import annotations

import sys
from math import sqrt as _sqrt
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# ── Project path ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dp_calibration.gaussian import calibrateAnalyticGaussianMechanism
from dp_estimators import EstimatorSystem
from noise_models import GaussianNoiseModel, LaplaceNoiseModel

_FIGURES_DIR = _ROOT / "figures"

# ── Plot style constants (match plotting/plot_symbolic_presentation.py) ────────
_LW = 2
_GRID_ALPHA = 0.3
_FS_LABEL = 10
_FS_LEGEND = 8


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════

def _coeffs_to_sympy(coeffs: Sequence[float], q_sym: sp.Symbol) -> sp.Expr:
    """Convert lowest-degree-first list to SymPy polynomial in q_sym."""
    return sp.expand(sum(sp.sympify(c) * q_sym**k for k, c in enumerate(coeffs)))


def _f_value(coeffs: Sequence[float], x_val: float) -> float:
    """Evaluate polynomial at x_val (coeffs lowest-degree-first)."""
    return float(np.polyval(list(reversed(coeffs)), x_val))


def _sympy_to_numpy_poly(expr: sp.Expr, x_sym: sp.Symbol) -> np.ndarray:
    """Return highest-to-lowest-degree float coefficients for np.polyval."""
    poly = sp.Poly(sp.expand(expr), x_sym)
    return np.array([float(sp.N(c)) for c in poly.all_coeffs()])


def _build_eval_fns(coeffs, *, noise_type: str, s: Optional[float] = None,
                    sigma: Optional[float] = None):
    """Build numpy evaluators for naive h(x) and unbiased g(x).

    Uses the existing EstimatorSystem — does not re-derive any formula.
    Returns (h_eval, g_eval, h_str, g_str) where h_eval/g_eval are callables
    mapping a float array to a float array.
    """
    q_sym = sp.Symbol("q", real=True)
    x_sym = sp.Symbol("x", real=True)
    f_q = _coeffs_to_sympy(coeffs, q_sym)

    if noise_type == "laplace":
        if s is None:
            raise ValueError("s is required for noise_type='laplace'")
        model = LaplaceNoiseModel(Delta=float(s), epsilon=1.0)
    elif noise_type == "gaussian":
        if sigma is None:
            raise ValueError("sigma is required for noise_type='gaussian'")
        model = GaussianNoiseModel(sigma=float(sigma))
    else:
        raise ValueError(f"noise_type must be 'laplace' or 'gaussian', got {noise_type!r}")

    system = EstimatorSystem(model, "q", "x")
    h_expr = sp.expand(system.estimator(f_q, biasedness="naive"))
    g_expr = sp.expand(system.estimator(f_q, biasedness="unbiased"))

    h_np = _sympy_to_numpy_poly(h_expr, x_sym)
    g_np = _sympy_to_numpy_poly(g_expr, x_sym)

    def h_eval(x_arr: np.ndarray) -> np.ndarray:
        return np.polyval(h_np, x_arr)

    def g_eval(x_arr: np.ndarray) -> np.ndarray:
        return np.polyval(g_np, x_arr)

    return h_eval, g_eval, str(h_expr), str(g_expr)


def _compute_gaps(h_vals: np.ndarray, g_vals: np.ndarray,
                  f_x: float, B: int = 200,
                  rng: np.random.Generator = None) -> dict:
    """Compute empirical gap statistics with SEs from paired samples.

    CLT SE for means/biases; paired bootstrap SE (B resamples) for
    variances, MSEs, and gap quantities.
    """
    N = len(h_vals)
    sqrt_N = _sqrt(N)

    mean_h = float(np.mean(h_vals))
    mean_g = float(np.mean(g_vals))
    var_h  = float(np.var(h_vals, ddof=0))
    var_g  = float(np.var(g_vals, ddof=0))
    mse_h  = float(np.mean((h_vals - f_x) ** 2))
    mse_g  = float(np.mean((g_vals - f_x) ** 2))

    # CLT SEs for means (and biases, which are mean - constant)
    se_bias_h = float(np.std(h_vals, ddof=1)) / sqrt_N
    se_bias_g = float(np.std(g_vals, ddof=1)) / sqrt_N

    # Bootstrap SEs for variance/MSE/gap — single loop, paired resampling
    boot_var_h  = np.empty(B)
    boot_var_g  = np.empty(B)
    boot_mse_h  = np.empty(B)
    boot_mse_g  = np.empty(B)
    for b in range(B):
        idx    = rng.integers(0, N, size=N)
        h_b    = h_vals[idx]
        g_b    = g_vals[idx]
        boot_var_h[b] = np.var(h_b, ddof=0)
        boot_var_g[b] = np.var(g_b, ddof=0)
        boot_mse_h[b] = np.mean((h_b - f_x) ** 2)
        boot_mse_g[b] = np.mean((g_b - f_x) ** 2)

    boot_var_gap = boot_var_g - boot_var_h
    boot_mse_gap = boot_mse_g - boot_mse_h

    return {
        "bias_h":    mean_h - f_x,    "se_bias_h":  se_bias_h,
        "bias_g":    mean_g - f_x,    "se_bias_g":  se_bias_g,
        "var_h":     var_h,           "se_var_h":   float(np.std(boot_var_h, ddof=1)),
        "var_g":     var_g,           "se_var_g":   float(np.std(boot_var_g, ddof=1)),
        "mse_h":     mse_h,           "se_mse_h":   float(np.std(boot_mse_h, ddof=1)),
        "mse_g":     mse_g,           "se_mse_g":   float(np.std(boot_mse_g, ddof=1)),
        "var_gap":   var_g - var_h,   "se_var_gap": float(np.std(boot_var_gap, ddof=1)),
        "mse_gap":   mse_g - mse_h,   "se_mse_gap": float(np.std(boot_mse_gap, ddof=1)),
        "f_x":       float(f_x),
        "N":         N,
    }


def _cubic_params(coeffs):
    """Return (a, b, c) for cubic a*x³ + b*x² + c*x + d from lowest-first coeffs."""
    if len(coeffs) < 4:
        raise ValueError(f"Need ≥4 coefficients for cubic, got {len(coeffs)}")
    return float(coeffs[3]), float(coeffs[2]), float(coeffs[1])


# ══════════════════════════════════════════════════════════════════════════════
# Theoretical formulas (pure functions, no estimation)
# ══════════════════════════════════════════════════════════════════════════════

def quad_var_gap(a: float, s: float) -> float:
    """Var gap for quadratic a*x² + ...: always 0 (Laplace)."""
    return 0.0


def quad_mse_gap(a: float, s: float) -> float:
    """MSE gap for quadratic a*x² + ... under Laplace(0, s): -4*a²*s⁴."""
    return -4.0 * a**2 * s**4


def cubic_var_gap(a: float, b: float, c: float, x: float, s: float) -> float:
    """Var gap for cubic a*x³+b*x²+c*x+d under Laplace(0, s).

    = -4*s⁴ * (54*a²*s² + 18*a²*x² + 12*a*b*x + 6*a*c)
    """
    return -4.0 * s**4 * (54*a**2*s**2 + 18*a**2*x**2 + 12*a*b*x + 6*a*c)


def cubic_mse_gap(a: float, b: float, c: float, x: float, s: float) -> float:
    """MSE gap for cubic a*x³+b*x²+c*x+d under Laplace(0, s).

    = -4*s⁴ * (54*a²*s² + 27*a²*x² + 18*a*b*x + 6*a*c + b²)
    """
    return -4.0 * s**4 * (54*a**2*s**2 + 27*a**2*x**2 + 18*a*b*x + 6*a*c + b**2)


def cubic_naive_bias(a: float, b: float, x: float, s: float) -> float:
    """Bias of the naive estimator h(X)=f(X) for a cubic under Laplace(0, s).

    = 6*a*x*s² + 2*b*s²
    """
    return 6.0*a*x*s**2 + 2.0*b*s**2


def s_star(a: float, b: float, c: float) -> Optional[float]:
    """Noise threshold above which K ≥ 0 for all x (globally safe).

    Returns sqrt((b²-3ac)/(27a²)) if b²>3ac, else None (already safe at any s).
    """
    disc = b**2 - 3.0*a*c
    if disc <= 0:
        return None
    return _sqrt(disc / (27.0 * a**2))


def k_value(a: float, b: float, c: float, s: float) -> float:
    """Minimum of the inner polynomial I_MSE over x; K < 0 iff unsafe bands exist.

    K = 54*a²*s² + 6*a*c - 2*b²
    """
    return 54.0*a**2*s**2 + 6.0*a*c - 2.0*b**2


def vertex(a: float, b: float) -> float:
    """Worst-case point x* = -b/(3a) where I_MSE is minimised."""
    return -b / (3.0 * a)


def mse_band_halfwidth(a: float, b: float, c: float, s: float) -> Optional[float]:
    """Half-width of the MSE-unsafe band (naive wins in MSE) when K < 0.

    = sqrt(-K / (27*a²));  returns None when K ≥ 0.
    """
    K = k_value(a, b, c, s)
    if K >= 0:
        return None
    return _sqrt(-K / (27.0 * a**2))


def var_band_halfwidth(a: float, b: float, c: float, s: float) -> Optional[float]:
    """Half-width of the Var-unsafe band (naive wins in variance) when K < 0.

    = sqrt(-K / (18*a²));  wider than MSE band by factor sqrt(3/2) ≈ 1.2247.
    Returns None when K ≥ 0.
    """
    K = k_value(a, b, c, s)
    if K >= 0:
        return None
    return _sqrt(-K / (18.0 * a**2))


# ══════════════════════════════════════════════════════════════════════════════
# Core function
# ══════════════════════════════════════════════════════════════════════════════

def mc_gaps(coeffs, x: float, *, noise_type: str,
            s: Optional[float] = None, sigma: Optional[float] = None,
            N: int = 200_000, seed: int = 0) -> dict:
    """Empirical variance/MSE-gap statistics for one (polynomial, noise, point).

    Parameters
    ----------
    coeffs : sequence of float
        Polynomial coefficients lowest-degree-first.
    x : float
        True value of the statistic (the expansion point).
    noise_type : 'laplace' or 'gaussian'
    s : float, optional
        Laplace scale Δ/ε.  Required when noise_type='laplace'.
    sigma : float, optional
        Gaussian std-dev.  Required when noise_type='gaussian'.
        Use ``calibrateAnalyticGaussianMechanism`` to obtain this.
    N : int
        Number of Monte Carlo samples (default 200 000).
    seed : int
        Seed for the local RNG.

    Returns
    -------
    dict with keys:
        noise_type, s, sigma, noise_variance, N, x, f_x,
        bias_h, se_bias_h, bias_g, se_bias_g,
        var_h, se_var_h, var_g, se_var_g,
        mse_h, se_mse_h, mse_g, se_mse_g,
        var_gap, se_var_gap, mse_gap, se_mse_gap.

    SE method
    ---------
    Means/biases: SE = sample_std / sqrt(N)  (CLT).
    Variances, MSEs, gap quantities: paired bootstrap with B=200 resamples.
    """
    if noise_type == "laplace":
        if s is None:
            raise ValueError("s is required for noise_type='laplace'")
        noise_variance = 2.0 * s**2
    elif noise_type == "gaussian":
        if sigma is None:
            raise ValueError("sigma is required for noise_type='gaussian'")
        noise_variance = sigma**2
    else:
        raise ValueError(f"noise_type must be 'laplace' or 'gaussian', got {noise_type!r}")

    rng = np.random.default_rng(seed)
    boot_rng = np.random.default_rng(seed + 1)

    h_eval, g_eval, _, _ = _build_eval_fns(
        coeffs, noise_type=noise_type, s=s, sigma=sigma
    )
    f_x = _f_value(coeffs, x)

    if noise_type == "laplace":
        noise = rng.laplace(0.0, float(s), size=N)
    else:
        noise = rng.normal(0.0, float(sigma), size=N)

    X = float(x) + noise
    h_vals = h_eval(X)
    g_vals = g_eval(X)

    stats = _compute_gaps(h_vals, g_vals, f_x, B=200, rng=boot_rng)
    return {
        "noise_type":     noise_type,
        "s":              float(s) if s is not None else None,
        "sigma":          float(sigma) if sigma is not None else None,
        "noise_variance": noise_variance,
        "N":              N,
        "x":              float(x),
        **stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Driver functions
# ══════════════════════════════════════════════════════════════════════════════

def sweep_x(coeffs, s: float, x_grid, *, N: int = 200_000, seed: int = 0) -> dict:
    """Experiment 2: sweep x, fixed Laplace scale s.

    Builds the estimator functions once (they do not depend on x) and reuses
    them for every x value.  Each x gets its own derived seed.

    Returns
    -------
    dict with keys:
        coeffs, s, x_grid, mc_results (list of gap dicts),
        theory dict with 'var_gap' and 'mse_gap' arrays.
    """
    rng = np.random.default_rng(seed)
    x_grid = np.asarray(x_grid, dtype=float)
    child_seeds = rng.integers(0, 2**31, size=len(x_grid))

    h_eval, g_eval, _, _ = _build_eval_fns(coeffs, noise_type="laplace", s=s)

    degree = len(coeffs) - 1
    if degree == 3:
        a, b, c = _cubic_params(coeffs)
    elif degree == 2:
        a = float(coeffs[2])

    mc_results = []
    for i, x_val in enumerate(x_grid):
        local_rng  = np.random.default_rng(int(child_seeds[i]))
        boot_rng   = np.random.default_rng(int(child_seeds[i]) + 1)
        noise      = local_rng.laplace(0.0, float(s), size=N)
        X          = float(x_val) + noise
        h_vals     = h_eval(X)
        g_vals     = g_eval(X)
        f_x        = _f_value(coeffs, x_val)
        stats      = _compute_gaps(h_vals, g_vals, f_x, B=200, rng=boot_rng)
        mc_results.append({"x": float(x_val), **stats})

    if degree == 3:
        theory_var = [cubic_var_gap(a, b, c, r["x"], s) for r in mc_results]
        theory_mse = [cubic_mse_gap(a, b, c, r["x"], s) for r in mc_results]
    elif degree == 2:
        theory_var = [quad_var_gap(a, s)] * len(mc_results)
        theory_mse = [quad_mse_gap(a, s)] * len(mc_results)
    else:
        theory_var = [None] * len(mc_results)
        theory_mse = [None] * len(mc_results)

    return {
        "coeffs":     list(coeffs),
        "s":          float(s),
        "x_grid":     x_grid.tolist(),
        "mc_results": mc_results,
        "theory":     {"var_gap": theory_var, "mse_gap": theory_mse},
    }


def sweep_s_at_vertex(coeffs, s_grid, *, N: int = 200_000, seed: int = 0) -> dict:
    """Experiment 3a: sweep s, evaluate at the vertex x* = -b/(3a).

    Shows the MSE gap at the worst-case point crossing zero at s = s_star.
    Rebuilds the estimator for each s (Laplace formula changes with scale).

    Returns
    -------
    dict with keys:
        coeffs, s_grid, x_star, s_star_val, mc_results,
        theory dict with 'mse_gap' array.
    """
    a, b, c = _cubic_params(coeffs)
    x_star = vertex(a, b)
    s_star_val = s_star(a, b, c)

    rng = np.random.default_rng(seed)
    s_grid = np.asarray(s_grid, dtype=float)
    child_seeds = rng.integers(0, 2**31, size=len(s_grid))

    mc_results = []
    theory_mse = []
    for i, s_val in enumerate(s_grid):
        h_eval, g_eval, _, _ = _build_eval_fns(
            coeffs, noise_type="laplace", s=float(s_val)
        )
        local_rng = np.random.default_rng(int(child_seeds[i]))
        boot_rng  = np.random.default_rng(int(child_seeds[i]) + 1)
        noise     = local_rng.laplace(0.0, float(s_val), size=N)
        X         = float(x_star) + noise
        h_vals    = h_eval(X)
        g_vals    = g_eval(X)
        f_x       = _f_value(coeffs, x_star)
        stats     = _compute_gaps(h_vals, g_vals, f_x, B=200, rng=boot_rng)
        mc_results.append({"s": float(s_val), **stats})
        theory_mse.append(cubic_mse_gap(a, b, c, x_star, float(s_val)))

    return {
        "coeffs":    list(coeffs),
        "s_grid":    s_grid.tolist(),
        "x_star":    x_star,
        "s_star_val": s_star_val,
        "mc_results": mc_results,
        "theory":    {"mse_gap": theory_mse},
    }


def measure_band(coeffs, s: float, x_grid, *, N: int = 200_000, seed: int = 0) -> dict:
    """Experiment 3b: measure unsafe bands for a cubic with s < s_star.

    Raises ValueError if s >= s_star (no unsafe band exists).

    Returns
    -------
    dict with keys:
        coeffs, s, x_grid, mc_results, theory (var_gap/mse_gap + band edges),
        measured (unsafe x lists, widths, ratio var_width/mse_width ≈ sqrt(3/2)).
    """
    a, b, c = _cubic_params(coeffs)
    ss = s_star(a, b, c)
    if ss is None or float(s) >= ss:
        raise ValueError(
            f"s={s} must be strictly less than s_star={ss} for an unsafe band to exist."
        )

    x_grid = np.asarray(x_grid, dtype=float)
    sweep = sweep_x(coeffs, s, x_grid, N=N, seed=seed)
    mc_results = sweep["mc_results"]

    x_star_val = vertex(a, b)
    mse_hw = mse_band_halfwidth(a, b, c, s)
    var_hw = var_band_halfwidth(a, b, c, s)

    # Measured unsafe intervals (where empirical gap > 0)
    var_unsafe_x = [r["x"] for r in mc_results if r["var_gap"] > 0]
    mse_unsafe_x = [r["x"] for r in mc_results if r["mse_gap"] > 0]

    def _width(xs):
        return (max(xs) - min(xs)) if len(xs) >= 2 else 0.0

    var_band_w = _width(var_unsafe_x)
    mse_band_w = _width(mse_unsafe_x)
    ratio = (var_band_w / mse_band_w) if mse_band_w > 0 else float("nan")

    return {
        "coeffs":     list(coeffs),
        "s":          float(s),
        "x_grid":     x_grid.tolist(),
        "mc_results": mc_results,
        "theory": {
            "var_gap":        sweep["theory"]["var_gap"],
            "mse_gap":        sweep["theory"]["mse_gap"],
            "x_star":         x_star_val,
            "var_band_edges": (x_star_val - var_hw, x_star_val + var_hw),
            "mse_band_edges": (x_star_val - mse_hw, x_star_val + mse_hw),
            "var_hw":         var_hw,
            "mse_hw":         mse_hw,
        },
        "measured": {
            "var_unsafe_x":  var_unsafe_x,
            "mse_unsafe_x":  mse_unsafe_x,
            "var_band_width": var_band_w,
            "mse_band_width": mse_band_w,
            "width_ratio":   ratio,          # expect ≈ sqrt(3/2) ≈ 1.2247
        },
    }


def compare_noise(coeffs, x: float, *, epsilon: float, delta: float, Delta: float,
                  N: int = 200_000, seed: int = 0) -> dict:
    """Experiment 4: matched-privacy Laplace vs Gaussian comparison.

    Calibrates Gaussian sigma via the analytic Balle–Wang mechanism.
    Intended for degree 2, 3, 4 polynomials at fixed (epsilon, delta, Delta).

    delta=1e-10 approximates pure DP (note in most use cases).

    Returns
    -------
    dict with keys:
        coeffs, x, epsilon, delta, Delta, s, sigma,
        laplace_variance (=2s²), gaussian_variance (=sigma²),
        laplace (mc_gaps result), gaussian (mc_gaps result),
        estimator_exprs dict with str forms of all four estimators.
    """
    s_val = Delta / epsilon
    sigma_val = calibrateAnalyticGaussianMechanism(
        float(epsilon), float(delta), float(Delta)
    )

    # Build estimator strings (for inspection / display)
    _, _, h_str_lap, g_str_lap = _build_eval_fns(
        coeffs, noise_type="laplace", s=s_val
    )
    _, _, h_str_gau, g_str_gau = _build_eval_fns(
        coeffs, noise_type="gaussian", sigma=sigma_val
    )

    res_lap = mc_gaps(
        coeffs, x, noise_type="laplace", s=s_val, N=N, seed=seed
    )
    res_gau = mc_gaps(
        coeffs, x, noise_type="gaussian", sigma=sigma_val, N=N, seed=seed + 1000
    )

    return {
        "coeffs":           list(coeffs),
        "x":                float(x),
        "epsilon":          float(epsilon),
        "delta":            float(delta),
        "Delta":            float(Delta),
        "s":                float(s_val),
        "sigma":            float(sigma_val),
        "laplace_variance": 2.0 * s_val**2,
        "gaussian_variance": sigma_val**2,
        "laplace":          res_lap,
        "gaussian":         res_gau,
        "estimator_exprs": {
            "laplace_naive":     h_str_lap,
            "laplace_unbiased":  g_str_lap,
            "gaussian_naive":    h_str_gau,
            "gaussian_unbiased": g_str_gau,
        },
    }


def sweep_epsilon_noise(coeffs, x: float, epsilon_grid, *, delta: float,
                        Delta: float, N: int = 200_000, seed: int = 0) -> dict:
    """Experiment 4 (sweep): unbiased estimator variance/MSE vs epsilon for both noise types.

    For each epsilon, computes Laplace s = Delta/epsilon and calibrates
    Gaussian sigma, then runs mc_gaps for both.

    delta defaults suggest using 1e-10 (approximates pure DP).

    Returns
    -------
    dict with convenience arrays ready for plotting:
        epsilon_vals, laplace_s, gaussian_sigma,
        laplace_unbiased_var, laplace_unbiased_mse,
        gaussian_unbiased_var, gaussian_unbiased_mse,
        se_* counterparts, and the full comparisons list.
    """
    epsilon_grid = np.asarray(epsilon_grid, dtype=float)
    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**31, size=len(epsilon_grid))

    comparisons = []
    for i, eps in enumerate(epsilon_grid):
        cmp = compare_noise(
            coeffs, x,
            epsilon=float(eps), delta=delta, Delta=Delta,
            N=N, seed=int(child_seeds[i]),
        )
        comparisons.append(cmp)

    def _col(key, sub=None):
        if sub:
            return [c[sub][key] for c in comparisons]
        return [c[key] for c in comparisons]

    return {
        "coeffs":       list(coeffs),
        "x":            float(x),
        "delta":        float(delta),
        "Delta":        float(Delta),
        "epsilon_vals": epsilon_grid.tolist(),
        "laplace_s":    _col("s"),
        "gaussian_sigma": _col("sigma"),
        "laplace_unbiased_var":    _col("var_g", "laplace"),
        "laplace_unbiased_mse":    _col("mse_g", "laplace"),
        "gaussian_unbiased_var":   _col("var_g", "gaussian"),
        "gaussian_unbiased_mse":   _col("mse_g", "gaussian"),
        "se_laplace_unbiased_var":  _col("se_var_g", "laplace"),
        "se_laplace_unbiased_mse":  _col("se_mse_g", "laplace"),
        "se_gaussian_unbiased_var": _col("se_var_g", "gaussian"),
        "se_gaussian_unbiased_mse": _col("se_mse_g", "gaussian"),
        "comparisons":  comparisons,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_figures_dir() -> Path:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return _FIGURES_DIR


def _save_figure(fig: plt.Figure, stem: str) -> None:
    """Save figure as both .pdf and .png to figures/."""
    d = _ensure_figures_dir()
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{ext}", dpi=180, bbox_inches="tight")


def plot_gaps_vs_x(sweep_x_result: dict) -> plt.Figure:
    """Scatter empirical var_gap and mse_gap vs x with theoretical overlays.

    Saves to figures/gaps_vs_x.pdf and returns the Figure.
    """
    res = sweep_x_result
    xs  = [r["x"]       for r in res["mc_results"]]
    vg  = [r["var_gap"]  for r in res["mc_results"]]
    mg  = [r["mse_gap"]  for r in res["mc_results"]]
    sve = [r["se_var_gap"] for r in res["mc_results"]]
    sme = [r["se_mse_gap"] for r in res["mc_results"]]
    tv  = res["theory"]["var_gap"]
    tm  = res["theory"]["mse_gap"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, emp, se, th, ylabel, title in zip(
        axes,
        [vg, mg], [sve, sme], [tv, tm],
        ["Var gap  Var(g) − Var(h)", "MSE gap  MSE(g) − MSE(h)"],
        ["Variance gap", "MSE gap"],
    ):
        ax.errorbar(xs, emp, yerr=se, fmt="o", ms=3, lw=0.8,
                    color="tab:blue", label="Empirical ± SE", zorder=3)
        if th[0] is not None:
            ax.plot(xs, th, lw=_LW, color="tab:orange", label="Theory")
        ax.axhline(0, lw=1, ls="--", color="black")
        ax.set_xlabel("x", fontsize=_FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
        ax.set_title(title, fontsize=_FS_LABEL)
        ax.legend(fontsize=_FS_LEGEND)
        ax.grid(alpha=_GRID_ALPHA)

    fig.tight_layout()
    _save_figure(fig, "gaps_vs_x")
    return fig


def plot_gap_vs_s_at_vertex(sweep_s_result: dict) -> plt.Figure:
    """Empirical MSE gap at x* vs s, with theoretical curve and s* marker.

    Saves to figures/gap_vs_s_at_vertex.pdf.
    """
    res   = sweep_s_result
    ss    = [r["s"]       for r in res["mc_results"]]
    mg    = [r["mse_gap"]  for r in res["mc_results"]]
    sme   = [r["se_mse_gap"] for r in res["mc_results"]]
    tm    = res["theory"]["mse_gap"]
    x_star_val = res["x_star"]
    s_star_val = res["s_star_val"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ss, mg, yerr=sme, fmt="o", ms=3, lw=0.8,
                color="tab:blue", label="Empirical ± SE", zorder=3)
    ax.plot(ss, tm, lw=_LW, color="tab:orange", label="Theory")
    ax.axhline(0, lw=1, ls="--", color="black")
    if s_star_val is not None:
        ax.axvline(s_star_val, lw=1.5, ls=":", color="tab:red",
                   label=f"$s^*$ ≈ {s_star_val:.4f}")
    ax.set_xlabel("Noise scale $s = \\Delta/\\varepsilon$", fontsize=_FS_LABEL)
    ax.set_ylabel("MSE gap at $x^*$", fontsize=_FS_LABEL)
    ax.set_title(f"MSE gap at vertex $x^* = -b/3a = {x_star_val:.3g}$", fontsize=_FS_LABEL)
    ax.legend(fontsize=_FS_LEGEND)
    ax.grid(alpha=_GRID_ALPHA)
    fig.tight_layout()
    _save_figure(fig, "gap_vs_s_at_vertex")
    return fig


def plot_unsafe_bands(measure_band_result: dict) -> plt.Figure:
    """Gap vs x with shaded measured unsafe bands and theoretical band edges.

    Two subplots: variance gap (left) and MSE gap (right).
    Saves to figures/unsafe_bands.pdf.
    """
    res  = measure_band_result
    xs   = [r["x"]         for r in res["mc_results"]]
    vg   = [r["var_gap"]    for r in res["mc_results"]]
    mg   = [r["mse_gap"]    for r in res["mc_results"]]
    sve  = [r["se_var_gap"] for r in res["mc_results"]]
    sme  = [r["se_mse_gap"] for r in res["mc_results"]]
    tv   = res["theory"]["var_gap"]
    tm   = res["theory"]["mse_gap"]
    meas = res["measured"]
    th   = res["theory"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    configs = [
        (vg, sve, tv, th["var_band_edges"],  meas["var_unsafe_x"],
         "Var gap  Var(g) − Var(h)", "Variance gap"),
        (mg, sme, tm, th["mse_band_edges"],  meas["mse_unsafe_x"],
         "MSE gap  MSE(g) − MSE(h)", "MSE gap"),
    ]
    for ax, (emp, se, theory, band_edges, unsafe_xs, ylabel, title) in zip(axes, configs):
        ax.errorbar(xs, emp, yerr=se, fmt="o", ms=3, lw=0.8,
                    color="tab:blue", label="Empirical ± SE", zorder=3)
        if theory[0] is not None:
            ax.plot(xs, theory, lw=_LW, color="tab:orange", label="Theory")
        ax.axhline(0, lw=1, ls="--", color="black")
        # Shade measured unsafe band
        if unsafe_xs:
            ax.axvspan(min(unsafe_xs), max(unsafe_xs), alpha=0.12,
                       color="tab:red", label="Measured unsafe")
        # Theoretical band edges
        ax.axvline(band_edges[0], lw=1.2, ls=":", color="tab:green",
                   label="Theory band edges")
        ax.axvline(band_edges[1], lw=1.2, ls=":", color="tab:green")
        ax.set_xlabel("x", fontsize=_FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
        ax.set_title(title, fontsize=_FS_LABEL)
        ax.legend(fontsize=_FS_LEGEND)
        ax.grid(alpha=_GRID_ALPHA)

    fig.tight_layout()
    _save_figure(fig, "unsafe_bands")
    return fig


def plot_noise_comparison(sweep_epsilon_result: dict) -> plt.Figure:
    """Unbiased estimator variance and MSE vs ε: Laplace vs Gaussian.

    Two subplots (variance, MSE).  Error bands from ± SE.
    Saves to figures/noise_comparison.pdf.
    """
    res  = sweep_epsilon_result
    eps  = res["epsilon_vals"]
    lv   = res["laplace_unbiased_var"]
    gv   = res["gaussian_unbiased_var"]
    lm   = res["laplace_unbiased_mse"]
    gm   = res["gaussian_unbiased_mse"]
    lv_se = res["se_laplace_unbiased_var"]
    gv_se = res["se_gaussian_unbiased_var"]
    lm_se = res["se_laplace_unbiased_mse"]
    gm_se = res["se_gaussian_unbiased_mse"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, lap, gau, lap_se, gau_se, ylabel in zip(
        axes,
        [lv, lm], [gv, gm],
        [lv_se, lm_se], [gv_se, gm_se],
        ["Unbiased variance", "Unbiased MSE"],
    ):
        eps_arr = np.asarray(eps)
        lap_arr = np.asarray(lap)
        gau_arr = np.asarray(gau)
        lap_se_arr = np.asarray(lap_se)
        gau_se_arr = np.asarray(gau_se)

        ax.plot(eps_arr, lap_arr, lw=_LW, color="tab:blue", label="Laplace")
        ax.fill_between(eps_arr, lap_arr - lap_se_arr, lap_arr + lap_se_arr,
                        alpha=0.20, color="tab:blue")
        ax.plot(eps_arr, gau_arr, lw=_LW, color="tab:orange",
                ls="--", label="Gaussian")
        ax.fill_between(eps_arr, gau_arr - gau_se_arr, gau_arr + gau_se_arr,
                        alpha=0.20, color="tab:orange")
        ax.set_xlabel("Privacy parameter $\\varepsilon$", fontsize=_FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
        ax.legend(fontsize=_FS_LEGEND)
        ax.grid(alpha=_GRID_ALPHA)

    fig.tight_layout()
    _save_figure(fig, "noise_comparison")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Shared axis helper
# ══════════════════════════════════════════════════════════════════════════════

def _add_epsilon_axis(ax: plt.Axes, Delta: float = 1.0) -> None:
    """Add a secondary top x-axis showing ε = Δ/s when the primary axis is s."""
    def _s_to_eps(s):
        s = np.asarray(s, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s != 0, Delta / s, np.inf)

    def _eps_to_s(e):
        e = np.asarray(e, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(e != 0, Delta / e, np.inf)

    sec = ax.secondary_xaxis("top", functions=(_s_to_eps, _eps_to_s))
    sec.set_xlabel(f"Privacy $\\varepsilon$  ($\\Delta={Delta}$)", fontsize=_FS_LABEL)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment Q — Exhaustive quadratic validation
# ══════════════════════════════════════════════════════════════════════════════

def experiment_Q(
    a_grid=None, b_grid=None, c_grid=None, x_grid=None, s_grid=None,
    N: int = 50_000, seed: int = 0,
) -> dict:
    """Exhaustive sweep over quadratic parameters for MSE-gap/Var-gap validation.

    For quadratics the unbiased estimator is g(X) = h(X) − 2·a·s², so
    evaluators are computed analytically (no EstimatorSystem call per config).

    Returns
    -------
    dict with keys: records (list of per-config dicts), aggregate arrays
    resid_mse, resid_var, n_se_mse, max_abs_resid_mse, max_n_se_mse, frac_pass_3se.
    """
    if a_grid is None:
        a_grid = [v for v in range(-5, 6) if v != 0]   # {-5,...,5}\{0}
    if b_grid is None:
        b_grid = [-5.0, -4.0, -2.0, 0.0, 2.0, 4.0, 5.0]
    if c_grid is None:
        c_grid = [-5.0, -4.0, -2.0, 0.0, 2.0, 4.0, 5.0]
    if x_grid is None:
        x_grid = np.linspace(-10.0, 10.0, 17)
    if s_grid is None:
        s_grid = [0.1, 0.3, 0.5, 0.75, 1.0, 1.5]

    a_arr = np.asarray(a_grid, dtype=float)
    b_arr = np.asarray(b_grid, dtype=float)
    c_arr = np.asarray(c_grid, dtype=float)
    x_arr = np.asarray(x_grid, dtype=float)
    s_arr = np.asarray(s_grid, dtype=float)

    rng     = np.random.default_rng(seed)
    records = []

    for a_val in a_arr:
        for s_val in s_arr:
            noise = rng.laplace(0.0, float(s_val), size=N)
            for b_val in b_arr:
                for c_val in c_arr:
                    for x_val in x_arr:
                        X      = float(x_val) + noise
                        h_vals = a_val * X**2 + b_val * X + c_val
                        g_vals = h_vals - 2.0 * a_val * s_val**2
                        f_x    = a_val * x_val**2 + b_val * x_val + c_val

                        Z           = (g_vals - f_x)**2 - (h_vals - f_x)**2
                        emp_mse_gap = float(np.mean(Z))
                        se_mse_gap  = float(np.std(Z, ddof=1) / _sqrt(N))

                        emp_var_gap  = float(
                            np.var(g_vals, ddof=0) - np.var(h_vals, ddof=0)
                        )
                        theo_mse_gap = quad_mse_gap(a_val, s_val)
                        theo_var_gap = quad_var_gap(a_val, s_val)  # always 0

                        resid_mse = emp_mse_gap - theo_mse_gap
                        resid_var = emp_var_gap - theo_var_gap

                        records.append({
                            "a": float(a_val), "b": float(b_val),
                            "c": float(c_val), "x": float(x_val),
                            "s": float(s_val),
                            "a2s4":         float(a_val**2 * s_val**4),
                            "emp_mse_gap":  emp_mse_gap,
                            "se_mse_gap":   se_mse_gap,
                            "emp_var_gap":  emp_var_gap,
                            "theo_mse_gap": theo_mse_gap,
                            "theo_var_gap": theo_var_gap,
                            "resid_mse":    resid_mse,
                            "resid_var":    resid_var,
                            "n_se_mse": (
                                abs(resid_mse) / se_mse_gap if se_mse_gap > 0 else 0.0
                            ),
                        })

    resid_mse_arr = np.array([r["resid_mse"]    for r in records])
    resid_var_arr = np.array([r["resid_var"]    for r in records])
    n_se_arr      = np.array([r["n_se_mse"]     for r in records])
    emp_mse_arr   = np.array([r["emp_mse_gap"]  for r in records])

    n_mse_neg    = int(np.sum(emp_mse_arr < 0))
    n_mse_nonneg = int(np.sum(emp_mse_arr >= 0))
    # "Significant" positive: emp_mse_gap > 0 AND emp/se > 2 (z > 2 from zero)
    se_arr_full  = np.array([r["se_mse_gap"] for r in records])
    z_from_zero  = np.where(se_arr_full > 0, emp_mse_arr / se_arr_full, 0.0)
    n_signif_pos = int(np.sum((emp_mse_arr > 0) & (z_from_zero > 2.0)))

    return {
        "records":           records,
        "a_grid":            a_arr.tolist(),
        "b_grid":            b_arr.tolist(),
        "c_grid":            c_arr.tolist(),
        "x_grid":            x_arr.tolist(),
        "s_grid":            s_arr.tolist(),
        "N":                 N,
        "resid_mse":         resid_mse_arr,
        "resid_var":         resid_var_arr,
        "n_se_mse":          n_se_arr,
        "emp_mse":           emp_mse_arr,
        "max_abs_resid_mse": float(np.max(np.abs(resid_mse_arr))),
        "max_abs_resid_var": float(np.max(np.abs(resid_var_arr))),
        "max_n_se_mse":      float(np.max(n_se_arr)),
        "frac_pass_3se":     float(np.mean(n_se_arr <= 3.0)),
        "n_mse_neg":         n_mse_neg,
        "n_mse_nonneg":      n_mse_nonneg,
        "n_signif_pos":      n_signif_pos,   # empirically positive AND z > 2 from 0
    }


def plot_Q_residuals(exp_Q_result: dict, Delta: float = 1.0) -> plt.Figure:
    """Four-panel standardised MSE-gap residual (z-score) vs each of a, b, c, x.

    Grid used: a ∈ {-5,...,5}\\{0}, b ∈ {-5,-4,-2,0,2,4,5}, c ∈ {-5,-4,-2,0,2,4,5},
    x ∈ linspace(-10, 10, 17), s ∈ {0.1, 0.3, 0.5, 0.75, 1.0, 1.5}.

    z-score = (empirical MSE-gap − theory) / SE_MSE-gap.
    Reference lines at 0 and ±3 are drawn.  A horizontal band centred on 0
    with spread ≈ 1 and essentially all points within ±3 confirms that the
    formula −4a²s⁴ is correct.

    Centred-on-zero with no tilt in the b, c, x panels confirms the
    MSE-gap is independent of b, c, x, as the theory −4a²s⁴ predicts.

    Saves to figures/Q_residuals.{pdf,png}.
    """
    records  = exp_Q_result["records"]
    frac_3se = exp_Q_result["frac_pass_3se"]

    params = ["a", "b", "c", "x"]
    labels = ["$a$", "$b$", "$c$", "$x$"]

    # z-score: (emp - theory) / se; cap |se| > 0 to avoid divide-by-zero
    z_scores = np.array([
        r["resid_mse"] / r["se_mse_gap"] if r["se_mse_gap"] > 0 else 0.0
        for r in records
    ])

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, par, lab in zip(axes, params, labels):
        xs_p = np.array([r[par] for r in records])
        ax.scatter(xs_p, z_scores, s=2, alpha=0.25, color="tab:blue", rasterized=True)
        ax.axhline(0,   lw=1.2, ls="--", color="black")
        ax.axhline( 3,  lw=0.8, ls=":",  color="tab:red",  label="$\\pm 3$")
        ax.axhline(-3,  lw=0.8, ls=":",  color="tab:red")
        ax.set_xlabel(lab, fontsize=_FS_LABEL)
        ax.grid(alpha=_GRID_ALPHA)
    axes[0].set_ylabel("Standardised residual  $(\\hat{g} - g_0)$ / SE",
                        fontsize=_FS_LABEL)
    axes[0].legend(fontsize=_FS_LEGEND)

    fig.suptitle(
        f"Exp. Q — Standardised MSE-gap residuals over quadratics  "
        f"({frac_3se*100:.1f}% within $\\pm 3$ SE,  "
        f"max = {exp_Q_result['max_n_se_mse']:.2f} SE,  "
        f"$\\Delta={Delta}$)\n"
        "No tilt in $b$, $c$, $x$ panels $\\Rightarrow$ "
        "MSE-gap independent of $b$, $c$, $x$ (confirms theory $-4a^2s^4$)",
        fontsize=9,
    )
    fig.tight_layout()
    _save_figure(fig, "Q_residuals")
    return fig


def plot_Q_var_residuals(exp_Q_result: dict, Delta: float = 1.0) -> plt.Figure:
    """Var-gap residual: should be ≡ 0 for all quadratic configs.

    Saves to figures/Q_var_residuals.{pdf,png}.
    """
    records = exp_Q_result["records"]
    ys_all  = [r["resid_var"] for r in records]
    max_r   = exp_Q_result["max_abs_resid_var"]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(range(len(records)), ys_all, s=1, alpha=0.4,
               color="tab:green", rasterized=True)
    ax.axhline(0, lw=1, ls="--", color="black")
    ax.set_xlabel("Configuration index", fontsize=_FS_LABEL)
    ax.set_ylabel("Var-gap residual (emp $-$ 0)", fontsize=_FS_LABEL)
    ax.set_title(
        f"Exp. Q — Var-gap residuals (identically 0 for quadratics)  "
        f"max|res| = {max_r:.2e}  ($\\Delta={Delta}$)",
        fontsize=9,
    )
    ax.grid(alpha=_GRID_ALPHA)
    fig.tight_layout()
    _save_figure(fig, "Q_var_residuals")
    return fig


def plot_Q_loglog(
    s_grid=None, a_val: float = 1.0, N: int = 50_000,
    seed: int = 42, Delta: float = 1.0,
) -> plt.Figure:
    """Log-log |MSE-gap| vs s for a = 1 quadratic; annotates fitted slope.

    Expected slope = 4 (MSE-gap ∝ s⁴).
    Saves to figures/Q_loglog.{pdf,png}.
    """
    if s_grid is None:
        s_grid = np.logspace(-1.0, 0.5, 30)

    s_arr    = np.asarray(s_grid, dtype=float)
    rng      = np.random.default_rng(seed)
    emp_gaps = []

    for s_val in s_arr:
        noise  = rng.laplace(0.0, float(s_val), size=N)
        X      = noise                           # x = 0, so X = 0 + noise
        h_vals = a_val * X**2
        g_vals = h_vals - 2.0 * a_val * s_val**2
        Z      = g_vals**2 - h_vals**2           # f_x = 0
        emp_gaps.append(float(np.mean(Z)))

    theo_gaps = np.array([quad_mse_gap(a_val, s) for s in s_arr])
    emp_arr   = np.array(emp_gaps)

    # Fit slope in log-log space using theory (exact, no noise)
    log_s  = np.log10(s_arr)
    log_th = np.log10(np.abs(theo_gaps))
    slope, intercept = np.polyfit(log_s, log_th, 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(s_arr, np.abs(theo_gaps), lw=_LW, color="tab:orange",
              label=f"Theory $4a^2s^4$")
    ax.loglog(s_arr, np.abs(emp_arr),  "o", ms=4, color="tab:blue",
              label="Empirical |MSE-gap|")

    s_fit   = np.array([s_arr[0], s_arr[-1]])
    gap_fit = 10 ** (intercept + slope * np.log10(s_fit))
    ax.loglog(s_fit, gap_fit, ls=":", lw=1.5, color="tab:red",
              label=f"Fitted slope = {slope:.2f}  (expect 4)")

    ax.set_xlabel(f"Noise scale $s$  ($\\Delta={Delta}$)", fontsize=_FS_LABEL)
    ax.set_ylabel("|MSE gap|", fontsize=_FS_LABEL)
    ax.set_title(
        f"Exp. Q — Log-log scaling: |MSE-gap| $\\propto s^4$  "
        f"($a={a_val}$, $x=0$, $\\Delta={Delta}$)",
        fontsize=9,
    )
    ax.legend(fontsize=_FS_LEGEND)
    ax.grid(alpha=_GRID_ALPHA, which="both")
    _add_epsilon_axis(ax, Delta=Delta)
    fig.tight_layout()
    _save_figure(fig, "Q_loglog")
    return fig


def plot_Q_collapse(exp_Q_result: dict, Delta: float = 1.0) -> plt.Figure:
    """Scatter |MSE-gap| vs a²s⁴ on log-log axes for all quadratic configs.

    Log-log axes are used because a²s⁴ spans more than 3 orders of magnitude
    across the grid (a∈{±1,…,±5}, s∈{0.3,…,1.0}).  All points are expected
    to collapse onto the line y = 4·(a²s⁴) regardless of b, c, or x, since
    MSE-gap = −4a²s⁴ is independent of those parameters.

    Prints a report of the number of configs where MSE-gap ≥ 0 (expected: 0,
    or at most a handful explainable by MC noise near a²s⁴ ≈ 0).
    Saves to figures/Q_collapse.{pdf,png}.
    """
    records     = exp_Q_result["records"]
    n_total     = len(records)
    n_neg       = exp_Q_result["n_mse_neg"]
    n_nonneg    = exp_Q_result["n_mse_nonneg"]
    n_signif    = exp_Q_result["n_signif_pos"]

    print(f"\n  Exp. Q collapse report:")
    print(f"    MSE-gap < 0  (unbiased wins) : {n_neg}/{n_total}  ({n_neg/n_total*100:.2f}%)")
    print(f"    MSE-gap ≥ 0 (sign flipped)   : {n_nonneg}/{n_total}  "
          f"— of these, {n_signif} are significant (z > 2 from 0)")
    if n_signif > 0:
        print(f"    *** {n_signif} GENUINE violation(s) where emp > 0 with z > 2 ***")
        for r in records:
            z = r["emp_mse_gap"] / r["se_mse_gap"] if r["se_mse_gap"] > 0 else 0.0
            if r["emp_mse_gap"] > 0 and z > 2.0:
                print(f"      a={r['a']:+.0f} b={r['b']:+.1f} c={r['c']:+.1f} "
                      f"x={r['x']:+.2f} s={r['s']:.2f}  "
                      f"gap={r['emp_mse_gap']:.3g}  SE={r['se_mse_gap']:.3g}  "
                      f"(z={z:.1f} from zero)")
    else:
        print(f"    No genuine violations — all sign-flips have z ≤ 2 "
              f"(consistent with MC noise near a²s⁴ ≈ 0).")

    s_vals   = sorted(set(r["s"] for r in records))
    cmap     = plt.cm.viridis
    colors   = [cmap(0.15 + 0.70 * i / max(len(s_vals) - 1, 1))
                for i in range(len(s_vals))]
    s_to_col = dict(zip(s_vals, colors))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    for s_val, col in zip(s_vals, colors):
        sub    = [r for r in records if r["s"] == s_val]
        x_pts  = np.array([r["a2s4"]              for r in sub])
        y_pts  = np.array([abs(r["emp_mse_gap"])   for r in sub])
        mask   = (x_pts > 0) & (y_pts > 0)
        ax.scatter(x_pts[mask], y_pts[mask], s=8, alpha=0.45,
                   color=col, label=f"$s={s_val}$  ($\\varepsilon={Delta/s_val:.2g}$)", rasterized=True)

    all_a2s4 = np.array([r["a2s4"] for r in records if r["a2s4"] > 0])
    x_line   = np.array([all_a2s4.min(), all_a2s4.max()])
    ax.plot(x_line, 4.0 * x_line, lw=_LW, color="tab:red",
            label=r"Theory $|\mathrm{MSE-gap}| = 4a^2(\Delta/\varepsilon)^4$")

    frac_str = f"{n_neg}/{n_total} = {n_neg/n_total*100:.1f}\\%"
    ax.set_xlabel(r"$a^2(\Delta/\varepsilon)^4$", fontsize=_FS_LABEL)
    ax.set_ylabel(r"$|\mathrm{MSE\text{-}gap}|$  $=$ $|\mathrm{MSE}(g) - \mathrm{MSE}(h)|$",
                  fontsize=_FS_LABEL)
    # "Exp. Q — MSE-gap collapses onto $-4a^2(\\Delta / \\epsilon)^4$  (all quadratics, all $b,c,x$)\n"
    ax.set_title(
        f"Fraction with MSE($g$) $<$ MSE($h$): {frac_str}",
        fontsize=9,
    )
    ax.legend(fontsize=_FS_LEGEND, loc="upper left")
    ax.grid(alpha=_GRID_ALPHA, which="both")
    fig.tight_layout()
    _save_figure(fig, "Q_collapse")
    return fig


def plot_Q_gap_vs_a(exp_Q_result: dict, Delta: float = 1.0) -> plt.Figure:
    """Signed empirical MSE-gap vs a; theory parabola −4a²s⁴ overlaid per s.

    Since MSE-gap = −4a²s⁴ does not depend on b, c, or x, all configs with
    the same (a, s) collapse to the same theoretical value.  Points are
    jittered slightly on the x-axis for visibility.
    Saves to figures/Q_gap_vs_a.{pdf,png}.
    """
    records = exp_Q_result["records"]
    s_vals  = sorted(set(r["s"] for r in records))
    cmap    = plt.cm.viridis
    colors  = [cmap(0.15 + 0.70 * i / max(len(s_vals) - 1, 1))
               for i in range(len(s_vals))]

    rng_jit = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    for s_val, col in zip(s_vals, colors):
        sub    = [r for r in records if r["s"] == s_val]
        a_pts  = np.array([r["a"]           for r in sub])
        mg_pts = np.array([r["emp_mse_gap"]  for r in sub])
        jitter = rng_jit.uniform(-0.08, 0.08, size=len(a_pts))
        ax.scatter(a_pts + jitter, mg_pts, s=4, alpha=0.25,
                   color=col, rasterized=True)

        a_fine = np.linspace(a_pts.min() - 0.5, a_pts.max() + 0.5, 300)
        ax.plot(a_fine, -4.0 * a_fine**2 * s_val**4, lw=_LW,
                color=col, label=f"$s={s_val}$  ($\\varepsilon={Delta/s_val:.2g}$)")

    ax.axhline(0, lw=1, ls="--", color="black")
    ax.set_xlabel("$a$", fontsize=_FS_LABEL)
    ax.set_ylabel("MSE gap  $\\mathrm{MSE}(g) - \\mathrm{MSE}(h)$", fontsize=_FS_LABEL)
    ax.set_title(
        # f"Exp. Q — Signed MSE-gap vs $a$;  theory $-4a^2(\\Delta / \\varepsilon)^4$ overlaid (one curve per $s = (\Delta / \varepsilon)$)\n"
        f"All points expected strictly below zero",
        fontsize=9,
    )
    ax.legend(fontsize=_FS_LEGEND, title="$s$ (theory curves)", title_fontsize=7)
    ax.grid(alpha=_GRID_ALPHA)
    fig.tight_layout()
    _save_figure(fig, "Q_gap_vs_a")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Experiment S — MSE-safe = Var-safe coincidence
# ══════════════════════════════════════════════════════════════════════════════

def plot_S_coincidence(
    coeffs=None, s_grid=None, x_fine=None, Delta: float = 1.0,
) -> plt.Figure:
    """Show min_x I_Var and min_x I_MSE both equal K, crossing zero at s*.

    Three curves (K analytical, numerical min of I_Var, numerical min of I_MSE)
    vs s on the same axes — they coincide by construction.
    Saves to figures/S_coincidence.{pdf,png}.
    """
    # try with different cubuc, 25*q**3 + 7*q**2 + 2*q + 1

    if coeffs is None:
    #    coeffs = [1.0, 2.0, 7.0, 25.0]   # f = 25x³ + 7x² + 2x + 1
        coeffs = [0.0, -3.0, 0.0, 1.0]   # f = x³ − 3x
    #    coeffs = [0.0, -3.0, 1.0, 3.0]   # f = 3x³ + x² − 3x
    if s_grid is None:
        s_grid = np.linspace(0.05, 1.5, 300)
    if x_fine is None:
        x_fine = np.linspace(-10.0, 10.0, 4000)

    a, b, c    = _cubic_params(coeffs)
    ss         = s_star(a, b, c)
    s_arr      = np.asarray(s_grid,  dtype=float)
    x_arr      = np.asarray(x_fine,  dtype=float)

    K_vals    = np.array([k_value(a, b, c, s) for s in s_arr])
    min_I_Var = np.array([
        np.min(54*a**2*s**2 + 18*a**2*x_arr**2 + 12*a*b*x_arr + 6*a*c)
        for s in s_arr
    ])
    min_I_MSE = np.array([
        np.min(54*a**2*s**2 + 27*a**2*x_arr**2 + 18*a*b*x_arr + 6*a*c + b**2)
        for s in s_arr
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_arr, K_vals,    lw=_LW + 1, color="tab:orange",
            label=r"$K(s)=54a^2s^2+6ac-2b^2$  (analytical)")
    ax.plot(s_arr, min_I_Var, lw=_LW, ls="--", color="tab:blue",
            label=r"$\min_x I_{\mathrm{Var}}(x)$  (numerical)")
    ax.plot(s_arr, min_I_MSE, lw=_LW, ls=":",  color="tab:green",
            label=r"$\min_x I_{\mathrm{MSE}}(x)$  (numerical)")
    ax.axhline(0, lw=1, ls="--", color="black")
    if ss is not None:
        ax.axvline(ss, lw=1.5, ls=":", color="tab:red",
                   label=f"$s^* \\approx {ss:.4f}$")
        mid_y = K_vals.min() * 0.4
        ax.annotate(
            f"$s^*$", xy=(ss, 0),
            xytext=(ss + 0.07, mid_y),
            fontsize=9, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red"),
        )
    ax.set_xlabel(f"Noise scale $s = \\Delta / \\epsilon$", fontsize=_FS_LABEL)
    ax.set_ylabel("Minimum of inner polynomial", fontsize=_FS_LABEL)
    ax.set_title(
        "MSE-safe $\\equiv$ Var-safe: both minima equal $k(s)$\n"
        f"For cubic $f = {coeffs[3]}x^3 + {coeffs[2]}x^2 + {coeffs[1]}x + {coeffs[0]}$,  $\\Delta = {Delta}$",
        fontsize=9,
    )
    ax.legend(fontsize=_FS_LEGEND)
    ax.grid(alpha=_GRID_ALPHA)
    #_add_epsilon_axis(ax, Delta=Delta)
    fig.tight_layout()
    _save_figure(fig, "S_coincidence")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Experiment P — Noise pushes cubic inner polynomial vertically
# ══════════════════════════════════════════════════════════════════════════════

def plot_P_inner(
    coeffs=None, s_levels=None, x_grid=None, Delta: float = 1.0,
) -> plt.Figure:
    """Overlay I_MSE(x) for several noise scales; colour gradient light→dark.

    As noise grows the whole parabola I_MSE(x) lifts upward; once s > s* the
    curve is everywhere positive (unbiased estimator wins globally).
    Saves to figures/P_inner.{pdf,png}.
    """
    if coeffs is None:
        coeffs = [0.0, -3.0, 0.0, 1.0]   # f = x³ − 3x
    a, b, c   = _cubic_params(coeffs)
    ss        = s_star(a, b, c)

    if s_levels is None:
        s_levels = [0.30, 0.45, float(ss) if ss is not None else 0.577, 0.75, 1.0]
    if x_grid is None:
        x_grid = np.linspace(-4.0, 4.0, 500)

    x_arr      = np.asarray(x_grid, dtype=float)
    x_star_val = vertex(a, b)

    n      = len(s_levels)
    cmap   = plt.cm.Blues
    colors = [cmap(0.30 + 0.60 * i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (s_val, col) in enumerate(zip(s_levels, colors)):
        I_MSE = 54*a**2*s_val**2 + 27*a**2*x_arr**2 + 18*a*b*x_arr + 6*a*c + b**2
        is_threshold = (ss is not None and abs(s_val - ss) < 1e-6)
        lw    = _LW + 1 if is_threshold else _LW
        ls    = "--"    if is_threshold else "-"
        label = (f"$s = s^* \\approx {ss:.3f}$ (threshold)"
                 if is_threshold else f"$s = {s_val:.2f}$")
        ax.plot(x_arr, I_MSE, lw=lw, ls=ls, color=col, label=label)

    ax.axhline(0, lw=1, ls="--", color="black")
    ax.axvline(x_star_val, lw=1.2, ls=":", color="tab:red",
               label=f"$x^* = {x_star_val:.2g}$ (vertex)")
    ax.set_xlabel("$x$", fontsize=_FS_LABEL)
    ax.set_ylabel(r"$I_{\mathrm{MSE}}(x)$", fontsize=_FS_LABEL)
    ax.set_title(
        r"Exp. P — $I_{\mathrm{MSE}}(x)$ for increasing noise scale"
        "\n"
        f"$f = x^3 - 3x$,  $\\Delta = {Delta}$,  "
        f"$s^* \\approx {ss:.4f}$ (curve $> 0$ everywhere $\\Leftrightarrow$ unbiased wins)",
        fontsize=9,
    )
    ax.legend(fontsize=_FS_LEGEND, loc="upper center")
    ax.grid(alpha=_GRID_ALPHA)
    fig.tight_layout()
    _save_figure(fig, "P_inner")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Experiment G — Laplace vs Gaussian, log-log + correction table
# ══════════════════════════════════════════════════════════════════════════════

def plot_G_loglog(sweep_epsilon_result: dict, Delta: float = 1.0) -> plt.Figure:
    """Unbiased-estimator variance and MSE vs ε on log-log axes.

    Annotates crossover points where one mechanism overtakes the other.
    Saves to figures/G_loglog.{pdf,png}.
    """
    res   = sweep_epsilon_result
    eps   = np.asarray(res["epsilon_vals"])
    lv    = np.asarray(res["laplace_unbiased_var"])
    gv    = np.asarray(res["gaussian_unbiased_var"])
    lm    = np.asarray(res["laplace_unbiased_mse"])
    gm    = np.asarray(res["gaussian_unbiased_mse"])
    lv_se = np.asarray(res["se_laplace_unbiased_var"])
    gv_se = np.asarray(res["se_gaussian_unbiased_var"])
    lm_se = np.asarray(res["se_laplace_unbiased_mse"])
    gm_se = np.asarray(res["se_gaussian_unbiased_mse"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    datasets = [
        (lv, gv, lv_se, gv_se, "Unbiased variance"),
        (lm, gm, lm_se, gm_se, "Unbiased MSE"),
    ]
    for ax, (lap, gau, lap_se, gau_se, ylabel) in zip(axes, datasets):
        ax.loglog(eps, lap, lw=_LW, color="tab:blue", label="Laplace")
        ax.fill_between(eps,
                        np.maximum(lap - lap_se, 1e-30),
                        lap + lap_se, alpha=0.20, color="tab:blue")
        ax.loglog(eps, gau, lw=_LW, ls="--", color="tab:orange", label="Gaussian")
        ax.fill_between(eps,
                        np.maximum(gau - gau_se, 1e-30),
                        gau + gau_se, alpha=0.20, color="tab:orange")

        cross = np.where(np.diff(np.sign(lap - gau)))[0]
        for ci in cross:
            eps_c = float(0.5 * (eps[ci] + eps[ci + 1]))
            ref_y = float(np.sqrt(np.abs(lap[ci] * lap[ci + 1])))
            ax.axvline(eps_c, lw=1, ls=":", color="gray")
            ax.annotate(
                f"crossover\n$\\varepsilon\\approx{eps_c:.2f}$",
                xy=(eps_c, ref_y),
                xytext=(eps_c * 1.6, ref_y * 2.0),
                fontsize=7, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

        ax.set_xlabel(f"Privacy $\\varepsilon$  ($\\Delta={Delta}$)", fontsize=_FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
        ax.set_title(ylabel, fontsize=_FS_LABEL)
        ax.legend(fontsize=_FS_LEGEND)
        ax.grid(alpha=_GRID_ALPHA, which="both")

    fig.suptitle(
        f"Exp. G — Matched-privacy Laplace vs Gaussian (log-log)  "
        f"$\\delta = {res['delta']:.0e}$,  $\\Delta = {Delta}$",
        fontsize=9,
    )
    fig.tight_layout()
    _save_figure(fig, "G_loglog")
    return fig


def print_estimator_table(
    degrees=(2, 3, 4),
    epsilon: float = 1.0,
    delta: float = 1e-10,
    Delta: float = 1.0,
) -> None:
    """Print unbiased-estimator correction structure for degrees 2, 3, 4.

    Highlights that degree-4 Gaussian carries an extra +3a·σ⁴ constant
    absent in the Laplace correction.
    """
    sigma = calibrateAnalyticGaussianMechanism(epsilon, delta, Delta)
    s_val = Delta / epsilon

    print("\n" + "=" * 70)
    print(f"Estimator correction table  (ε={epsilon}, δ={delta:.0e}, Δ={Delta})")
    print(f"  Laplace  s = Δ/ε = {s_val:.4f}   Var(noise) = 2s² = {2*s_val**2:.4f}")
    print(f"  Gaussian σ (Balle-Wang) = {sigma:.4f}   Var(noise) = σ² = {sigma**2:.4f}")
    print("=" * 70)

    canonical = {
        2: ([0.0, 0.0, 1.0],             "x^2"),
        3: ([0.0, 0.0, 0.0, 1.0],        "x^3"),
        4: ([0.0, 0.0, 0.0, 0.0, 1.0],   "x^4"),
    }
    for deg in degrees:
        coeffs_d, poly_str = canonical[deg]
        _, _, _, g_str_lap = _build_eval_fns(coeffs_d, noise_type="laplace",  s=s_val)
        _, _, _, g_str_gau = _build_eval_fns(coeffs_d, noise_type="gaussian", sigma=sigma)
        print(f"\n  Degree {deg}  f(x) = {poly_str}")
        print(f"    Laplace  g(x) = {g_str_lap}")
        print(f"    Gaussian g(x) = {g_str_gau}")
        if deg == 4:
            print("    *** Degree-4 Gaussian carries extra +3a·σ⁴ constant (from 4th cumulant)")


# ══════════════════════════════════════════════════════════════════════════════
# Closed-form MSE helpers for degree-4 polynomials at q=0
# ══════════════════════════════════════════════════════════════════════════════

def _quartic_mse_laplace(a, b, c, d, e, s):
    """MSE = Var of the Laplace unbiased estimator for a quartic at q=0.

    g_Lap(x) = f(x) − s²f''(x).  MSE(q=0) = E_ξ[(g(ξ) − f(0))²],  ξ~Lap(0,s).
    Laplace moments: E[ξ^k] = k!·s^k (even k), 0 (odd k).
    The constant term e cancels and is not used.
    All arguments broadcast as numpy arrays.
    """
    s2    = s ** 2
    alpha = a
    beta  = b
    gamma = c - 12.0 * a * s2
    delt  = d - 6.0 * b * s2
    eps0  = -2.0 * c * s2
    M2 = 2.0 * s2
    M4 = 24.0 * s2 ** 2
    M6 = 720.0 * s2 ** 3
    M8 = 40320.0 * s2 ** 4
    return (alpha**2 * M8
            + (2.0*alpha*gamma + beta**2) * M6
            + (2.0*alpha*eps0 + 2.0*beta*delt + gamma**2) * M4
            + (2.0*gamma*eps0 + delt**2) * M2
            + eps0**2)


def _quartic_mse_gaussian(a, b, c, d, e, sigma):
    """MSE = Var of the Gaussian unbiased estimator for a quartic at q=0.

    g_Gauss(x) = ax⁴+bx³+(c−6aσ²)x²+(d−3bσ²)x+(e−cσ²+3aσ⁴).
    MSE(q=0) = E_ξ[(g(ξ) − f(0))²],  ξ~N(0,σ²).
    Gaussian moments: E[ξ^k] = (k−1)!!·σ^k (even k), 0 (odd k).
    The constant term e cancels and is not used.
    All arguments broadcast as numpy arrays.
    """
    sig2  = sigma ** 2
    alpha = a
    beta  = b
    gamma = c - 6.0 * a * sig2
    delt  = d - 3.0 * b * sig2
    eps0  = -c * sig2 + 3.0 * a * sig2 ** 2
    M2 = sig2
    M4 = 3.0 * sig2 ** 2
    M6 = 15.0 * sig2 ** 3
    M8 = 105.0 * sig2 ** 4
    return (alpha**2 * M8
            + (2.0*alpha*gamma + beta**2) * M6
            + (2.0*alpha*eps0 + 2.0*beta*delt + gamma**2) * M4
            + (2.0*gamma*eps0 + delt**2) * M2
            + eps0**2)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment G (Part A) — Multi-δ matched-privacy curves
# ══════════════════════════════════════════════════════════════════════════════

def experiment_G_multidelta(
    coeffs=None,
    epsilon_grid=None,
    delta_values=None,
    Delta: float = 1.0,
) -> dict:
    """Closed-form matched-privacy MSE for a quartic, varying δ.

    Evaluated at the true parameter q=0 (exact; no Monte Carlo).
    Laplace is δ-independent; Gaussian σ is calibrated per (ε, δ) via Balle–Wang.

    Returns
    -------
    dict with:
        coeffs, Delta, epsilon_grid, delta_values,
        laplace_s (n_eps), laplace_mse (n_eps),
        gaussian_sigma (n_delta × n_eps), gaussian_mse (n_delta × n_eps).
    """
    if coeffs is None:
        coeffs = [1.0, 0.0, 1.0, 0.0, 1.0]   # q⁴+q²+1
    if epsilon_grid is None:
        epsilon_grid = np.logspace(-1.0, 1.0, 25)
    if delta_values is None:
        delta_values = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

    if len(coeffs) < 5:
        raise ValueError("Need at least 5 coefficients for a degree-4 polynomial.")

    eps_arr = np.asarray(epsilon_grid, dtype=float)
    n_eps   = len(eps_arr)
    n_delta = len(delta_values)

    e_c, d_c, c_c, b_c, a_c = (float(coeffs[k]) for k in range(5))

    s_arr   = Delta / eps_arr
    lap_mse = np.maximum(_quartic_mse_laplace(a_c, b_c, c_c, d_c, e_c, s_arr), 0.0)

    gauss_sigma = np.zeros((n_delta, n_eps))
    gauss_mse   = np.zeros((n_delta, n_eps))
    for i_d, dv in enumerate(delta_values):
        for i_e, eps_val in enumerate(eps_arr):
            sig = calibrateAnalyticGaussianMechanism(
                float(eps_val), float(dv), float(Delta)
            )
            gauss_sigma[i_d, i_e] = sig
            gauss_mse[i_d, i_e]   = float(
                np.maximum(_quartic_mse_gaussian(a_c, b_c, c_c, d_c, e_c, sig), 0.0)
            )

    return {
        "coeffs":         list(coeffs),
        "Delta":          float(Delta),
        "epsilon_grid":   eps_arr.tolist(),
        "delta_values":   list(delta_values),
        "laplace_s":      s_arr.tolist(),
        "laplace_mse":    lap_mse.tolist(),
        "gaussian_sigma": gauss_sigma.tolist(),
        "gaussian_mse":   gauss_mse.tolist(),
    }


def plot_G_multidelta(exp_G_multidelta_result: dict) -> plt.Figure:
    """Part A: Laplace vs Gaussian MSE on log-log axes for multiple δ values.

    Laplace is a bold red reference (δ-independent).  Gaussian family is graded
    blue (light = small δ, dark = large δ).  Crossover with Laplace marked by
    vertical dotted lines.
    Saves to figures/G_multidelta.{pdf,png}.
    """
    res          = exp_G_multidelta_result
    eps          = np.asarray(res["epsilon_grid"])
    lap_mse      = np.asarray(res["laplace_mse"])
    gau_mse      = np.asarray(res["gaussian_mse"])   # (n_delta, n_eps)
    delta_values = res["delta_values"]
    n_delta      = len(delta_values)
    Delta        = res["Delta"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    cmap   = plt.cm.Blues
    colors = [cmap(0.30 + 0.55 * i / max(n_delta - 1, 1)) for i in range(n_delta)]

    for i_d, (dv, col) in enumerate(zip(delta_values, colors)):
        gm = gau_mse[i_d]
        ax.plot(eps, gm, lw=_LW, color=col,
                label=f"Gaussian  $\\delta = {dv:.0e}$")
        cross = np.where(np.diff(np.sign(gm - lap_mse)))[0]
        for ci in cross:
            eps_c = float(0.5 * (eps[ci] + eps[ci + 1]))
            ax.axvline(eps_c, lw=0.7, ls=":", color=col, alpha=0.6)

    ax.plot(eps, lap_mse, lw=_LW + 1.5, color="tab:red",
            label="Laplace  ($\\delta$-independent)")

    ax.set_xlabel(f"Privacy $\\varepsilon$  ($\\Delta={Delta}$)", fontsize=_FS_LABEL)
    ax.set_ylabel("MSE of unbiased estimator  ($=$ Var at $q=0$)", fontsize=_FS_LABEL)
    ax.set_title(
        "Exp. G (Part A) — Matched-privacy Laplace vs Gaussian, varying $\\delta$\n"
        f"$q=0$,  $\\Delta={Delta}$  (closed form, no Monte Carlo)",
        fontsize=9,
    )
    ax.legend(fontsize=_FS_LEGEND, loc="upper right")
    ax.grid(alpha=_GRID_ALPHA, which="both")
    fig.tight_layout()
    _save_figure(fig, "G_multidelta")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Experiment G (Part B) — Exhaustive quartic sweep
# ══════════════════════════════════════════════════════════════════════════════

_NAMED_QUARTICS_DEFAULT = [
    ("$T_4 = 8q^4{-}8q^2{+}1$",  [1.0,  0.0,  -8.0, 0.0,  8.0]),
    ("$q^4$",                      [0.0,  0.0,   0.0, 0.0,  1.0]),
    ("$q^4{+}q^2{+}1$",           [1.0,  0.0,   1.0, 0.0,  1.0]),
    ("$10q^4{-}10q^2$",           [0.0,  0.0, -10.0, 0.0, 10.0]),
]


def experiment_G_quartic_sweep(
    a_grid=None,
    bcd_grid=None,
    epsilon_grid=None,
    delta: float = 1e-10,
    Delta: float = 1.0,
    named_quartics=None,
) -> dict:
    """Closed-form R = MSE_Gauss/MSE_Lap over a grid of quartics at q=0.

    All computation is exact (no Monte Carlo); vectorised over all quartics and
    ε values in a single numpy broadcast.

    Parameters
    ----------
    a_grid : array-like
        Leading coefficient grid (default: {-10,...,10}\\{0}, 20 values).
    bcd_grid : array-like
        Values for b, c, d, e (default: {-10,-5,0,5,10}).
        Total quartics = len(a_grid) × len(bcd_grid)⁴.
    epsilon_grid : array-like
        Privacy ε grid (default: np.logspace(-1, 1, 25)).
    delta, Delta : float
        Privacy δ and sensitivity Δ.
    named_quartics : list of (name, coeffs), optional
        Special quartics to highlight individually.

    Returns
    -------
    dict with:
        epsilon_grid, laplace_s, gaussian_sigma, delta, Delta, n_polys, n_eps,
        ratio_R (numpy array n_polys × n_eps),
        R_median, R_p10, R_p90, R_min, R_max, frac_R_gt1 (each length n_eps),
        named_quartics (list of dicts: name, coeffs, ratio_R).
    """
    if a_grid is None:
        a_grid = [v for v in range(-10, 11) if v != 0]
    if bcd_grid is None:
        bcd_grid = [-10.0, -5.0, 0.0, 5.0, 10.0]
    if epsilon_grid is None:
        epsilon_grid = np.logspace(-1.0, 1.0, 25)
    if named_quartics is None:
        named_quartics = _NAMED_QUARTICS_DEFAULT

    a_arr   = np.asarray(a_grid,   dtype=float)
    bcd_arr = np.asarray(bcd_grid, dtype=float)
    eps_arr = np.asarray(epsilon_grid, dtype=float)
    n_eps   = len(eps_arr)

    print(f"  Building quartic grid: {len(a_arr)} × {len(bcd_arr)}⁴ …",
          end=" ", flush=True)
    a_mg, b_mg, c_mg, d_mg, e_mg = np.meshgrid(
        a_arr, bcd_arr, bcd_arr, bcd_arr, bcd_arr, indexing="ij"
    )
    a_flat = a_mg.ravel()
    b_flat = b_mg.ravel()
    c_flat = c_mg.ravel()
    d_flat = d_mg.ravel()
    e_flat = e_mg.ravel()
    n_polys = len(a_flat)
    print(f"{n_polys} quartics, {n_eps} ε values.", flush=True)

    s_arr     = Delta / eps_arr
    sigma_arr = np.array([
        calibrateAnalyticGaussianMechanism(float(ev), float(delta), float(Delta))
        for ev in eps_arr
    ])

    print("  Computing closed-form MSE (vectorised) …", end=" ", flush=True)
    mse_lap = np.maximum(
        _quartic_mse_laplace(
            a_flat[:, None], b_flat[:, None], c_flat[:, None],
            d_flat[:, None], e_flat[:, None], s_arr[None, :],
        ), 0.0,
    )   # (n_polys, n_eps)
    mse_gau = np.maximum(
        _quartic_mse_gaussian(
            a_flat[:, None], b_flat[:, None], c_flat[:, None],
            d_flat[:, None], e_flat[:, None], sigma_arr[None, :],
        ), 0.0,
    )   # (n_polys, n_eps)
    print("done.", flush=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_R = np.where(mse_lap > 1e-30, mse_gau / mse_lap, np.nan)

    R_median   = np.zeros(n_eps)
    R_p10      = np.zeros(n_eps)
    R_p90      = np.zeros(n_eps)
    R_min      = np.zeros(n_eps)
    R_max      = np.zeros(n_eps)
    frac_R_gt1 = np.zeros(n_eps)
    for i_e in range(n_eps):
        col  = ratio_R[:, i_e]
        ok   = np.isfinite(col) & (col > 0)
        if ok.any():
            vals           = col[ok]
            R_median[i_e]  = float(np.median(vals))
            R_p10[i_e]     = float(np.percentile(vals, 10))
            R_p90[i_e]     = float(np.percentile(vals, 90))
            R_min[i_e]     = float(vals.min())
            R_max[i_e]     = float(vals.max())
            frac_R_gt1[i_e] = float(np.mean(vals > 1.0))

    named_results = []
    for name, nc in named_quartics:
        nc5 = list(nc) + [0.0] * (5 - len(nc))
        a_n = float(nc5[4]); b_n = float(nc5[3]); c_n = float(nc5[2])
        d_n = float(nc5[1]); e_n = float(nc5[0])
        r_lap = np.maximum(
            _quartic_mse_laplace(a_n, b_n, c_n, d_n, e_n, s_arr), 0.0
        )
        r_gau = np.maximum(
            _quartic_mse_gaussian(a_n, b_n, c_n, d_n, e_n, sigma_arr), 0.0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            r_R = np.where(r_lap > 1e-30, r_gau / r_lap, np.nan)
        named_results.append({
            "name":    name,
            "coeffs":  list(nc),
            "ratio_R": r_R.tolist(),
        })

    return {
        "epsilon_grid":   eps_arr.tolist(),
        "laplace_s":      s_arr.tolist(),
        "gaussian_sigma": sigma_arr.tolist(),
        "delta":          float(delta),
        "Delta":          float(Delta),
        "n_polys":        n_polys,
        "n_eps":          n_eps,
        "ratio_R":        ratio_R,         # kept as numpy array
        "R_median":       R_median.tolist(),
        "R_p10":          R_p10.tolist(),
        "R_p90":          R_p90.tolist(),
        "R_min":          R_min.tolist(),
        "R_max":          R_max.tolist(),
        "frac_R_gt1":     frac_R_gt1.tolist(),
        "named_quartics": named_results,
    }


def plot_G_quartic_distribution(exp_G_quartic_result: dict) -> plt.Figure:
    """Part B: Distribution of R = MSE_Gauss/MSE_Lap over the quartic grid.

    Shows median, 10th-90th percentile band, min/max thin dashed lines, and
    individual curves for named quartics.  R=1 marks parity.
    Saves to figures/G_quartic_dist.{pdf,png}.
    """
    res     = exp_G_quartic_result
    eps     = np.asarray(res["epsilon_grid"])
    R_med   = np.asarray(res["R_median"])
    R_p10   = np.asarray(res["R_p10"])
    R_p90   = np.asarray(res["R_p90"])
    R_min   = np.asarray(res["R_min"])
    R_max   = np.asarray(res["R_max"])
    frac    = np.asarray(res["frac_R_gt1"])
    delta   = res["delta"]
    Delta   = res["Delta"]
    n_polys = res["n_polys"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    pos = (R_p10 > 0) & (R_p90 > 0)
    ax.fill_between(eps[pos], R_p10[pos], R_p90[pos],
                    alpha=0.20, color="tab:blue", label="10th–90th pct.")
    pos_min = (R_min > 0)
    pos_max = (R_max > 0)
    ax.plot(eps[pos_min], R_min[pos_min], lw=0.8, ls="--",
            color="tab:blue", alpha=0.5, label="Min / max")
    ax.plot(eps[pos_max], R_max[pos_max], lw=0.8, ls="--",
            color="tab:blue", alpha=0.5)
    pos_med = (R_med > 0)
    ax.plot(eps[pos_med], R_med[pos_med],
            lw=_LW + 1.5, color="tab:blue", label="Median $R$")

    ax.axhline(1.0, lw=1.2, ls="--", color="black", label="$R=1$ (parity)")

    markers = ["o", "s", "D", "^"]
    for i_q, q_info in enumerate(res["named_quartics"]):
        r_vals = np.asarray(q_info["ratio_R"])
        ok     = np.isfinite(r_vals) & (r_vals > 0)
        if ok.any():
            col = plt.cm.tab10(i_q / 10.0)
            ax.plot(eps[ok], r_vals[ok], lw=_LW, color=col,
                    marker=markers[i_q % 4], ms=4, markevery=5,
                    label=q_info["name"])

    ax.set_xlabel(f"Privacy $\\varepsilon$  ($\\Delta={Delta}$)", fontsize=_FS_LABEL)
    ax.set_ylabel(
        r"$R(\varepsilon) = \mathrm{MSE}_{\mathrm{Gauss}} / \mathrm{MSE}_{\mathrm{Lap}}$",
        fontsize=_FS_LABEL,
    )
    frac_lo = float(frac.min()) * 100
    frac_hi = float(frac.max()) * 100
    ax.set_title(
        f"Exp. G (Part B) — $R$ over {n_polys} quartics  "
        f"($\\delta={delta:.0e}$,  $\\Delta={Delta}$,  $q=0$)\n"
        f"Fraction with $R>1$ (Laplace wins in MSE): "
        f"{frac_lo:.0f}%–{frac_hi:.0f}% across $\\varepsilon$",
        fontsize=9,
    )
    ax.legend(fontsize=_FS_LEGEND, loc="upper right")
    ax.grid(alpha=_GRID_ALPHA, which="both")
    fig.tight_layout()
    _save_figure(fig, "G_quartic_dist")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Validation harness
# ══════════════════════════════════════════════════════════════════════════════

def _check(label: str, empirical: float, se: float, theoretical: float,
           tol: float = 3.0, abs_tol: float = 1e-10) -> bool:
    """Print PASS/FAIL and return True if |emp - theo| <= max(tol*SE, abs_tol).

    abs_tol guards against false failures when theory==0 and both the empirical
    value and the bootstrap SE are pure floating-point noise (~1e-16).
    """
    diff   = abs(empirical - theoretical)
    passes = diff <= max(tol * se, abs_tol)
    status = "PASS" if passes else "FAIL"
    print(f"  [{status}] {label}")
    print(f"          empirical={empirical:+.6g}  theory={theoretical:+.6g}"
          f"  diff={diff:.3g}  SE={se:.3g}  ({diff/se if se>0 else float('inf'):.1f} SE)")
    return passes


def run_all(N: int = 200_000, seed: int = 0) -> dict[str, list[bool]]:
    """Run all validation experiments and print PASS/FAIL summary.

    Returns a dict mapping experiment name → list of bool (True=PASS).
    """
    results: dict[str, list[bool]] = {}

    # ── Polynomial definitions ────────────────────────────────────────────────
    quad_coeffs  = [1.0, 2.0, 1.0]            # f = x² + 2x + 1, a=1
    mono_coeffs  = [0.0, 1.0, 0.0, 1.0]       # f = x³ + x,    a=1, b=0, c=1
    nonm_coeffs  = [0.0, -3.0, 0.0, 1.0]      # f = x³ − 3x,   a=1, b=0, c=-3
    deg4_coeffs  = [1.0, 0.0, 1.0, 0.0, 1.0]  # f = x⁴ + x² + 1

    # ── Experiment 1: Quadratic ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Experiment 1 — Quadratic  f = x² + 2x + 1")
    print("="*60)
    exp1 = []
    for x_val, s_val in [(0.0, 0.5), (2.0, 1.0), (-1.0, 0.75)]:
        r = mc_gaps(quad_coeffs, x_val, noise_type="laplace", s=s_val, N=N, seed=seed)
        a_q = float(quad_coeffs[2])
        exp1.append(_check(
            f"var_gap  x={x_val}, s={s_val}",
            r["var_gap"], r["se_var_gap"], quad_var_gap(a_q, s_val)
        ))
        exp1.append(_check(
            f"mse_gap  x={x_val}, s={s_val}",
            r["mse_gap"], r["se_mse_gap"], quad_mse_gap(a_q, s_val)
        ))
    results["quadratic"] = exp1

    # ── Experiment 2: Monotonic cubic ─────────────────────────────────────────
    print("\n" + "="*60)
    print("Experiment 2 — Monotonic cubic  f = x³ + x  (globally safe)")
    print("="*60)
    a_m, b_m, c_m = _cubic_params(mono_coeffs)
    print(f"  s_star = {s_star(a_m, b_m, c_m)}  (None ⇒ safe at any noise)")
    exp2 = []
    for x_val, s_val in [(1.0, 0.5), (0.0, 1.0), (-2.0, 0.5)]:
        r = mc_gaps(mono_coeffs, x_val, noise_type="laplace", s=s_val, N=N, seed=seed)
        exp2.append(_check(
            f"var_gap  x={x_val}, s={s_val}",
            r["var_gap"], r["se_var_gap"],
            cubic_var_gap(a_m, b_m, c_m, x_val, s_val)
        ))
        exp2.append(_check(
            f"mse_gap  x={x_val}, s={s_val}",
            r["mse_gap"], r["se_mse_gap"],
            cubic_mse_gap(a_m, b_m, c_m, x_val, s_val)
        ))
        exp2.append(_check(
            f"bias_h   x={x_val}, s={s_val}",
            r["bias_h"], r["se_bias_h"],
            cubic_naive_bias(a_m, b_m, x_val, s_val)
        ))
    results["monotonic_cubic"] = exp2

    # ── Experiment 3: Non-monotonic cubic ─────────────────────────────────────
    print("\n" + "="*60)
    print("Experiment 3 — Non-monotonic cubic  f = x³ − 3x  (threshold & bands)")
    print("="*60)
    a_n, b_n, c_n = _cubic_params(nonm_coeffs)
    ss_n = s_star(a_n, b_n, c_n)
    print(f"  s_star ≈ {ss_n:.6f}  (unsafe band exists for s < s_star)")
    exp3 = []

    # 3a: gap at vertex vs s
    s_grid_3a = np.linspace(0.1, 1.2, 20)
    for x_val, s_val in [(0.0, 0.3), (0.0, 0.6), (1.0, 0.5)]:
        r = mc_gaps(nonm_coeffs, x_val, noise_type="laplace", s=s_val, N=N, seed=seed)
        exp3.append(_check(
            f"mse_gap  x={x_val}, s={s_val}",
            r["mse_gap"], r["se_mse_gap"],
            cubic_mse_gap(a_n, b_n, c_n, x_val, s_val)
        ))
        exp3.append(_check(
            f"var_gap  x={x_val}, s={s_val}",
            r["var_gap"], r["se_var_gap"],
            cubic_var_gap(a_n, b_n, c_n, x_val, s_val)
        ))

    # 3b: band width ratio
    s_unsafe = 0.3  # < s_star
    x_grid_3b = np.linspace(-3.0, 3.0, 80)
    band_res = measure_band(nonm_coeffs, s_unsafe, x_grid_3b, N=N, seed=seed)
    ratio = band_res["measured"]["width_ratio"]
    expected_ratio = _sqrt(3.0 / 2.0)
    print(f"\n  Band width ratio (var/mse): measured={ratio:.4f}  "
          f"theory={expected_ratio:.4f}")
    # Soft check: within 10 % of theory (grid discretisation limits precision)
    ratio_ok = abs(ratio - expected_ratio) / expected_ratio < 0.10
    print(f"  [{'PASS' if ratio_ok else 'FAIL'}] width_ratio within 10% of sqrt(3/2)")
    exp3.append(ratio_ok)
    results["nonmonotonic_cubic"] = exp3

    # ── Experiment 4: Matched-privacy comparison ───────────────────────────────
    print("\n" + "="*60)
    print("Experiment 4 — Matched-privacy Laplace vs Gaussian")
    print("="*60)
    EPS, DELTA, DELTA_SENS = 1.0, 1e-10, 1.0
    exp4 = []

    for name, coeffs_e4, x_val in [
        ("quadratic",       quad_coeffs, 1.0),
        ("monotonic cubic", mono_coeffs, 1.0),
        ("nonmon. cubic",   nonm_coeffs, 1.0),
        ("degree-4",        deg4_coeffs, 1.0),
    ]:
        cmp = compare_noise(
            coeffs_e4, x_val,
            epsilon=EPS, delta=DELTA, Delta=DELTA_SENS,
            N=N, seed=seed,
        )
        s_v   = cmp["s"]
        sig_v = cmp["sigma"]
        print(f"\n  {name}:  s={s_v:.4f}  sigma={sig_v:.4f}")
        if name == "degree-4":
            g_str_l = cmp["estimator_exprs"]["laplace_unbiased"]
            g_str_g = cmp["estimator_exprs"]["gaussian_unbiased"]
            print(f"    Laplace unbiased:  {g_str_l}")
            print(f"    Gaussian unbiased: {g_str_g}")

        for noise_key in ("laplace", "gaussian"):
            r = cmp[noise_key]
            exp4.append(_check(
                f"bias_g=0  {name}/{noise_key}  x={x_val}",
                r["bias_g"], r["se_bias_g"], 0.0
            ))

    results["matched_privacy"] = exp4

    # ── Experiment Q: exhaustive quadratic sweep ───────────────────────────────
    print("\n" + "="*60)
    print("Experiment Q — Exhaustive quadratic sweep  (N=50 000 per config)")
    print("="*60)
    expQ = experiment_Q(N=min(N, 50_000), seed=seed)
    n_total_Q  = len(expQ["records"])
    n_pass_Q   = int(np.sum(expQ["n_se_mse"] <= 3.0))
    frac_Q     = n_pass_Q / n_total_Q if n_total_Q > 0 else 0.0
    n_neg_Q    = expQ["n_mse_neg"]
    n_nonneg_Q = expQ["n_mse_nonneg"]
    print(f"  Configs tested:            {n_total_Q}")
    print(f"  Max |MSE-gap residual|:    {expQ['max_abs_resid_mse']:.3g}")
    print(f"  Max residual (in SE):      {expQ['max_n_se_mse']:.2f}")
    print(f"  Max |Var-gap residual|:    {expQ['max_abs_resid_var']:.2e}  (exact 0 expected)")
    print(f"  Passing within 3 SE:       {n_pass_Q}/{n_total_Q}  ({frac_Q*100:.1f}%)")
    n_signif_Q = expQ["n_signif_pos"]
    print(f"  MSE-gap < 0 (unbiased wins): {n_neg_Q}/{n_total_Q}  ({n_neg_Q/n_total_Q*100:.2f}%)")
    print(f"  Sign-flipped (emp ≥ 0):      {n_nonneg_Q}/{n_total_Q}  "
          f"— significant (z>2): {n_signif_Q}  (expect 0)")
    if n_signif_Q > 0:
        print(f"  *** {n_signif_Q} genuine sign violation(s) with z > 2 ***")
    # With ~3000 configs, ~0.3% failing 3-SE threshold is expected by chance.
    # PASS criterion: ≥99% within 3 SE and max residual within 5 SE.
    expQ_pass = bool(frac_Q >= 0.99 and expQ["max_n_se_mse"] <= 5.0)
    print(f"  [{'PASS' if expQ_pass else 'FAIL'}] ≥99% within 3 SE "
          f"and max residual ≤ 5 SE")
    results["quadratic_sweep_Q"] = [expQ_pass]

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = True
    for exp_name, checks in results.items():
        n_pass = sum(checks)
        n_total = len(checks)
        status = "PASS" if n_pass == n_total else "FAIL"
        print(f"  [{status}] {exp_name}: {n_pass}/{n_total}")
        if n_pass < n_total:
            all_passed = False
    print("\n" + ("ALL EXPERIMENTS PASSED" if all_passed else "SOME FAILURES — see above"))
    return results


if __name__ == "__main__":
    run_all()
