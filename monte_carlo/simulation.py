"""Monte Carlo simulation utilities for symbolic estimator systems.

This module evaluates naive and unbiased estimators empirically while preserving the
project's symbolic workflow:
- Build symbolic estimators with `EstimatorSystem`
- Substitute ALL symbols to numeric values before runtime evaluation
- Compare empirical metrics to symbolic/theoretical metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import sympy as sp

from dp_calibration.SigmaFromEpsilon import SigmaFromEpsilon
from dp_estimators import EstimatorSystem
from noise_models import GaussianNoiseModel, LaplaceNoiseModel


@dataclass(frozen=True)
class SimulationConfig:
    noise: str
    q_value: float
    n_samples: int = 100000
    seed: Optional[int] = 1337
    Delta: float = 1.0
    epsilon: float = 1.0
    delta: float = 1e-10
    sigma: Optional[float] = None


def _as_real_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


def _align_polynomial_to_q(f: sp.Expr, q_symbol: sp.Symbol) -> sp.Expr:
    """Replace a single free symbol in f with q_symbol when needed."""
    f_sym = sp.sympify(f)
    if f_sym.has(q_symbol):
        return f_sym

    free_syms = list(f_sym.free_symbols)
    if len(free_syms) == 1:
        return f_sym.subs(free_syms[0], q_symbol)
    return f_sym


def _build_system(cfg: SimulationConfig) -> tuple[EstimatorSystem, Dict[str, Any]]:
    """Build symbolic estimator system + numeric substitution dictionary."""
    q = _as_real_symbol("q")
    x = _as_real_symbol("x")

    noise = cfg.noise.lower()
    if noise == "laplace":
        Delta = _as_real_symbol("Delta")
        epsilon = _as_real_symbol("epsilon")
        model = LaplaceNoiseModel(Delta=Delta, epsilon=epsilon)
        system = EstimatorSystem(noise_model=model, q=q, x=x)
        subs = {
            system.q: cfg.q_value,
            Delta: cfg.Delta,
            epsilon: cfg.epsilon,
        }
    elif noise == "gaussian":
        sigma = _as_real_symbol("sigma")
        model = GaussianNoiseModel(sigma=sigma)
        system = EstimatorSystem(noise_model=model, q=q, x=x)

        sigma_value = cfg.sigma
        if sigma_value is None:
            sigma_value = SigmaFromEpsilon.numeric(
                epsilon_value=cfg.epsilon,
                delta_value=cfg.delta,
                Delta_value=cfg.Delta,
            )

        subs = {
            system.q: cfg.q_value,
            sigma: float(sigma_value),
        }
    else:
        raise ValueError("noise must be 'laplace' or 'gaussian'")

    return system, subs


def _to_numeric_expr(
    expr: sp.Expr,
    subs: Dict[sp.Symbol, Any],
    allowed_free_symbols: Optional[set[sp.Symbol]] = None,
) -> sp.Expr:
    """Apply substitutions and ensure only allowed free symbols remain."""
    local = sp.sympify(expr).subs(subs)
    leftovers = set(local.free_symbols)
    if allowed_free_symbols:
        leftovers = leftovers - allowed_free_symbols
    if leftovers:
        raise ValueError(
            f"Expression still contains symbols {leftovers} after substitution: {local}"
        )
    return sp.N(local)


def _eval_expr_on_samples(expr_x: sp.Expr, x_symbol: sp.Symbol, x_samples: np.ndarray) -> np.ndarray:
    """Evaluate a numeric-in-x symbolic expression on vector samples."""
    poly = sp.Poly(expr_x, x_symbol)
    coeffs = [float(sp.N(c)) for c in poly.all_coeffs()]
    return np.asarray(np.polyval(coeffs, x_samples), dtype=float)


def _metric_summary(values: np.ndarray, target: float) -> Dict[str, float]:
    mean = float(np.mean(values))
    variance = float(np.var(values))
    mse = float(np.mean((values - target) ** 2))
    bias = mean - target
    return {
        "mean": mean,
        "variance": variance,
        "mse": mse,
        "bias": float(bias),
    }


def _constant_report(f_sym: sp.Expr, q_symbol: sp.Symbol) -> Optional[Dict[str, Any]]:
    """Return a synthetic compare-report for constant polynomials."""
    if sp.expand(f_sym) == 0:
        const_expr = sp.Integer(0)
        zero = sp.Integer(0)
        return {
            "polynomial": const_expr,
            "naive": {
                "estimator": const_expr,
                "mean": const_expr,
                "variance": zero,
                "mse": zero,
                "bias": zero,
            },
            "unbiased": {
                "estimator": const_expr,
                "mean": const_expr,
                "variance": zero,
                "mse": zero,
            },
        }

    try:
        poly = sp.Poly(sp.expand(f_sym), q_symbol)
        if poly.is_zero:
            deg = 0
        else:
            deg = poly.degree()
    except Exception:
        return None

    if deg != 0:
        return None

    const_expr = sp.expand(f_sym)
    zero = sp.Integer(0)
    return {
        "polynomial": const_expr,
        "naive": {
            "estimator": const_expr,
            "mean": const_expr,
            "variance": zero,
            "mse": zero,
            "bias": zero,
        },
        "unbiased": {
            "estimator": const_expr,
            "mean": const_expr,
            "variance": zero,
            "mse": zero,
        },
    }


def simulate_polynomial(f: sp.Expr, cfg: SimulationConfig) -> Dict[str, Any]:
    """Run one empirical simulation and compare against symbolic predictions."""
    rng = np.random.default_rng(cfg.seed)

    system, subs = _build_system(cfg)
    f_sym = _align_polynomial_to_q(f, system.q)

    report = _constant_report(f_sym, system.q)
    if report is None:
        report = system.compare(f_sym)

    g_naive_x = _to_numeric_expr(
        report["naive"]["estimator"], subs, allowed_free_symbols={system.x}
    )
    g_unbiased_x = _to_numeric_expr(
        report["unbiased"]["estimator"], subs, allowed_free_symbols={system.x}
    )

    target = float(_to_numeric_expr(report["polynomial"], subs))
    x_symbol = system.x

    if cfg.noise.lower() == "laplace":
        b = float(cfg.Delta / cfg.epsilon)
        noise = rng.laplace(loc=0.0, scale=b, size=cfg.n_samples)
    else:
        sigma_val = float(subs[_as_real_symbol("sigma")])
        noise = rng.normal(loc=0.0, scale=sigma_val, size=cfg.n_samples)

    x_samples = cfg.q_value + noise

    naive_samples = _eval_expr_on_samples(g_naive_x, x_symbol, x_samples)
    unbiased_samples = _eval_expr_on_samples(g_unbiased_x, x_symbol, x_samples)

    empirical_naive = _metric_summary(naive_samples, target)
    empirical_unbiased = _metric_summary(unbiased_samples, target)

    symbolic_naive = {
        "mean": float(_to_numeric_expr(report["naive"]["mean"], subs)),
        "variance": float(_to_numeric_expr(report["naive"]["variance"], subs)),
        "mse": float(_to_numeric_expr(report["naive"]["mse"], subs)),
        "bias": float(_to_numeric_expr(report["naive"]["bias"], subs)),
    }
    symbolic_unbiased = {
        "mean": float(_to_numeric_expr(report["unbiased"]["mean"], subs)),
        "variance": float(_to_numeric_expr(report["unbiased"]["variance"], subs)),
        "mse": float(_to_numeric_expr(report["unbiased"]["mse"], subs)),
        "bias": float(_to_numeric_expr(report["unbiased"]["mean"], subs) - target),
    }

    return {
        "config": {
            "noise": cfg.noise,
            "q": cfg.q_value,
            "n_samples": cfg.n_samples,
            "seed": cfg.seed,
            "Delta": cfg.Delta,
            "epsilon": cfg.epsilon,
            "delta": cfg.delta,
            "sigma": cfg.sigma,
        },
        "polynomial": str(sp.expand(f_sym)),
        "target": target,
        "empirical": {
            "naive": empirical_naive,
            "unbiased": empirical_unbiased,
        },
        "symbolic": {
            "naive": symbolic_naive,
            "unbiased": symbolic_unbiased,
        },
        "symbolic_estimators": {
            "naive": str(sp.expand(g_naive_x)),
            "unbiased": str(sp.expand(g_unbiased_x)),
        },
    }


def absolute_relative_errors(result: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute absolute and relative empirical-vs-symbolic errors."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {"naive": {}, "unbiased": {}}
    for est in ("naive", "unbiased"):
        for metric in ("mean", "variance", "mse", "bias"):
            emp = result["empirical"][est][metric]
            sym = result["symbolic"][est][metric]
            abs_err = abs(emp - sym)
            rel_err = abs_err / abs(sym) if sym != 0 else float("inf")
            out[est][metric] = {
                "absolute": float(abs_err),
                "relative": float(rel_err),
            }
    return out


def print_result(result: Dict[str, Any]) -> None:
    cfg = result["config"]
    print(
        f"noise={cfg['noise']}, poly={result['polynomial']}, q={cfg['q']}, "
        f"epsilon={cfg['epsilon']}, n={cfg['n_samples']}, seed={cfg['seed']}"
    )

    for est in ("naive", "unbiased"):
        emp = result["empirical"][est]
        sym = result["symbolic"][est]
        print(
            f"  {est:8s} empirical  mean={emp['mean']:.6g}, var={emp['variance']:.6g}, "
            f"mse={emp['mse']:.6g}, bias={emp['bias']:.6g}"
        )
        print(
            f"  {est:8s} symbolic   mean={sym['mean']:.6g}, var={sym['variance']:.6g}, "
            f"mse={sym['mse']:.6g}, bias={sym['bias']:.6g}"
        )

    errs = absolute_relative_errors(result)
    print("  errors (empirical vs symbolic)")
    for est in ("naive", "unbiased"):
        e = errs[est]
        print(
            f"    {est:8s} mean_abs={e['mean']['absolute']:.3e}, "
            f"var_rel={e['variance']['relative']:.3e}, mse_rel={e['mse']['relative']:.3e}"
        )
