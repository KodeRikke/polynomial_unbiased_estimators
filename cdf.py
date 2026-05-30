import numpy as np
import sympy as sp

from noise_models import LaplaceNoiseModel
from dp_estimators import EstimatorSystem

__all__ = [
    "laplace_cdf",
    "build_in_intervals",
    "laplace_poly_coeffs",
    "cdf_of_polynomial_in_laplace",
]


def laplace_cdf(z, s):
    """CDF of Lap(0, s) evaluated at z."""
    if z <= 0:
        return 0.5 * np.exp(z / s)
    return 1.0 - 0.5 * np.exp(-z / s)


def build_in_intervals(real_roots, coeffs_shifted):
    """Return list of (a, b) intervals where P(z) - t <= 0.

    real_roots: sorted list of real roots of P(z) - t.
    coeffs_shifted: numpy coefficient array of P(z) - t (highest-degree first).
    """
    breakpoints = [-np.inf] + list(real_roots) + [np.inf]
    intervals_in = []
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        if np.isinf(a) and np.isinf(b):
            test = 0.0
        elif np.isinf(a):
            test = b - 1.0
        elif np.isinf(b):
            test = a + 1.0
        else:
            test = 0.5 * (a + b)
        if np.polyval(coeffs_shifted, test) <= 0:
            intervals_in.append((a, b))
    return intervals_in


def laplace_poly_coeffs(f_sym, q_val, s_val, biasedness="naive"):
    """Return numpy coefficient array (highest-degree first) of P(Z) = estimator(q+Z) - f(q).

    Concretely evaluates at q=q_val with Laplace scale s=s_val.

    f_sym: sympy expression for f in terms of sympy symbol q.
    q_val: concrete value of q.
    s_val: Laplace noise scale (s = Delta/epsilon).
    biasedness: "naive" or "unbiased".
    """
    q = sp.Symbol('q', real=True)
    x = sp.Symbol('x', real=True)
    Z = sp.Symbol('Z', real=True)

    noise_model = LaplaceNoiseModel(Delta=s_val, epsilon=1)
    system = EstimatorSystem(noise_model, q, x)
    estimator = system.estimator(f_sym, biasedness=biasedness)

    P_Z = sp.expand(estimator.subs(x, q + Z) - sp.sympify(f_sym))
    P_Z_concrete = sp.expand(P_Z.subs(q, q_val))

    poly = sp.Poly(P_Z_concrete, Z)
    return np.array([float(c) for c in poly.all_coeffs()])


def cdf_of_polynomial_in_laplace(poly_coeffs_in_Z, s, t_grid):
    """Compute F_{P(Z)}(t) for each t in t_grid, where Z ~ Lap(0, s).

    poly_coeffs_in_Z: numpy coefficient array (highest-degree first) of P(Z).
    s: Laplace noise scale.
    t_grid: 1-D array of threshold values.

    Returns a numpy array of CDF values F_{P(Z)}(t) for each t in t_grid.
    """
    cdfs = []
    for t in t_grid:
        coeffs_shifted = np.array(poly_coeffs_in_Z, dtype=float)
        coeffs_shifted[-1] -= t  # form P(Z) - t
        roots = np.roots(coeffs_shifted)
        real_roots = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
        intervals_in = build_in_intervals(real_roots, coeffs_shifted)
        prob = sum(laplace_cdf(b, s) - laplace_cdf(a, s) for (a, b) in intervals_in)
        cdfs.append(prob)
    return np.array(cdfs)
