"""
Analyze coefficient safety regions for cubic polynomials.

Given theoretical MSE/variance gap expressions for cubic functions:
  f(x) = ax^3 + bx^2 + cx + d

Determines for which coefficient combinations the unbiased estimator 
theoretically dominates across all x values (high-noise, low-noise, etc.).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class CubicGapAnalysis:
    """Analysis result for a cubic coefficient combination."""
    
    a: float
    b: float
    c: float
    
    # At noise scale s = Δ/ε
    noise_scale: float  # s value
    minimum_value: float  # 54a²s² + 6ac - 2b² (minimum of the inner polynomial)
    x_at_minimum: float  # x = -b/(3a) where the minimum is attained
    is_globally_safe: bool  # True if minimum_value >= 0 (unbiased always wins)
    mse_unsafe_half_width: float | None   # None if globally safe
    var_unsafe_half_width: float | None
    min_safe_noise_scale: float           # sqrt((b² - 3ac) / (27a²)) if b²>3ac else 0    
    
#def mse_inner_cubic(a: float, b: float, c: float, x: float, s: float) -> float:
#    """
#    The inner polynomial I_MSE(x) from MSE(g) - MSE(h) = -4s^4 * I_MSE(x).
#    Unbiased estimator wins (lower MSE) iff I_MSE(x) >= 0.
#    """
#    return 54*a**2*s**2 + 27*a**2*x**2 + 18*a*b*x + b**2 + 6*a*c
#
#
#def variance_inner_cubic(a: float, b: float, c: float, x: float, s: float) -> float:
#    """
#    Inner polynomial I_Var(x) from Var(g) - Var(h) = -4s^4 * I_Var(x).
#    Unbiased has lower variance iff I_Var(x) >= 0.
#    """
#    return 54*a**2*s**2 + 18*a**2*x**2 + 12*a*b*x + 6*a*c


def analyze_coefficient_pair(
    a: float, b: float, c: float, noise_scale: float
) -> CubicGapAnalysis:
    """
    Analyze MSE and variance gap safety for a cubic with given coefficients.
    
    Uses the closed-form minimum of the inner polynomial:
        I_MSE(x) = 27a²(x + b/(3a))² + M
        I_Var(x) = 18a²(x + b/(3a))² + M
    where
        M = 54a²s² + 6ac - 2b²
    
    Both inners share the same minimum value M, attained at x = -b/(3a).
    The unbiased estimator is globally safe (lower MSE and lower variance 
    for all x) iff M >= 0.
    """
    s = noise_scale
    M = 54 * a**2 * s**2 + 6 * a * c - 2 * b**2
    
    is_globally_safe = (M >= 0)
    x_minimum = -b / (3 * a)  # location of the worst-case x
    
    if M < 0:
        half_width_mse = float(np.sqrt(-M / (27 * a**2)))
        half_width_var = float(np.sqrt(-M / (18 * a**2)))
    else:
        half_width_mse = None
        half_width_var = None

    return CubicGapAnalysis(
        a=a,
        b=b,
        c=c,
        noise_scale=s,
        minimum_value=M,
        x_at_minimum=x_minimum,
        is_globally_safe=is_globally_safe,
        mse_unsafe_half_width=half_width_mse,
        var_unsafe_half_width=half_width_var,
        # for each (a,b,c), the noise threshold above which that cubic is globally safe — independent of the specific s
        min_safe_noise_scale=np.sqrt((b**2 - 3*a*c) / (27*a**2)) if b**2 > 3*a*c else 0.0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cubic polynomial safety regions (unbiased dominance)"
    )
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Coefficient values to test",
    )
    parser.add_argument(
        "--epsilon-values",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0],
        help="Epsilon values (ε) in the noise scale to test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/analysis",
        help="Output directory for results",
    )
    return parser.parse_args()

def write_report(results, safe_results, unsafe, epsilon_values, write):
    """Write the full report. `write` is a callable taking a single string."""
    write(f"\n{'='*100}\n")
    write("Cubic Polynomial Safety Analysis\n")
    write(f"{'='*100}\n\n")
    
    write(f"Total combinations tested: {len(results)}\n")
    write(f"Globally safe (unbiased wins for all x): {len(safe_results)}\n")
    write(f"Globally unsafe (unbiased loses on some band of x): {len(unsafe)}\n\n")
    
    if safe_results:
        write("Globally safe combinations:\n")
        write("─" * 100 + "\n")
        for r in sorted(safe_results, key=lambda r: (r.noise_scale, r.a, r.b, r.c)):
            write(
                f"  (a={r.a:+.1f}, b={r.b:+.1f}, c={r.c:+.1f}) @ s={r.noise_scale:.2f}: "
                f"safety margin M={r.minimum_value:.2e} "
                f"(worst-case x={r.x_at_minimum:.2f})\n"
            )
    
    if unsafe:
        write(f"\nUnsafe combinations (showing danger bands):\n")
        write("─" * 100 + "\n")
        write("(Variance-unsafe band is wider than MSE-unsafe band by factor √(27/18) ≈ 1.22)\n\n")
        for r in sorted(unsafe, key=lambda r: (r.noise_scale, r.a, r.b, r.c)):
            x_min = r.x_at_minimum
            write(
                f"  (a={r.a:+.1f}, b={r.b:+.1f}, c={r.c:+.1f}) @ s={r.noise_scale:.2f}: "
                f"M={r.minimum_value:.2e}, "
                f"MSE-unsafe ⊂ [{x_min - r.mse_unsafe_half_width:.2f}, {x_min + r.mse_unsafe_half_width:.2f}], "
                f"Var-unsafe ⊂ [{x_min - r.var_unsafe_half_width:.2f}, {x_min + r.var_unsafe_half_width:.2f}], "
                f"safe at s ≥ {r.min_safe_noise_scale:.3f}\n"
            )
    
    write("\nFraction of coefficient grid that is globally safe, by noise scale:\n")
    write("─" * 60 + "\n")
    for eps in sorted(epsilon_values, reverse=True):
        s = 1.0 / eps
        at_this_s = [r for r in results if abs(r.noise_scale - s) < 1e-9]
        n_safe = sum(1 for r in at_this_s if r.is_globally_safe)
        n_total = len(at_this_s)
        pct = 100 * n_safe / n_total if n_total else 0
        write(f"  ε={eps:5.2f} (s={s:.3f}): {n_safe:5d}/{n_total} = {pct:5.1f}% safe\n")

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all coefficient combinations
    results = []
    for a in args.coefficients:
        for b in args.coefficients:
            for c in args.coefficients:
                # Skip a=0 (not really cubic)
                if a == 0:
                    continue
                
                for epsilon in args.epsilon_values:
                    # Convert epsilon to noise scale (s = Δ/ε)
                    s = 1.0 / epsilon
                    analysis = analyze_coefficient_pair(a, b, c, s)
                    results.append(analysis)

    # Find "always safe" regions
    safe_results = [r for r in results if r.is_globally_safe]

    print(f"\n{'='*100}")
    print(f"Cubic Polynomial Safety Analysis")
    print(f"{'='*100}\n")

    print(f"Total combinations tested: {len(results)}")
    print(f"Globally safe (unbiased wins for all x): {len(safe_results)}")
    print(f"Globally unsafe (unbiased loses on some band of x): {len(results) - len(safe_results)}\n")

    if safe_results:
        print("Globally safe combinations:")
        print("─" * 100)
        for r in sorted(safe_results, key=lambda r: (r.noise_scale, r.a, r.b, r.c)):
            print(
                f"  (a={r.a:+.1f}, b={r.b:+.1f}, c={r.c:+.1f}) @ s={r.noise_scale:.2f}: "
                f"safety margin M={r.minimum_value:.2e} "
                f"(worst-case x={r.x_at_minimum:.2f})"
            )

    unsafe = [r for r in results if not r.is_globally_safe]

    # Terminal
    write_report(results, safe_results, unsafe, args.epsilon_values, lambda s: print(s, end=''))

    # File
    summary_path = out_dir / "cubic_safety_analysis.txt"
    with summary_path.open("w") as f:
        write_report(results, safe_results, unsafe, args.epsilon_values, f.write)

if __name__ == "__main__":
    main()
