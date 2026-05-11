import sys
import csv
import argparse
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monte_carlo.simulation import (
    SimulationConfig,
    absolute_relative_errors,
    print_result,
    simulate_polynomial,
)


@dataclass(frozen=True)
class PolynomialSpec:
    name: str
    expression: sp.Expr
    family: str
    degree: int
    sweep_parameter: str = "fixed"
    sweep_value: float = float("nan")


def _poly_degree(poly_expr: sp.Expr, q: sp.Symbol) -> int:
    try:
        return int(sp.Poly(sp.expand(poly_expr), q).degree())
    except Exception:
        return 0


def _make_spec(
    name: str,
    expression: sp.Expr,
    q: sp.Symbol,
    family: str,
    sweep_parameter: str = "fixed",
    sweep_value: float = float("nan"),
) -> PolynomialSpec:
    return PolynomialSpec(
        name=name,
        expression=sp.expand(expression),
        family=family,
        degree=_poly_degree(expression, q),
        sweep_parameter=sweep_parameter,
        sweep_value=float(sweep_value),
    )


def polynomial_suite(q: sp.Symbol) -> list[PolynomialSpec]:
    fixed = [
        _make_spec("quadratic", q**2, q, "baseline"),
        _make_spec("cubic", q**3, q, "baseline"),
        _make_spec("fifth", q**5 + q**4 + q**3 + q**2, q, "baseline"),
        _make_spec("degree_8", q**8 - 2 * q**6 + 3 * q**4 - q**2 + 1, q, "baseline"),
        _make_spec("degree_10", q**10 - 3 * q**8 + 2 * q**5 + q, q, "baseline"),
        _make_spec("degree_13", q**13 - 2 * q**10 + 4 * q**7 - 3 * q**4 + 2 * q, q, "baseline"),
        _make_spec("chebyshev_T3", sp.chebyshevt(3, q), q, "baseline"),
        _make_spec("chebyshev_T9", sp.chebyshevt(9, q), q, "baseline"),
    ]

    # Broader families to better see general behavior trends.
    monomials = [
        _make_spec(f"monomial_deg_{d}", q**d, q, "baseline")
        for d in [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
    ]
    chebyshev = [
        _make_spec(f"chebyshev_T{d}", sp.chebyshevt(d, q), q, "baseline")
        for d in [2, 4, 6, 8, 10, 12, 14, 16]
    ]

    return fixed + monomials + chebyshev


def coefficient_sweep_suite(q: sp.Symbol) -> list[PolynomialSpec]:
    values = [sp.Integer(-2), 
              sp.Integer(-1), 
              sp.Rational(-1, 2),
              sp.Rational(1, 2), 
              sp.Integer(1), 
              sp.Integer(2),
              sp.Integer(5),
              sp.Integer(10),
              ]
    specs: list[PolynomialSpec] = []

    def add_sweep(family: str, parameter: str, expr_builder):
        for value in values:
            expr = expr_builder(value)
            value_label = str(value).replace("/", "_")
            specs.append(
                _make_spec(
                    name=f"{family}_{parameter}_{value_label}",
                    expression=expr,
                    q=q,
                    family=family,
                    sweep_parameter=parameter,
                    sweep_value=value,
                )
            )

    add_sweep("quadratic_leading", "a", lambda a: a * q**2 + q + 1)
    add_sweep("cubic_leading", "a", lambda a: a * q**3 + q**2 + q + 1)
    add_sweep("cubic_quadratic", "b", lambda b: q**3 + b * q**2 + q + 1)
    add_sweep("cubic_linear", "c", lambda c: q**3 + q**2 + c * q + 1)
    add_sweep("degree_4_leading", "a", lambda a: a * q**4 + q**3 + q**2 + q + 1)
    add_sweep("degree_4_cubic", "b", lambda b: q**4 + b * q**3 + q**2 + q + 1)
    add_sweep("degree_5_leading", "a", lambda a: a * q**5 + q**4 + q**3 + q**2 + q + 1)
    add_sweep("degree_5_quartic", "b", lambda b: q**5 + b * q**4 + q**3 + q**2 + q + 1)
    add_sweep("degree_6_leading", "a", lambda a: a * q**6 + q**5 + q**4 + q**3 + q**2 + q + 1)
    add_sweep("degree_8_leading", "a", lambda a: a * q**8 + q**7 + q**6 + q**5 + q**4 + q**3 + q**2 + q + 1)
    add_sweep("degree_10_leading", "a", lambda a: a * q**10 + q**9 + q**8 + q**7 + q**6 + q**5 + q**4 + q**3 + q**2 + q + 1)

    return specs


def cubic_quartic_ratio_suite(q: sp.Symbol) -> list[PolynomialSpec]:
    """
    Targeted cubic and quartic families with sparse pairwise criss-cross design.
    
    Combines one-factor-at-a-time and pairwise interactions:
      Cubic:   f(x) = x^3 + a*x^2 + b*x + c
      Quartic: f(x) = x^4 + a*x^3 + b*x^2 + c*x + d
    
    Strategy: 
      1. Vary each coefficient independently (one-factor-at-a-time).
      2. Add pairwise interactions: vary two coefficients together in a grid.
      3. Use sparse coefficient grid {-2, -1, 0, 1, 2} for efficiency.
    """
    coeff_values_full = [sp.Integer(-2), sp.Integer(-1), sp.Rational(-1, 2),
                         sp.Integer(0), sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)]
    coeff_values_sparse = [sp.Integer(-2), sp.Integer(-1), sp.Integer(0), sp.Integer(1), sp.Integer(2)]
    specs: list[PolynomialSpec] = []

    def _coeff_to_name(c):
        return str(c).replace("/", "_").replace("-", "m")

    # === CUBIC: x^3 + a*x^2 + b*x + c ===
    # Baseline
    specs.append(_make_spec("cubic_baseline", q**3, q, "cubic_ratios"))
    
    # One-factor-at-a-time (full range)
    for a in coeff_values_full:
        specs.append(_make_spec(f"cubic_a_only_{_coeff_to_name(a)}", q**3 + a*q**2, q, "cubic_ratios"))
    for b in coeff_values_full:
        specs.append(_make_spec(f"cubic_b_only_{_coeff_to_name(b)}", q**3 + b*q, q, "cubic_ratios"))
    for c in coeff_values_full:
        specs.append(_make_spec(f"cubic_c_only_{_coeff_to_name(c)}", q**3 + c, q, "cubic_ratios"))
    
    # Pairwise interactions (sparse grid)
    for a in coeff_values_sparse:
        for b in coeff_values_sparse:
            specs.append(_make_spec(f"cubic_ab_pair_{_coeff_to_name(a)}_{_coeff_to_name(b)}", 
                                   q**3 + a*q**2 + b*q, q, "cubic_ratios"))
    for a in coeff_values_sparse:
        for c in coeff_values_sparse:
            specs.append(_make_spec(f"cubic_ac_pair_{_coeff_to_name(a)}_{_coeff_to_name(c)}", 
                                   q**3 + a*q**2 + c, q, "cubic_ratios"))
    for b in coeff_values_sparse:
        for c in coeff_values_sparse:
            specs.append(_make_spec(f"cubic_bc_pair_{_coeff_to_name(b)}_{_coeff_to_name(c)}", 
                                   q**3 + b*q + c, q, "cubic_ratios"))

    # === QUARTIC: x^4 + a*x^3 + b*x^2 + c*x + d ===
    # Baseline
    specs.append(_make_spec("quartic_baseline", q**4, q, "quartic_ratios"))
    
    # One-factor-at-a-time (full range)
    for a in coeff_values_full:
        specs.append(_make_spec(f"quartic_a_only_{_coeff_to_name(a)}", q**4 + a*q**3, q, "quartic_ratios"))
    for b in coeff_values_full:
        specs.append(_make_spec(f"quartic_b_only_{_coeff_to_name(b)}", q**4 + b*q**2, q, "quartic_ratios"))
    for c in coeff_values_full:
        specs.append(_make_spec(f"quartic_c_only_{_coeff_to_name(c)}", q**4 + c*q, q, "quartic_ratios"))
    for d in coeff_values_full:
        specs.append(_make_spec(f"quartic_d_only_{_coeff_to_name(d)}", q**4 + d, q, "quartic_ratios"))
    
    # Pairwise interactions (sparse grid)
    for a in coeff_values_sparse:
        for b in coeff_values_sparse:
            specs.append(_make_spec(f"quartic_ab_pair_{_coeff_to_name(a)}_{_coeff_to_name(b)}", 
                                   q**4 + a*q**3 + b*q**2, q, "quartic_ratios"))
    for a in coeff_values_sparse:
        for c in coeff_values_sparse:
            specs.append(_make_spec(f"quartic_ac_pair_{_coeff_to_name(a)}_{_coeff_to_name(c)}", 
                                   q**4 + a*q**3 + c*q, q, "quartic_ratios"))
    for a in coeff_values_sparse:
        for d in coeff_values_sparse:
            specs.append(_make_spec(f"quartic_ad_pair_{_coeff_to_name(a)}_{_coeff_to_name(d)}", 
                                   q**4 + a*q**3 + d, q, "quartic_ratios"))
    for b in coeff_values_sparse:
        for c in coeff_values_sparse:
            specs.append(_make_spec(f"quartic_bc_pair_{_coeff_to_name(b)}_{_coeff_to_name(c)}", 
                                   q**4 + b*q**2 + c*q, q, "quartic_ratios"))
    for b in coeff_values_sparse:
        for d in coeff_values_sparse:
            specs.append(_make_spec(f"quartic_bd_pair_{_coeff_to_name(b)}_{_coeff_to_name(d)}", 
                                   q**4 + b*q**2 + d, q, "quartic_ratios"))
    for c in coeff_values_sparse:
        for d in coeff_values_sparse:
            specs.append(_make_spec(f"quartic_cd_pair_{_coeff_to_name(c)}_{_coeff_to_name(d)}", 
                                   q**4 + c*q + d, q, "quartic_ratios"))

    return specs


def edgecase_suite(q: sp.Symbol) -> list[PolynomialSpec]:
    return [
        _make_spec("constant", sp.Integer(7), q, "edgecase"),
        _make_spec("linear", q, q, "edgecase"),
        _make_spec("zero_poly", sp.Integer(0), q, "edgecase"),
        _make_spec("odd_high_degree", q**9 - q**7 + q**3, q, "edgecase"),
        _make_spec("alternating_even", q**12 - q**10 + q**8 - q**6 + q**4 - q**2 + 1, q, "edgecase"),
        _make_spec("small_coeff_high_degree", sp.Rational(1, 50) * q**12 - sp.Rational(1, 10) * q**9 + q, q, "edgecase"),
    ]


def _result_row(spec: PolynomialSpec, result: dict) -> dict:
    cfg = result["config"]
    errs = absolute_relative_errors(result)
    row = {
        "noise": cfg["noise"],
        "poly_name": spec.name,
        "family": spec.family,
        "sweep_parameter": spec.sweep_parameter,
        "sweep_value": spec.sweep_value,
        "degree": spec.degree,
        "q": cfg["q"],
        "epsilon": cfg["epsilon"],
        "n_samples": cfg["n_samples"],
        "seed": cfg["seed"],
        "target": result["target"],
    }
    for est in ("naive", "unbiased"):
        row[f"emp_{est}_mean"] = result["empirical"][est]["mean"]
        row[f"emp_{est}_var"] = result["empirical"][est]["variance"]
        row[f"emp_{est}_mse"] = result["empirical"][est]["mse"]
        row[f"emp_{est}_bias"] = result["empirical"][est]["bias"]
        row[f"sym_{est}_mean"] = result["symbolic"][est]["mean"]
        row[f"sym_{est}_var"] = result["symbolic"][est]["variance"]
        row[f"sym_{est}_mse"] = result["symbolic"][est]["mse"]
        row[f"sym_{est}_bias"] = result["symbolic"][est]["bias"]
        row[f"err_{est}_var_rel"] = errs[est]["variance"]["relative"]
        row[f"err_{est}_mse_rel"] = errs[est]["mse"]["relative"]

    row["emp_var_ratio_unbiased_over_naive"] = (
        row["emp_unbiased_var"] / row["emp_naive_var"] if row["emp_naive_var"] != 0 else float("inf")
    )
    row["emp_mse_ratio_unbiased_over_naive"] = (
        row["emp_unbiased_mse"] / row["emp_naive_mse"] if row["emp_naive_mse"] != 0 else float("inf")
    )
    row["sym_var_ratio_unbiased_over_naive"] = (
        row["sym_unbiased_var"] / row["sym_naive_var"] if row["sym_naive_var"] != 0 else float("inf")
    )
    row["sym_mse_ratio_unbiased_over_naive"] = (
        row["sym_unbiased_mse"] / row["sym_naive_mse"] if row["sym_naive_mse"] != 0 else float("inf")
    )
    return row


def run_for_noise(
    noise: str,
    polynomials: list[PolynomialSpec],
    epsilons: list[float],
    q_values: list[float],
    n_samples: int,
    seed: int,
    verbose: bool,
):
    q_symbol = sp.Symbol("q", real=True)
    rows = []
    print("=" * 100)
    print(f"Noise model: {noise}")

    for spec in polynomials:
        for q_val in q_values:
            for eps in epsilons:
                cfg = SimulationConfig(
                    noise=noise,
                    q_value=q_val,
                    n_samples=n_samples,
                    seed=seed,
                    Delta=1.0,
                    epsilon=eps,
                    delta=1e-10,
                )
                result = simulate_polynomial(spec.expression, cfg)
                rows.append(_result_row(spec, result))
                if verbose:
                    print(f"[{spec.name}]", end=" ")
                    print_result(result)
                    print("-" * 100)
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo study for naive vs unbiased estimators")
    parser.add_argument("--samples", type=int, default=20000, help="Monte Carlo samples per configuration")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/monte_carlo/monte_carlo_results.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--study-mode",
        choices=["baseline", "coefficients", "cubic_quartic", "both"],
        default="both",
        help="Which polynomial families to include in the study",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-configuration details")
    parser.add_argument("--quick", action="store_true", help="Run a smaller quick sanity study")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=None,
        help="Custom epsilon values to sweep (overrides defaults)",
    )
    parser.add_argument(
        "--q-values",
        type=float,
        nargs="+",
        default=None,
        help="Custom q values to sweep (overrides defaults)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    q = sp.Symbol("q", real=True)

    baseline = polynomial_suite(q)
    coefficient_sweeps = coefficient_sweep_suite(q)
    cubic_quartic = cubic_quartic_ratio_suite(q)
    edges = edgecase_suite(q)

    if args.study_mode == "baseline":
        study_polys = baseline
    elif args.study_mode == "coefficients":
        study_polys = coefficient_sweeps
    elif args.study_mode == "cubic_quartic":
        study_polys = cubic_quartic
    else:
        study_polys = baseline + coefficient_sweeps

    if args.quick:
        epsilons = [0.5, 1.0]
        q_values = [0.0, 2.0]
        args.samples = min(args.samples, 20000)
    else:
        # Lower epsilon means stronger privacy / larger noise.
        if args.epsilons is not None:
            epsilons = list(args.epsilons)
        else:
            epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0]
        if args.q_values is not None:
            q_values = list(args.q_values)
        else:
            q_values = [-15.0, -12.0, -10.0, -5.0, -4.0, -2.0, -1.0,
                         0.0, 1.0, 2.0, 4.0, 5.0, 10.0, 12.0, 15.0]

    all_rows = []
    all_rows.extend(
        run_for_noise(
            "laplace",
            study_polys,
            epsilons,
            q_values,
            n_samples=args.samples,
            seed=args.seed,
            verbose=args.verbose,
        )
    )
    all_rows.extend(
        run_for_noise(
            "gaussian",
            study_polys,
            epsilons,
            q_values,
            n_samples=args.samples,
            seed=args.seed,
            verbose=args.verbose,
        )
    )

    print("=" * 100)
    print("Edge cases")
    all_rows.extend(
        run_for_noise(
            "laplace",
            edges,
            epsilons=epsilons,
            q_values=q_values,
            n_samples=args.samples,
            seed=args.seed,
            verbose=args.verbose,
        )
    )
    all_rows.extend(
        run_for_noise(
            "gaussian",
            edges,
            epsilons=epsilons,
            q_values=q_values,
            n_samples=args.samples,
            seed=args.seed,
            verbose=args.verbose,
        )
    )

    output_path = Path(args.output)
    write_csv(all_rows, output_path)
    print("=" * 100)
    print(f"Saved {len(all_rows)} rows to {output_path}")
    print("Tip: run plotting/plot_monte_carlo_study.py to generate plots from this CSV.")


if __name__ == "__main__":
    main()
