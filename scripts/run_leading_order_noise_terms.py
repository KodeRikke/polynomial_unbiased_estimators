from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotting.plot_symbolic_presentation import build_reports


def _leading_term(expr: sp.Expr, t: sp.Symbol, max_order: int = 14) -> tuple[int | None, sp.Expr | None]:
    """Return first non-zero Taylor term around t=0 as (order, coefficient)."""
    expr = sp.simplify(expr)
    try:
        series_expr = sp.series(expr, t, 0, max_order + 1).removeO().expand()
        for order in range(max_order + 1):
            coeff = sp.simplify(series_expr.coeff(t, order))
            if coeff.is_zero is False:
                return order, coeff
            if coeff.is_zero is None and sp.simplify(coeff) != 0:
                return order, coeff
    except Exception:
        pass

    # Fallback: derivative-based extraction of Taylor coefficients.
    for order in range(max_order + 1):
        try:
            coeff = sp.simplify(sp.diff(expr, t, order).subs(t, 0) / sp.factorial(order))
            if coeff.is_zero is False:
                return order, coeff
            if coeff.is_zero is None and sp.simplify(coeff) != 0:
                return order, coeff
        except Exception:
            continue

    return None, None


def _sign_label(value: sp.Expr) -> str:
    value = sp.simplify(value)
    if value.is_real is False:
        return "complex"
    if value.is_zero:
        return "0"
    if value.is_positive:
        return "+"
    if value.is_negative:
        return "-"
    try:
        num = float(value.evalf())
    except Exception:
        return "?"
    if num > 0:
        return "+"
    if num < 0:
        return "-"
    return "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate leading-order noise-term summary table")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/analysis",
        help="Directory for markdown and CSV outputs",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=14,
        help="Max Taylor order to search for first non-zero term",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = build_reports()

    q = sp.Symbol("q", real=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)
    Delta = sp.Symbol("Delta", real=True, positive=True)
    sigma = sp.Symbol("sigma", real=True, positive=True)
    t = sp.Symbol("t", real=True, positive=True)

    rows: list[dict[str, str]] = []

    for poly_name, report_pair in reports.items():
        for noise_name, report in report_pair.items():
            for metric in ["mse"]:#["variance", "mse"]:
                delta_expr = sp.simplify(report["naive"][metric] - report["unbiased"][metric])

                if noise_name == "laplace":
                    # t is Laplace noise scale: t = Delta / epsilon.
                    # Expand in t by substituting epsilon = Delta / t (so Delta stays fixed and t varies).
                    delta_t = sp.simplify(delta_expr.subs({epsilon: Delta / t, Delta: Delta}))
                    scale_name = "noise_scale"
                else:
                    # t is Gaussian std: t = sigma.
                    delta_t = sp.simplify(delta_expr.subs({sigma: t}))
                    scale_name = "noise_scale"

                order, coeff = _leading_term(delta_t, t, max_order=args.max_order)

                if order is None or coeff is None:
                    leading_term = "not-found"
                    coeff_q0 = "n/a"
                    coeff_q1 = "n/a"
                    sign_q0 = "?"
                    sign_q1 = "?"
                else:
                    leading_term = f"({sp.sstr(sp.simplify(coeff))}) * {scale_name}^{order}"
                    coeff_at_q0 = sp.simplify(coeff.subs(q, 0))
                    coeff_at_q1 = sp.simplify(coeff.subs(q, 1))
                    coeff_q0 = sp.sstr(coeff_at_q0)
                    coeff_q1 = sp.sstr(coeff_at_q1)
                    sign_q0 = _sign_label(coeff_at_q0)
                    sign_q1 = _sign_label(coeff_at_q1)

                # Evaluate at q=0.001 and q=0.1
                coeff_at_q001 = sp.simplify(coeff.subs(q, 0.001)) if coeff else None
                coeff_at_q01 = sp.simplify(coeff.subs(q, 0.1)) if coeff else None
                sign_q001 = _sign_label(coeff_at_q001) if coeff_at_q001 else "?"
                sign_q01 = _sign_label(coeff_at_q01) if coeff_at_q01 else "?"
                coeff_q001 = sp.sstr(coeff_at_q001) if coeff_at_q001 else "n/a"
                coeff_q01 = sp.sstr(coeff_at_q01) if coeff_at_q01 else "n/a"
                # Evaluate at q=0.3 and q=0.5
                coeff_at_q03 = sp.simplify(coeff.subs(q, 0.3)) if coeff else None
                coeff_at_q05 = sp.simplify(coeff.subs(q, 0.5)) if coeff else None
                sign_q03 = _sign_label(coeff_at_q03) if coeff_at_q03 else "?"
                sign_q05 = _sign_label(coeff_at_q05) if coeff_at_q05 else "?"
                coeff_q03 = sp.sstr(coeff_at_q03) if coeff_at_q03 else "n/a"
                coeff_q05 = sp.sstr(coeff_at_q05) if coeff_at_q05 else "n/a"
                # Evaluate at q=0.2 and q=0.4
                coeff_at_q02 = sp.simplify(coeff.subs(q, 0.2)) if coeff else None
                coeff_at_q04 = sp.simplify(coeff.subs(q, 0.4)) if coeff else None
                sign_q02 = _sign_label(coeff_at_q02) if coeff_at_q02 else "?"
                sign_q04 = _sign_label(coeff_at_q04) if coeff_at_q04 else "?"
                coeff_q02 = sp.sstr(coeff_at_q02) if coeff_at_q02 else "n/a"
                coeff_q04 = sp.sstr(coeff_at_q04) if coeff_at_q04 else "n/a"

                # In increasing q order: q=0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 1
                rows.append(
                    {
                        "polynomial": poly_name,
                        "noise": noise_name,
                        "metric_diff": f"naive_{metric}-unbiased_{metric}",
                        "leading_term": leading_term,
                        "coeff_q0": coeff_q0,
                        "sign_q0": sign_q0,
                        "coeff_q001": coeff_q001,
                        "sign_q001": sign_q001,
                        "coeff_q01": coeff_q01,
                        "sign_q01": sign_q01,
                        "coeff_q02": coeff_q02,
                        "sign_q02": sign_q02,
                        "coeff_q04": coeff_q04,
                        "sign_q04": sign_q04,
                        "coeff_q03": coeff_q03,
                        "sign_q03": sign_q03,
                        "coeff_q05": coeff_q05,
                        "sign_q05": sign_q05,
                        "coeff_q1": coeff_q1,
                        "sign_q1": sign_q1,
                    }
                )

    csv_path = out_dir / "leading_order_noise_terms.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "polynomial",
                "noise",
                "metric_diff",
                "leading_term",
                "coeff_q0",
                "sign_q0",
                "coeff_q001",
                "sign_q001",
                "coeff_q01",
                "sign_q01",
                "coeff_q02",
                "sign_q02",
                "coeff_q04",
                "sign_q04",
                "coeff_q03",
                "sign_q03",
                "coeff_q05",
                "sign_q05",
                "coeff_q1",
                "sign_q1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "leading_order_noise_terms.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Leading-order noise terms\n\n")
        f.write("Difference analyzed: naive - unbiased. Positive values imply unbiased is better for that metric.\n\n")
        f.write("**IMPORTANT:** This table shows the leading-order Taylor coefficient at **small noise scales** (t → 0, i.e., very high privacy / low ε). At practical DP noise scales (ε ∈ [0.5, 5]), **higher-order terms often dominate** and can flip the sign. For example, Chebyshev T9 has a negative leading t⁴ term, but a large positive t⁶ term that overwhelms it at t=1. Thus, the sign at practical noise scales may differ from this asymptotic prediction.\n\n")
        f.write("Laplace expansion variable: t = Delta/epsilon (with Delta fixed to 1 in expansion).\n")
        f.write("Gaussian expansion variable: t = sigma.\n\n")
        f.write("The sign columns show whether the difference (naive - unbiased) is positive (+) or negative (−) at each q value.\n\n")
        f.write("| Polynomial | Noise | Metric diff | Leading term | sign@q=0 | sign@q=0.001 | sign@q=0.1 | sign@q=0.2 | sign@q=0.3 | sign@q=0.4 | sign@q=0.5 | sign@q=1 |\n")
        f.write("|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\\n")
        for row in rows:
            f.write(
                "| {polynomial} | {noise} | {metric_diff} | {leading_term} | {sign_q0} | {sign_q001} | {sign_q01} | {sign_q02} | {sign_q03} | {sign_q04} | {sign_q05} | {sign_q1} |\n".format(
                    **row
                )
            )

    print(f"Saved table: {md_path}")
    print(f"Saved csv:   {csv_path}")


if __name__ == "__main__":
    main()
