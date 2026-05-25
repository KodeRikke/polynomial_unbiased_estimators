"""Analyze cubic unbiased-safety regions in coefficient-ratio space.

For cubic f(x) = a x^3 + b x^2 + c x + d with a != 0, define:
  u = b / a, v = c / a

Given Laplace scale s = Delta / epsilon, the inner terms are
  I_mse(x) = 54 s^2 + 27 x^2 + 18 u x + u^2 + 6 v
  I_var(x) = 54 s^2 + 18 x^2 + 12 u x + 6 v

Since
  MSE(naive) - MSE(unbiased) = -4 s^4 I_mse(x)
  Var(naive) - Var(unbiased) = -4 s^4 I_var(x),
unbiased wins when I_mse(x) < 0 and I_var(x) < 0.

On a bounded interval [x_min, x_max], each inner term is convex in x,
so the maximum over the interval is attained at an endpoint.
Therefore unbiased is safe iff both endpoint maxima are < 0.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RatioPoint:
    epsilon: float
    noise_scale: float
    u: float
    v: float
    mse_endpoint_max: float
    var_endpoint_max: float
    mse_safe: bool
    var_safe: bool

    @property
    def both_safe(self) -> bool:
        return self.mse_safe and self.var_safe

    @property
    def safety_margin(self) -> float:
        """Positive means safe for both (larger is more robust)."""
        return min(-self.mse_endpoint_max, -self.var_endpoint_max)


def mse_inner_ratio(u: float, v: float, x: float, s: float) -> float:
    return 54.0 * s * s + 27.0 * x * x + 18.0 * u * x + u * u + 6.0 * v


def var_inner_ratio(u: float, v: float, x: float, s: float) -> float:
    return 54.0 * s * s + 18.0 * x * x + 12.0 * u * x + 6.0 * v


def endpoint_max(inner_fn, u: float, v: float, s: float, x_min: float, x_max: float) -> float:
    left = inner_fn(u, v, x_min, s)
    right = inner_fn(u, v, x_max, s)
    return float(max(left, right))


def analyze_ratio_grid(
    epsilon: float,
    delta: float,
    x_min: float,
    x_max: float,
    u_values: np.ndarray,
    v_values: np.ndarray,
) -> list[RatioPoint]:
    s = delta / epsilon
    points: list[RatioPoint] = []

    for u in u_values:
        for v in v_values:
            mse_max = endpoint_max(mse_inner_ratio, float(u), float(v), s, x_min, x_max)
            var_max = endpoint_max(var_inner_ratio, float(u), float(v), s, x_min, x_max)
            points.append(
                RatioPoint(
                    epsilon=float(epsilon),
                    noise_scale=float(s),
                    u=float(u),
                    v=float(v),
                    mse_endpoint_max=mse_max,
                    var_endpoint_max=var_max,
                    mse_safe=bool(mse_max < 0.0),
                    var_safe=bool(var_max < 0.0),
                )
            )

    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze safe (u=b/a, v=c/a) regions for cubic unbiased estimator")
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0])
    parser.add_argument("--delta", type=float, default=1.0, help="Sensitivity Delta (s = Delta / epsilon)")
    parser.add_argument("--x-min", type=float, default=-15.0)
    parser.add_argument("--x-max", type=float, default=15.0)
    parser.add_argument("--u-min", type=float, default=-12.0)
    parser.add_argument("--u-max", type=float, default=12.0)
    parser.add_argument("--u-points", type=int, default=241)
    parser.add_argument("--v-min", type=float, default=-120.0)
    parser.add_argument("--v-max", type=float, default=40.0)
    parser.add_argument("--v-points", type=int, default=321)
    parser.add_argument("--output-dir", type=str, default="reports/analysis")
    return parser.parse_args()


def write_points_csv(path: Path, points: list[RatioPoint], x_min: float, x_max: float, delta: float) -> None:
    header = (
        "epsilon,noise_scale,delta,x_min,x_max,u,v,"
        "mse_endpoint_max,var_endpoint_max,mse_safe,var_safe,both_safe,safety_margin\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for p in points:
            f.write(
                f"{p.epsilon:.10g},{p.noise_scale:.10g},{delta:.10g},{x_min:.10g},{x_max:.10g},"
                f"{p.u:.10g},{p.v:.10g},{p.mse_endpoint_max:.10g},{p.var_endpoint_max:.10g},"
                f"{int(p.mse_safe)},{int(p.var_safe)},{int(p.both_safe)},{p.safety_margin:.10g}\n"
            )


def write_boundary_csv(path: Path, epsilon: float, delta: float, x_min: float, x_max: float, u_values: np.ndarray) -> None:
    """Write the exact upper boundary v_upper(u) from endpoint inequalities.

    Safe region is v < v_upper(u), where v_upper is the min of four endpoint-derived bounds
    (MSE left/right and VAR left/right).
    """
    s = delta / epsilon
    with path.open("w", encoding="utf-8") as f:
        f.write("epsilon,noise_scale,delta,x_min,x_max,u,v_upper\n")
        for u in u_values:
            um = float(u)
            mse_left = -(54.0 * s * s + 27.0 * x_min * x_min + 18.0 * um * x_min + um * um) / 6.0
            mse_right = -(54.0 * s * s + 27.0 * x_max * x_max + 18.0 * um * x_max + um * um) / 6.0
            var_left = -(54.0 * s * s + 18.0 * x_min * x_min + 12.0 * um * x_min) / 6.0
            var_right = -(54.0 * s * s + 18.0 * x_max * x_max + 12.0 * um * x_max) / 6.0
            v_upper = min(mse_left, mse_right, var_left, var_right)
            f.write(f"{epsilon:.10g},{s:.10g},{delta:.10g},{x_min:.10g},{x_max:.10g},{um:.10g},{v_upper:.10g}\n")


def write_summary(path: Path, epsilon_to_points: dict[float, list[RatioPoint]], args: argparse.Namespace) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("Cubic Ratio-Space Safety Analysis\n")
        f.write("=" * 100 + "\n\n")
        f.write("Definitions:\n")
        f.write("  u = b/a, v = c/a (a != 0)\n")
        f.write("  s = Delta / epsilon\n")
        f.write("  unbiased safe if I_mse(x) < 0 and I_var(x) < 0 for all x in [x_min, x_max]\n")
        f.write("  with endpoint check due to convexity in x\n\n")

        f.write("Configuration:\n")
        f.write(f"  epsilon values: {args.epsilon_values}\n")
        f.write(f"  Delta: {args.delta}\n")
        f.write(f"  x range: [{args.x_min}, {args.x_max}]\n")
        f.write(f"  u grid: [{args.u_min}, {args.u_max}] with {args.u_points} points\n")
        f.write(f"  v grid: [{args.v_min}, {args.v_max}] with {args.v_points} points\n\n")

        total_per_eps = args.u_points * args.v_points
        for eps in sorted(epsilon_to_points.keys()):
            pts = epsilon_to_points[eps]
            both = [p for p in pts if p.both_safe]
            mse_only = [p for p in pts if p.mse_safe and not p.var_safe]
            var_only = [p for p in pts if p.var_safe and not p.mse_safe]

            frac = (len(both) / total_per_eps) if total_per_eps else 0.0
            f.write(f"epsilon={eps:.4g} (s={args.delta/eps:.4g}):\n")
            f.write(f"  both safe: {len(both)} / {total_per_eps} ({frac:.2%})\n")
            f.write(f"  mse only: {len(mse_only)}\n")
            f.write(f"  var only: {len(var_only)}\n")

            if both:
                best = max(both, key=lambda p: p.safety_margin)
                f.write(
                    "  best robustness point: "
                    f"u={best.u:.3f}, v={best.v:.3f}, margin={best.safety_margin:.3f}\n"
                )
            else:
                f.write("  no jointly safe ratio points in the searched grid\n")
            f.write("\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    u_values = np.linspace(args.u_min, args.u_max, args.u_points)
    v_values = np.linspace(args.v_min, args.v_max, args.v_points)

    epsilon_to_points: dict[float, list[RatioPoint]] = {}

    for epsilon in args.epsilon_values:
        points = analyze_ratio_grid(
            epsilon=float(epsilon),
            delta=float(args.delta),
            x_min=float(args.x_min),
            x_max=float(args.x_max),
            u_values=u_values,
            v_values=v_values,
        )
        epsilon_to_points[float(epsilon)] = points

        eps_token = str(epsilon).replace(".", "p")
        points_path = out_dir / f"cubic_ratio_regions_eps_{eps_token}.csv"
        boundary_path = out_dir / f"cubic_ratio_boundary_eps_{eps_token}.csv"

        write_points_csv(points_path, points, args.x_min, args.x_max, args.delta)
        write_boundary_csv(boundary_path, float(epsilon), args.delta, args.x_min, args.x_max, u_values)

        both_count = sum(1 for p in points if p.both_safe)
        print(
            f"epsilon={epsilon:.4g}: both_safe={both_count}/{len(points)} "
            f"({both_count/len(points):.2%})"
        )

    summary_path = out_dir / "cubic_ratio_regions_summary.txt"
    write_summary(summary_path, epsilon_to_points, args)

    print("\n" + "=" * 100)
    print(f"Summary written to: {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
