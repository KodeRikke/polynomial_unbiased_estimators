"""
Script for finding the safe zones for cubic polynomials.
Here it is conditioned by the inequality for the minimum
constant term, i.e.
    M = 54 a^2 s^2 - 2 b^2 + 6 a c
    M >= 0 ---> s^2 >= (b^2 - 3 a c) / (27 a^2)
Then, when the MSE- and var-gap is calculated as
    naive - unbiased = -4 s^4 I (I is the inner term),
the unbiased estimator is safe when I < 0,
and the naive estimator wins when I > 0.
If M < 0, unbiased is safe only in a band around
x = -b / (3a) with band:
    x = -b / (3a) +/- sqrt(M / 27 a^2)
For variance, M is the same, but the band is:
    x = -b / (3a) +/- sqrt(M / 18 a^2)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CubicBandResult:
    """Safe-band analysis for one (a, b, c, s) combination."""

    a: float
    b: float
    c: float
    noise_scale: float  # s = Delta / epsilon

    # M = 54a²s² - 2b² + 6ac  (minimum of I_mse over all x)
    M: float = field(init=False)
    x_center: float = field(init=False)  # -b / (3a)

    # Half-widths; None when M >= 0 (no safe zone)
    mse_half_width: float | None = field(init=False)
    var_half_width: float | None = field(init=False)

    mse_band_lo: float | None = field(init=False)
    mse_band_hi: float | None = field(init=False)
    var_band_lo: float | None = field(init=False)
    var_band_hi: float | None = field(init=False)

    def __post_init__(self) -> None:
        a, b, c, s = self.a, self.b, self.c, self.noise_scale
        self.M = 54.0 * a**2 * s**2 - 2.0 * b**2 + 6.0 * a * c
        self.x_center = -b / (3.0 * a)

        if self.M >= 0.0:
            self.mse_half_width = None
            self.var_half_width = None
            self.mse_band_lo = None
            self.mse_band_hi = None
            self.var_band_lo = None
            self.var_band_hi = None
        else:
            # -M > 0, so the sqrt is real
            neg_M = -self.M
            self.mse_half_width = np.sqrt(neg_M / (27.0 * a**2))
            self.var_half_width = np.sqrt(neg_M / (18.0 * a**2))
            self.mse_band_lo = self.x_center - self.mse_half_width
            self.mse_band_hi = self.x_center + self.mse_half_width
            self.var_band_lo = self.x_center - self.var_half_width
            self.var_band_hi = self.x_center + self.var_half_width

    @property
    def has_safe_zone(self) -> bool:
        return self.M < 0.0

    @property
    def joint_band_lo(self) -> float | None:
        """Lower bound of the zone where both MSE and variance are safe.

        MSE band is always narrower (sqrt(-M/27) < sqrt(-M/18)), so the
        joint safe zone equals the MSE band.
        """
        return self.mse_band_lo

    @property
    def joint_band_hi(self) -> float | None:
        return self.mse_band_hi

    @property
    def joint_half_width(self) -> float | None:
        return self.mse_half_width

    @property
    def noise_threshold(self) -> float:
        """Minimum s for a safe zone to exist: s_crit = sqrt((b² - 3ac) / (27a²)).

        Safe zone exists iff s < s_crit (equivalently M < 0).
        Returns nan when (b² - 3ac) <= 0, meaning no safe zone is ever possible.
        """
        numerator = self.b**2 - 3.0 * self.a * self.c
        if numerator <= 0.0:
            return float("nan")
        return np.sqrt(numerator / (27.0 * self.a**2))


def compute_bands(a: float, b: float, c: float, s: float) -> CubicBandResult:
    return CubicBandResult(a=a, b=b, c=c, noise_scale=s)


def write_csv(path: Path, results: list[CubicBandResult]) -> None:
    header = (
        "a,b,c,noise_scale,M,x_center,noise_threshold,"
        "mse_half_width,var_half_width,"
        "mse_band_lo,mse_band_hi,var_band_lo,var_band_hi,"
        "joint_band_lo,joint_band_hi,joint_half_width,has_safe_zone\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for r in results:
            def _fmt(v: float | None) -> str:
                return "" if v is None else f"{v:.10g}"

            f.write(
                f"{r.a:.10g},{r.b:.10g},{r.c:.10g},{r.noise_scale:.10g},"
                f"{r.M:.10g},{r.x_center:.10g},{_fmt(r.noise_threshold)},"
                f"{_fmt(r.mse_half_width)},{_fmt(r.var_half_width)},"
                f"{_fmt(r.mse_band_lo)},{_fmt(r.mse_band_hi)},"
                f"{_fmt(r.var_band_lo)},{_fmt(r.var_band_hi)},"
                f"{_fmt(r.joint_band_lo)},{_fmt(r.joint_band_hi)},"
                f"{_fmt(r.joint_half_width)},{int(r.has_safe_zone)}\n"
            )


def write_summary(path: Path, results: list[CubicBandResult], args: argparse.Namespace) -> None:
    safe = [r for r in results if r.has_safe_zone]
    with path.open("w", encoding="utf-8") as f:
        f.write("Cubic Safe-Band Analysis\n")
        f.write("=" * 100 + "\n\n")
        f.write("Theory:\n")
        f.write("  f(x) = ax³ + bx² + cx + d,  s = Delta/epsilon\n")
        f.write("  M = 54a²s² - 2b² + 6ac  (minimum of MSE inner term over x)\n")
        f.write("  Safe zone exists iff M < 0, i.e. s < sqrt((b² - 3ac)/(27a²))\n")
        f.write("  Center: x* = -b/(3a)\n")
        f.write("  MSE safe band:  x* ± sqrt(-M/(27a²))\n")
        f.write("  Var safe band:  x* ± sqrt(-M/(18a²))  [always wider]\n")
        f.write("  Joint safe band = MSE band  (narrower, contained in var band)\n\n")

        f.write("Configuration:\n")
        f.write(f"  a values: {args.a_values}\n")
        f.write(f"  b values: {args.b_values}\n")
        f.write(f"  c values: {args.c_values}\n")
        f.write(f"  epsilon values: {args.epsilon_values}\n")
        f.write(f"  Delta: {args.delta}\n\n")

        total = len(results)
        f.write(f"Results:\n")
        f.write(f"  Total (a, b, c, s) combinations: {total}\n")
        f.write(f"  Combinations with safe zone (M < 0): {len(safe)} ({len(safe)/total:.1%})\n\n")

        if safe:
            f.write("Sample safe combinations (joint band widths):\n")
            f.write("─" * 100 + "\n")
            for r in sorted(safe, key=lambda r: (-r.joint_half_width, r.noise_scale, r.a, r.b, r.c))[:40]:
                f.write(
                    f"  (a={r.a:+.2g}, b={r.b:+.2g}, c={r.c:+.2g}) s={r.noise_scale:.4g}: "
                    f"M={r.M:.4g}  x*={r.x_center:.4g}  "
                    f"joint=[{r.joint_band_lo:.4g}, {r.joint_band_hi:.4g}]  "
                    f"half-width={r.joint_half_width:.4g}  "
                    f"var-hw={r.var_half_width:.4g}\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact safe bands for cubic unbiased estimator"
    )
    parser.add_argument(
        "--a-values", type=float, nargs="+",
        default=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
        help="Leading coefficient a values (a=0 skipped)",
    )
    parser.add_argument(
        "--b-values", type=float, nargs="+",
        default=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "--c-values", type=float, nargs="+",
        default=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "--epsilon-values", type=float, nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0],
    )
    parser.add_argument("--delta", type=float, default=1.0, help="Sensitivity Delta")
    parser.add_argument("--output-dir", type=str, default="reports/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CubicBandResult] = []

    for a in args.a_values:
        if a == 0.0:
            continue
        for b in args.b_values:
            for c in args.c_values:
                for epsilon in args.epsilon_values:
                    s = args.delta / epsilon
                    results.append(compute_bands(float(a), float(b), float(c), float(s)))

    safe = [r for r in results if r.has_safe_zone]

    print(f"\n{'='*80}")
    print("Cubic Safe-Band Analysis")
    print(f"{'='*80}")
    print(f"Total combinations: {len(results)}")
    print(f"With safe zone (M < 0): {len(safe)} ({len(safe)/len(results):.1%})\n")

    if safe:
        best = max(safe, key=lambda r: r.joint_half_width)
        print(
            f"Widest joint band:  (a={best.a:+.2g}, b={best.b:+.2g}, c={best.c:+.2g}) "
            f"s={best.noise_scale:.4g}  M={best.M:.4g}  "
            f"joint=[{best.joint_band_lo:.4g}, {best.joint_band_hi:.4g}]  "
            f"half-width={best.joint_half_width:.4g}"
        )

    csv_path = out_dir / "cubic_safe_bands.csv"
    write_csv(csv_path, results)

    summary_path = out_dir / "cubic_safe_bands_summary.txt"
    write_summary(summary_path, results, args)

    print(f"\nCSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
