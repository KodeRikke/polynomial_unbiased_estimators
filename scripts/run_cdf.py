"""
run_cdf.py  —  Entry point for CDF plots of polynomial estimators under Laplace noise.

Two figures are generated:
  cdf_comparison_horizontal.png  :  1×4 grid, panels 1 → 2 → 3 → 4 left to right
  cdf_comparison_vertical.png    :  2×2 grid, panels 1 & 4 on top, panels 2 & 3 on bottom

Output is saved to  plots/cdf/  (created automatically if needed).

============================  EDIT HERE  ============================
The only section you need to edit is CASES below.

CASES entries  (one dict per panel, 1-indexed in the comments)
--------------
  f_sym   : the polynomial f(q) as a sympy expression in the symbol q
              Examples:  q**3
                         q**3 - 3*q
                         sp.chebyshevt(3, q)
  q_val   : the concrete query value at which to evaluate
  s_val   : Laplace noise scale  s = Delta / epsilon
  t_range : (t_min, t_max) — horizontal extent of the CDF panel
  label   : title string for the panel  (LaTeX math is supported)
  t_n     : (optional) number of grid points along t, default 400

To add a new case, append a new dict to CASES and update the
arrangement in main() if needed (0-indexed into CASES).
=====================================================================
"""
import sympy as sp
from plotting.plot_cdf import plot_cdf

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  EDIT HERE  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

q = sp.Symbol("q", real=True)

# For a cubic f=aq^3+bq^2+cq+d the unsafe threshold at q with two real critical points is
#   s* = sqrt((b^2 - 3ac) / (27a^2))
# s < s*  →  unbiased estimator may violate monotonicity  (unsafe regime)
# s > s*  →  unbiased estimator is always monotone         (safe regime)
#
#   f = q^3 - 3q  :  s* = sqrt(9/27)  = sqrt(1/3) ≈ 0.58
#   f = q^3 - 9q  :  s* = sqrt(81/27) = sqrt(3)   ≈ 1.73

CASES = [
    # Panel 1 — safe polynomial, noise well below any unsafe threshold
    {
        "f_sym":   q**3 + q,
        "q_val":   0,
        "s_val":   0.3,         # epsilon ≈ 3.0, Delta = 1.0
        "t_range": (-3, 3), 
        "label":   r"$f(q)=q^3+q$ with $s=0.3$",
    },
    # Panel 2 — unsafe polynomial, but s > s*  →  effectively safe noise level
    {
        "f_sym":   q**3 - 3*q,
        "q_val":   0,
        "s_val":   0.7,        # s=0.7 > s*≈0.58  →  safe
        "t_range": (-6, 6),
        "label":   r"$f(q)=q^3-3q$ with $s=0.7$",
    },
    # Panel 3 — unsafe polynomial, s < s*  →  unsafe regime
    {
        "f_sym":   q**3 - 3*q,
        "q_val":   0,
        "s_val":   0.3,         # s=0.3 < s*≈0.58  →  unsafe
        "t_range": (-3, 3),
        "label":   r"$f(q)=q^3-3q$ with $s=0.3$",
    },
    # Panel 4 — more extreme unsafe polynomial, s << s*
    {
        "f_sym":   q**3 - 9*q,
        "q_val":   0,
        "s_val":   0.3,         # s=0.3 << s*≈1.73  →  strongly unsafe
        "t_range": (-8, 8),
        "label":   r"$f(q)=q^3-9q$ with $s=0.3$",
    },
]
# Baselines: q**3 
# Always globally safe, b**2 =< 3ac: q**3 + q SO 0**2 <= 3*1*1 = 3
# Unsafe example, b**2 > 3ac: q**3 - 3q SO (-3)**2 > 3*1*0 = 0
# s.t. s* = sqrt{(b**2 - 3ac)/(27a**2)} = sqrt{(9-0)/27} = sqrt{1/3} \approx 0.58
# Extreme unsafe example, b**2 >> 3ac: q**3 - 9q SO (-9)**2 >> 3*1*0 = 0
# s.t. s* = sqrt{(b**2 - 3ac)/(27a**2)} = sqrt{(81-0)/27} = sqrt{3} \approx 1.73

# Panel 1: SAFE example

# Panel 2 and 3: UNSAFE example, with different noise levels to show how the CDF changes
# Panel 2 with s = 1.0(epsilon = 1.0), safe
# Panel 3 with s = 0.3(epsilon = 3.0), unsafe
# Panel 4 with s = 0.3(epsilon = 3.0), unsafe, more extreme

#  q**3 + 3*q**2 + 1*q,

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def main():
    # Figure 1: 1×4 grid — panels 1, 2, 3, 4 left to right
    plot_cdf(
        cases=CASES,
        save_path="plots/cdf",
        filename="cdf_comparison_horizontal_alternativ",
    )

    # Figure 2: 2×2 grid — panels 1 & 4 on top row, panels 2 & 3 on bottom row
    plot_cdf(
        cases=CASES,
        save_path="plots/cdf",
        filename="cdf_comparison_vertical_alternative", 
        arrangement=[[0, 3], [1, 2]],
    )


if __name__ == "__main__":
    main()
