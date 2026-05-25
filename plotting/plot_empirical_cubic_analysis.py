"""Empirical analysis plots for the cubic polynomial MSE-gap theory.

Style anchored to plotting/plot_symbolic_presentation.py:
  colours  = plt.cm.tab10(np.linspace(0, 1, n))
  linewidth = 2
  grid      = ax.grid(True, alpha=0.3)
  backend   = Agg
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_safe_zone_boundary",
    "plot_safe_fraction_vs_epsilon",
    "plot_gap_function",
    "plot_monte_carlo_validation",
]

# ── Style constants (matching plot_symbolic_presentation.py) ─────────────────
_LW = 2
_GRID_ALPHA = 0.3
_FS_TITLE = 11
_FS_LABEL = 10
_FS_LEGEND = 8


def _tab10(n: int) -> np.ndarray:
    """n evenly-spaced RGBA colours from the tab10 qualitative map."""
    pts = np.linspace(0, 1, n) if n > 1 else np.array([0.0])
    return plt.cm.tab10(pts)


# ── Closed-form helpers ──────────────────────────────────────────────────────

def _inner_poly(
    a: float, b: float, c: float, s: float, x: np.ndarray
) -> np.ndarray:
    """I_MSE_cubic(x) = 54a²s² + 27a²x² + 18abx + b² + 6ac."""
    return 54*a**2*s**2 + 27*a**2*x**2 + 18*a*b*x + b**2 + 6*a*c


def _K(a: float, b: float, c: float, s: float) -> float:
    """K = 54a²s² + 6ac − 2b²  (minimum of I_MSE_cubic, attained at x = −b/3a)."""
    return float(54*a**2*s**2 + 6*a*c - 2*b**2)


def _gap_theory(a: float, b: float, c: float, q: float, s: float) -> float:
    """Closed-form MSE gap: −4s⁴ · I_MSE_cubic(q)."""
    return float(-4 * s**4 * _inner_poly(a, b, c, s, np.asarray([q]))[0])


# ── Figure 1 ─────────────────────────────────────────────────────────────────

def plot_safe_zone_boundary(
    noise_scales: list[float],
    shade_s: float,
    Delta: float = 1.0,
    beta_range: tuple[float, float] = (-10.0, 10.0),
    gamma_range: tuple[float, float] = (-25.0, 5.0),
    n_pts: int = 600,
) -> plt.Figure:
    """Validates that the safety boundary γ = (β² − 27s²)/3 correctly
    separates safe from unsafe normalised coefficient pairs (β, γ) = (b/a, c/a).
    Each curve corresponds to a fixed noise scale s = Δ/ε; the safe region
    (γ ≥ boundary) is shaded for one chosen s.  The monotonicity boundary
    γ = β²/3 (s → 0 limit) is shown as a reference."""
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    beta = np.linspace(beta_range[0], beta_range[1], n_pts)
    gamma_min, gamma_max = gamma_range

    cmap = plt.cm.viridis
    s_vals = sorted(noise_scales)
    norm = mcolors.Normalize(vmin=min(s_vals), vmax=max(s_vals))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # ── One boundary curve per noise scale ───────────────────────────────────
    for s in s_vals:
        boundary = (beta**2 - 27*s**2) / 3
        ax.plot(beta, boundary, color=cmap(norm(s)), linewidth=_LW)

    # ── Monotonicity boundary: s → 0 limit, γ = β²/3 ────────────────────────
    # Solid black, longer dashes, slightly thicker so it reads distinctly.
    mono_bnd = beta**2 / 3
    mono_line, = ax.plot(
        beta, mono_bnd,
        color="black", linewidth=1.5, linestyle=(0, (9, 4)),
        label=r"$s=0$ (monotonicity boundary)",
    )

    # ── Shade safe region for shade_s ────────────────────────────────────────
    bnd_shade = (beta**2 - 27*shade_s**2) / 3
    shade_color = cmap(norm(shade_s))
    eps_shade = Delta / shade_s
    shade_patch = ax.fill_between(
        beta, bnd_shade, gamma_max,
        where=(bnd_shade <= gamma_max),
        alpha=0.30, color=shade_color,
        label=rf"Safe region ($s={shade_s:g}$, $\varepsilon={eps_shade:.3g}$)",
    )

    # Text label inside the shaded region near the top, indicating it continues
    # above the plot boundary.
    ax.text(
        0.0, gamma_max - 0.7,
        rf"Safe region ($s={shade_s:g}$)" + "\n" + r"$\uparrow$ unbounded above",
        ha="center", va="top",
        fontsize=7, color="black",
        style="italic",
    )

    # ── Reference annotations ─────────────────────────────────────────────────
    hline = ax.axhline(
        0.0, color="black", linewidth=0.8, linestyle="--",
        label=r"$\gamma = 0$",
    )

    # Origin dot: b=0, c=0 → f(q) = aq³ (monotonic, sits on both boundaries)
    ax.scatter([0.0], [0.0], color="black", s=40, zorder=6)
    # Place annotation in the upper-left corner; that region is free of curves
    # (all boundary parabolas exit the top of the axes for |β| ≳ 7–9 when s is small,
    # and for large s they are deep in the negative γ area on that side).
    ax.annotate(
        r"$f(q)=aq^3$ (monotonic)" + "\n" + r"$(\beta,\gamma)=(0,0)$",
        xy=(0.0, 0.0),
        xytext=(-8.2, 3.6),
        fontsize=7,
        ha="left", va="center",
        arrowprops=dict(
            arrowstyle="->", color="black", lw=0.7,
            connectionstyle="arc3,rad=-0.15",
        ),
    )

    # ── Axes and labels ───────────────────────────────────────────────────────
    ax.set_xlim(beta_range)
    ax.set_ylim(gamma_min, gamma_max)
    ax.set_xlabel(r"$\beta = b/a$", fontsize=_FS_LABEL)
    ax.set_ylabel(r"$\gamma = c/a$", fontsize=_FS_LABEL)
    # Title is intentionally concise; full formula & variable definitions
    # belong in the figure caption.
    ax.set_title(
        r"Safe-zone boundary in normalised coefficient space $(\beta,\,\gamma)$",
        fontsize=_FS_TITLE,
    )
    ax.grid(True, alpha=_GRID_ALPHA)

    # ── Legend ────────────────────────────────────────────────────────────────
    # Proxy artist for the coloured boundary curves (colourbar carries the s values).
    curves_proxy = Line2D(
        [0], [0],
        color=cmap(norm(float(np.mean(s_vals)))), linewidth=_LW,
        label=r"Boundary $\gamma = (\beta^2 - 27s^2)/3$ (see colorbar)",
    )
    ax.legend(
        handles=[curves_proxy, mono_line, shade_patch, hline],
        fontsize=_FS_LEGEND, frameon=True,
        loc="lower right",
    )

    # ── Colourbar mapping s → viridis, ticks labelled "s=X (ε=Y)" ───────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_ticks(s_vals)
    cbar.set_ticklabels(
        [rf"$s={s:g}$  ($\varepsilon={Delta/s:.3g}$)" for s in s_vals],
        fontsize=_FS_LEGEND,
    )
    cbar.set_label(
        r"Noise scale $s = \Delta/\varepsilon$,  $\Delta=1$",
        fontsize=_FS_LABEL,
    )

    fig.tight_layout()
    return fig


# ── Figure 2 ─────────────────────────────────────────────────────────────────

def plot_safe_fraction_vs_epsilon(
    epsilon_grid: np.ndarray,
    grid_sizes: list[int],
    Delta: float = 1.0,
) -> plt.Figure:
    """Validates that K = 54a²s² + 6ac − 2b² ≥ 0 becomes harder to satisfy
    as ε increases (noise decreases), so the safe fraction is non-increasing.
    One curve per entry in grid_sizes: for half-width g, coefficients a, b, c
    run over all integers in [−g, g] with a ≠ 0.
    Prints per-grid-size tables to stdout."""
    # Guard against a trailing comma in the caller turning epsilon_grid into a tuple.
    epsilon_grid = np.asarray(epsilon_grid).ravel()

    colors = plt.cm.viridis(np.linspace(0, 1, len(grid_sizes)))

    print(
        f"\n[Fig 2] Safe fraction vs ε — {len(grid_sizes)} grid sizes: {grid_sizes}"
    )
    print(
        f"         ε grid: {len(epsilon_grid)} points from "
        f"{epsilon_grid[0]:.3f} to {epsilon_grid[-1]:.3f},  Δ={Delta:g}"
    )

    all_fractions: dict[int, np.ndarray] = {}

    for g in grid_sizes:
        # Build integer coefficient arrays for this half-width.
        a_g = np.array([a for a in range(-g, g + 1) if a != 0], dtype=float)
        b_g = np.arange(-g, g + 1, dtype=float)
        c_g = np.arange(-g, g + 1, dtype=float)

        # Broadcasting shapes: (n_a,1,1), (1,n_b,1), (1,1,n_c)
        A = a_g[:, None, None]
        B = b_g[None, :, None]
        C = c_g[None, None, :]
        n_total = a_g.size * b_g.size * c_g.size

        # Static part of K that does not depend on ε (precomputed once per grid).
        static_K = 6*A*C - 2*B**2
        A2 = A**2

        fracs = np.empty(len(epsilon_grid))
        for i, eps in enumerate(epsilon_grid):
            s = Delta / eps
            fracs[i] = float(np.mean(54 * A2 * s**2 + static_K >= 0))

        all_fractions[g] = fracs

        # Print a condensed table (~10 rows) for this grid size.
        print(
            f"\n  Grid ±{g:2d}: {n_total:>9,} points"
            f"  (a: {a_g.size:3d}, b: {b_g.size:3d}, c: {c_g.size:3d})"
        )
        print(f"  {'epsilon':>10}  {'s = Δ/ε':>10}  {'fraction_safe':>14}")
        print("  " + "-" * 38)
        for i in np.round(np.linspace(0, len(epsilon_grid) - 1, 10)).astype(int):
            print(
                f"  {epsilon_grid[i]:10.4f}  {Delta/epsilon_grid[i]:10.4f}"
                f"  {fracs[i]:14.6f}"
            )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for (g, fracs), col in zip(all_fractions.items(), colors):
        ax.plot(
            epsilon_grid, fracs,
            color=col, linewidth=_LW,
            label=rf"$|a|,|b|,|c| \leq {g}$",
        )

    ax.set_xlabel(r"$\varepsilon$", fontsize=_FS_LABEL)
    ax.set_ylabel("Fraction safe", fontsize=_FS_LABEL)
    ax.set_title(
        r"Fraction of integer $(a,b,c)$ grid with $K = 54a^2s^2 + 6ac - 2b^2 \geq 0$"
        r" vs $\varepsilon$",
        fontsize=_FS_TITLE,
    )
    ax.set_xlim(float(epsilon_grid[0]), float(epsilon_grid[-1]))
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=_FS_LEGEND, frameon=True, title="Coefficient range", title_fontsize=_FS_LEGEND)
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()
    return fig


# ── Figure 3 ─────────────────────────────────────────────────────────────────

def plot_gap_function(
    a: float,
    b: float,
    c: float,
    noise_scales: list[float],
    x_vals: np.ndarray,
) -> plt.Figure:
    """Validates the vertical-translation structure of I_MSE_cubic: all curves
    share the quadratic shape 27a²(x + b/3a)² but differ only in the constant
    offset 54a²s², proving the MSE gap is a pure vertical shift across noise scales."""
    colors = _tab10(len(noise_scales))
    worst_x = -b / (3.0 * a)

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, s in enumerate(noise_scales):
        I = _inner_poly(a, b, c, s, x_vals)
        k_val = _K(a, b, c, s)
        label = rf"$s = {s:.4g}$,  $k = {k_val:+.4g}$"
        ax.plot(x_vals, I, color=colors[idx], linewidth=_LW, linestyle="-", label=label)

        if k_val < 0:
            # Shade the unsafe band (I < 0) for curves that dip below zero
            unsafe = I < 0.0
            ax.fill_between(
                x_vals, I, 0.0,
                where=unsafe,
                alpha=0.18, color=colors[idx],
            )
            # Annotate the unsafe band with a text label (avoid crowding the legend)
            if unsafe.any():
                mid_idx = np.where(unsafe)[0][len(np.where(unsafe)[0]) // 2]
                ax.annotate(
                    "unsafe band",
                    xy=(x_vals[mid_idx], I[mid_idx] / 2),
                    fontsize=7, ha="center", va="center",
                    color=colors[idx],
                )

    # Reference lines
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(
        worst_x, color="dimgray", linewidth=1.0, linestyle="--",
        label=rf"Worst case $x = -b/(3a) = {worst_x:g}$",
    )

    # Polynomial label for title
    terms = [f"{a:g}q^3"]
    if b != 0:
        terms.append(f"{b:+g}q^2")
    if c != 0:
        terms.append(f"{c:+g}q")
    poly_str = "".join(terms)

    ax.set_xlabel(r"$x$", fontsize=_FS_LABEL)
    ax.set_ylabel(r"$I_{\mathrm{MSE}}(x)$", fontsize=_FS_LABEL)
    ax.set_title(
        rf"Gap inner polynomial $f(q)={poly_str}$: "
        r"vertical translations across noise scales",
        fontsize=_FS_TITLE,
    )
    ax.legend(fontsize=_FS_LEGEND, frameon=True, loc="upper right")
    ax.grid(True, alpha=_GRID_ALPHA)
    ax.set_xlim(float(x_vals[0]), float(x_vals[-1]))
    fig.tight_layout()
    return fig


# ── Figure 4 ─────────────────────────────────────────────────────────────────

def plot_monte_carlo_validation(
    tuples: list[tuple[float, float, float, float, float]],
    N: int = 10_000,
    seed: int = 4243,
) -> tuple[plt.Figure, dict]:
    """Validates the closed-form MSE gap −4s⁴ · I_MSE_cubic(q) against empirical
    differences computed from N raw Laplace samples; scatter of (theo, emp) pairs
    should lie on y = x.  A secondary panel shows relative error vs magnitude,
    exposing systematic bias that the scatter might hide.

    Raises AssertionError with full diagnostics (non-zero exit) if any tuple's
    empirical gap deviates from the theoretical value by more than 3 standard
    errors.  No figure is produced on failure."""
    rng = np.random.default_rng(seed)
    results: list[dict] = []

    for a, b, c, q, s in tuples:
        Z = rng.laplace(loc=0.0, scale=float(s), size=N)
        x = float(q) + Z
        target = float(a*q**3 + b*q**2 + c*q)

        g_naive = a*x**3 + b*x**2 + c*x
        # Unbiased Laplace estimator: f(x) − s² f″(x),  f″(x) = 6ax + 2b
        g_unbiased = g_naive - s**2 * (6.0*a*x + 2.0*b)

        d = (g_unbiased - target)**2 - (g_naive - target)**2   # paired difference
        gap_emp = float(np.mean(d))
        se = float(np.std(d, ddof=1) / np.sqrt(N))
        gap_theo = _gap_theory(a, b, c, q, s)
        k_val = _K(a, b, c, s)
        disc_se = float(abs(gap_emp - gap_theo) / se) if se > 0.0 else float("inf")

        results.append({
            "tuple": (a, b, c, q, s),
            "K": k_val,
            "gap_emp": gap_emp,
            "gap_theory": gap_theo,
            "se": se,
            "discrepancy_se": disc_se,
        })

    # ── Verification: must pass BEFORE creating any figure ───────────────────
    failed = [r for r in results if r["discrepancy_se"] > 3.0]
    if failed:
        lines = []
        for r in failed:
            a, b, c, q, s = r["tuple"]
            lines.append(
                f"  tuple (a={a:g}, b={b:g}, c={c:g}, q={q:.5g}, s={s:.5g})\n"
                f"    empirical  = {r['gap_emp']:.6g}  ±SE {r['se']:.3g}"
                f"  (3·SE = ±{3*r['se']:.3g})\n"
                f"    theory     = {r['gap_theory']:.6g}\n"
                f"    |emp−theo| = {abs(r['gap_emp']-r['gap_theory']):.3g}"
                f"  =  {r['discrepancy_se']:.2f} SE"
            )
        raise AssertionError(
            "Monte Carlo verification FAILED — discrepancy > 3 SE:\n"
            + "\n".join(lines)
        )

    # ── Build arrays ─────────────────────────────────────────────────────────
    gap_emp_arr = np.array([r["gap_emp"] for r in results])
    gap_theo_arr = np.array([r["gap_theory"] for r in results])
    se_arr = np.array([r["se"] for r in results])

    # Symlog linthresh = median |theoretical gap| (excluding exact zeros)
    nonzero_abs = np.abs(gap_theo_arr[np.abs(gap_theo_arr) > 1e-12])
    linthresh = float(np.median(nonzero_abs)) if nonzero_abs.size > 0 else 1.0

    colors = _tab10(len(tuples))
    scatter_labels = [
        rf"(a={a:g},b={b:g},c={c:g},q={q:.3g},s={s:.3g})  K={r['K']:+.3g}"
        for (a, b, c, q, s), r in zip(tuples, results)
    ]

    fig, (ax_sc, ax_re) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: scatter empirical vs theoretical ──────────────────────────
    all_vals = np.concatenate([gap_emp_arr, gap_theo_arr])
    finite = all_vals[np.isfinite(all_vals)]
    lo, hi = float(finite.min()), float(finite.max())
    pad = (hi - lo) * 0.05
    # Dense reference line so it renders smoothly on symlog axes
    ref = np.linspace(lo - pad, hi + pad, 400)
    ax_sc.plot(ref, ref, "k--", linewidth=0.8, label=r"$y = x$", zorder=0)

    for idx, r in enumerate(results):
        ax_sc.errorbar(
            r["gap_theory"], r["gap_emp"],
            yerr=3.0 * r["se"],
            fmt="o", color=colors[idx],
            linewidth=_LW * 0.6, capsize=4, capthick=1.0,
            label=scatter_labels[idx],
            zorder=5,
        )

    ax_sc.set_xscale("symlog", linthresh=linthresh)
    ax_sc.set_yscale("symlog", linthresh=linthresh)
    ax_sc.set_xlabel("Theoretical MSE gap", fontsize=_FS_LABEL)
    ax_sc.set_ylabel("Empirical MSE gap", fontsize=_FS_LABEL)
    ax_sc.set_title(
        rf"MC validation: empirical vs theoretical (N={N:,})",
        fontsize=_FS_TITLE,
    )
    ax_sc.legend(fontsize=6, frameon=True, loc="upper left")
    ax_sc.grid(True, alpha=_GRID_ALPHA)

    # ── Right panel: relative error vs |theoretical gap| ─────────────────────
    # Skip tuples where gap_theory ≈ 0 (relative error undefined)
    mask = np.abs(gap_theo_arr) > 1e-12
    abs_theo_m = np.abs(gap_theo_arr[mask])
    rel_err_m = (
        np.abs(gap_emp_arr[mask] - gap_theo_arr[mask]) / abs_theo_m
    )
    colors_m = [colors[i] for i in range(len(results)) if mask[i]]

    for ae, re, col in zip(abs_theo_m, rel_err_m, colors_m):
        ax_re.scatter(ae, re, color=col, s=80, zorder=5)

    # Reference: expected MC noise floor ≈ 3/√N
    ref_floor = 3.0 / np.sqrt(N)
    ax_re.axhline(
        ref_floor, color="black", linewidth=0.8, linestyle=":",
        label=rf"$3/\sqrt{{N}} \approx {ref_floor:.4f}$",
    )

    ax_re.set_xscale("log")
    ax_re.set_yscale("log")
    ax_re.set_xlabel(r"$|\mathrm{theoretical\ gap}|$", fontsize=_FS_LABEL)
    ax_re.set_ylabel(
        r"$|\mathrm{emp} - \mathrm{theo}|\,/\,|\mathrm{theo}|$",
        fontsize=_FS_LABEL,
    )
    ax_re.set_title("Relative error vs gap magnitude", fontsize=_FS_TITLE)
    ax_re.legend(fontsize=_FS_LEGEND, frameon=True)
    ax_re.grid(True, alpha=_GRID_ALPHA, which="both")

    fig.suptitle(
        f"Monte Carlo validation: cubic MSE gap, Laplace noise"
        f"  (N={N:,}, seed={seed})",
        fontsize=_FS_TITLE + 1,
    )
    fig.tight_layout()

    return fig, {
        "passed": True,
        "n_tuples": len(results),
        "linthresh": linthresh,
        "results": results,
    }
