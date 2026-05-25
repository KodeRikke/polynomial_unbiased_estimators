# mc_validation.py

Monte Carlo validation of closed-form variance/MSE-gap formulas for univariate
polynomials under Laplace and Gaussian differential-privacy noise.

The file lives at `mc/mc_validation.py` and is self-contained: it imports from
the project root (`dp_estimators.py`, `noise_models.py`,
`dp_calibration/gaussian.py`) but adds nothing to those modules.

---

## What the file does

The module has four layers:

**1. Theoretical formulas** (pure functions, no randomness)
Closed-form expressions for the variance gap, MSE gap, naive bias, noise
threshold `s*`, safety constant `K`, vertex `x*`, and unsafe band half-widths
for quadratic and cubic polynomials under Laplace noise.

**2. Closed-form MSE helpers for degree-4 polynomials** (pure functions, no randomness)
Two private helpers — `_quartic_mse_laplace` and `_quartic_mse_gaussian` —
compute the exact MSE (= Var, since the estimator is unbiased) of the unbiased
estimator for a quartic at the true parameter `q=0`, using Laplace or Gaussian
noise moments respectively.  Both are fully numpy-vectorised and support
broadcasting, enabling the exhaustive quartic sweep (Exp. G Part B) to be
computed in a single array operation over 12 500 × 25 parameter combinations.

The derivation uses:
- Laplace moments: `E[ξ^k] = k! s^k` (k even), 0 (k odd)
- Gaussian moments: `E[ξ^k] = (k−1)!! σ^k` (k even), 0 (k odd)
- The constant term `e` cancels from `g(q+ξ) − f(q)` at `q=0` and is not used

**3. Monte Carlo core and drivers**
- `mc_gaps` — the fundamental building block: draws `N` noise samples,
  evaluates both the naive estimator `h(X)` and the unbiased estimator `g(X)`
  (obtained from the existing `EstimatorSystem`), and returns empirical
  bias / variance / MSE / gap statistics with standard errors.
- `sweep_x` — varies the true value `x` at fixed noise scale `s`.
- `sweep_s_at_vertex` — varies `s` while fixing `x` at the worst-case point
  `x* = −b/(3a)`.
- `measure_band` — quantifies the unsafe interval (where naive wins) for a
  sub-threshold noise scale.
- `compare_noise` — matched-privacy Laplace vs Gaussian comparison at fixed
  `(ε, δ, Δ)`, calibrating the Gaussian `σ` via the Balle–Wang analytic
  mechanism.
- `sweep_epsilon_noise` — sweeps `ε` for the matched-privacy comparison,
  producing the privacy–utility curve.
- `experiment_Q` — exhaustive sweep over quadratic parameters for MSE-gap and
  Var-gap validation (analytical, no EstimatorSystem per config).
- `experiment_G_multidelta` — **Exp. G Part A**: closed-form matched-privacy
  MSE for one quartic across a range of ε values and δ ∈ {1e-10, …, 1e-2}.
  No Monte Carlo; uses `_quartic_mse_laplace` / `_quartic_mse_gaussian`.
- `experiment_G_quartic_sweep` — **Exp. G Part B**: closed-form ratio
  R = MSE_Gaussian / MSE_Laplace over a grid of 12 500 quartics at `q=0`.
  Fully vectorised; no Monte Carlo.

**4. Plotting helpers**
`plot_gaps_vs_x`, `plot_gap_vs_s_at_vertex`, `plot_unsafe_bands`,
`plot_noise_comparison`, `plot_Q_residuals`, `plot_Q_var_residuals`,
`plot_Q_loglog`, `plot_Q_collapse`, `plot_Q_gap_vs_a`,
`plot_S_coincidence`, `plot_P_inner`,
`plot_G_loglog`, `plot_G_multidelta`, `plot_G_quartic_distribution`
— each saves a PDF+PNG to `figures/` and returns the `matplotlib.Figure` object.

**5. Validation harness**
`run_all()` runs four experiments on fixed polynomial examples and prints a
PASS/FAIL summary. Every check tests whether the empirical result is within
3 standard errors of the closed-form prediction.

---

## What `run_plots.py` produces

Run from the project root:

```bash
python mc/run_plots.py
```

| Figure | File | Experiment |
|--------|------|------------|
| 1 | `gaps_vs_x` | Empirical var/MSE gap vs x (Exp. 2) |
| 2 | `gap_vs_s_at_vertex` | MSE gap at worst-case x* vs s (Exp. 3a) |
| 3 | `unsafe_bands` | Unsafe band widths, measured vs theory (Exp. 3b) |
| 4 | `noise_comparison` | Laplace vs Gaussian unbiased var/MSE (Exp. 4) |
| 5 | `Q_residuals` | Standardised MSE-gap residuals over quadratics (Exp. Q) |
| 6 | `Q_var_residuals` | Var-gap residuals ≡ 0 for quadratics (Exp. Q) |
| 7 | `Q_loglog` | Log-log \|MSE-gap\| vs s, slope ≈ 4 (Exp. Q) |
| 8 | `Q_collapse` | \|MSE-gap\| vs a²s⁴ collapse onto slope-1 line (Exp. Q) |
| 9 | `Q_gap_vs_a` | Signed MSE-gap vs a, theory parabola per s (Exp. Q) |
| 10 | `S_coincidence` | min_x I_Var = min_x I_MSE = K (Exp. S) |
| 11 | `P_inner` | I_MSE(x) overlay for several s (Exp. P) |
| 12 | `G_loglog` | Laplace vs Gaussian on log-log axes (Exp. G) |
| 13 | `G_multidelta` | Gaussian family for δ ∈ {1e-10,…,1e-2} (Exp. G Part A) |
| 14 | `G_quartic_dist` | R = MSE_Gauss/MSE_Lap over 12 500 quartics (Exp. G Part B) |

**Monte Carlo figures — runtime scales with `N`:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 12.
Each draws `N` noise samples per grid point; the default `N = 200 000` gives
standard errors of roughly 0.1 % of the signal for typical polynomial choices.
Reduce `N` for a fast preview; increase it for publication-quality error bands.

**Closed-form / theory-only figures — instant, `N`-independent:** 10, 11, 13, 14.
These evaluate analytical expressions or vectorised numpy formulas with no random
draws at all.  Results are fully deterministic and do not change with `N`.

---

## Coefficient convention

Coefficients are passed **lowest-degree-first**:

```
coeffs = [c0, c1, c2, ...]   →   f(x) = c0 + c1·x + c2·x² + …
```

For the theoretical formula functions the standard notation is used:

| Degree | Standard form | Mapping from `coeffs` |
|--------|---------------|-----------------------|
| 2 | `a·x² + b·x + c` | `a=coeffs[2]`, `b=coeffs[1]`, `c=coeffs[0]` |
| 3 | `a·x³ + b·x² + c·x + d` | `a=coeffs[3]`, `b=coeffs[2]`, `c=coeffs[1]`, `d=coeffs[0]` |
| 4 | `a·x⁴ + b·x³ + c·x² + d·x + e` | `a=coeffs[4]`, …, `e=coeffs[0]` |

---

## Noise conventions

| Mechanism | Parameter | Variance | Privacy |
|-----------|-----------|----------|---------|
| Laplace | `s = Δ/ε` | `2s²` | `(ε, 0)`-DP |
| Gaussian | `σ` from `calibrateAnalyticGaussianMechanism(ε, δ, Δ)` | `σ²` | `(ε, δ)`-DP |

The Gaussian `σ` is always calibrated with the Balle–Wang (ICML 2018) analytic
mechanism — **not** the Dwork–Roth `√(2 ln(1.25/δ))` formula.
For a scalar (univariate) query, `ℓ₁ = ℓ₂` sensitivity, so the same `Δ` is
passed to both mechanisms.

---

## Reproducibility

All randomness goes through a **local** `numpy.random.default_rng(seed)` — no
global state is touched. The same seed always produces identical results.

The closed-form experiments (Exp. G Parts A and B) use no random draws at all;
their results are fully deterministic given the coefficient and ε grids.

### Fixed seeds used in `run_all()`

`run_all()` uses `seed=0` throughout (the default). Each driver function
derives per-point child seeds from the master seed with
`rng.integers(0, 2**31, size=...)`, so every individual Monte Carlo run is
reproducible even if the grid size changes.

### Reproducing a specific result by hand

```python
import sys
sys.path.insert(0, "/path/to/Debiasing-in-Private-Statistics")   # project root

from mc.mc_validation import mc_gaps, cubic_mse_gap

# Non-monotonic cubic  f = x³ − 3x  at  x=0, s=0.3
coeffs = [0.0, -3.0, 0.0, 1.0]
r = mc_gaps(coeffs, x=0.0, noise_type="laplace", s=0.3, N=200_000, seed=0)

print(f"empirical mse_gap = {r['mse_gap']:.6f}")
print(f"theory    mse_gap = {cubic_mse_gap(1.0, 0.0, -3.0, 0.0, 0.3):.6f}")
```

### Reproducing the closed-form Exp. G experiments

```python
import numpy as np
from mc.mc_validation import (
    experiment_G_multidelta, plot_G_multidelta,
    experiment_G_quartic_sweep, plot_G_quartic_distribution,
)

# Part A — multi-δ curves for f = q⁴+q²+1
res_A = experiment_G_multidelta()   # uses defaults: q⁴+q²+1, ε∈[0.1,10], δ∈{1e-10,…,1e-2}
fig_A = plot_G_multidelta(res_A)    # saves figures/G_multidelta.{pdf,png}

# Part B — quartic sweep  (vectorised, ~1 s)
res_B = experiment_G_quartic_sweep()   # 12 500 quartics × 25 ε values
fig_B = plot_G_quartic_distribution(res_B)   # saves figures/G_quartic_dist.{pdf,png}
```

### Reproducing Experiment Q

```python
import numpy as np
from mc.mc_validation import (
    experiment_Q,
    plot_Q_residuals, plot_Q_var_residuals,
    plot_Q_loglog, plot_Q_collapse, plot_Q_gap_vs_a,
)

# Full grid, N=50 000 per (a, s) pair  (~5–10 min)
expQ = experiment_Q(N=50_000, seed=0)

# Or smaller for a quick check
expQ = experiment_Q(
    a_grid=[-2, -1, 1, 2],
    b_grid=[-2, 0, 2],
    c_grid=[-2, 0, 2],
    x_grid=np.linspace(-5, 5, 9),
    s_grid=[0.3, 0.75, 1.5],
    N=10_000, seed=0,
)

fig5 = plot_Q_residuals(expQ)       # figures/Q_residuals.{pdf,png}
fig6 = plot_Q_var_residuals(expQ)   # figures/Q_var_residuals.{pdf,png}
fig7 = plot_Q_loglog(N=50_000, seed=0)   # standalone log-log run
fig8 = plot_Q_collapse(expQ)        # figures/Q_collapse.{pdf,png}
fig9 = plot_Q_gap_vs_a(expQ)        # figures/Q_gap_vs_a.{pdf,png}

# Summary statistics
print(f"Configs: {len(expQ['records'])}")
print(f"Frac within 3 SE: {expQ['frac_pass_3se']*100:.1f}%")
print(f"Max residual: {expQ['max_n_se_mse']:.2f} SE")
print(f"Significant sign violations (z>2): {expQ['n_signif_pos']}")
```

### Reproducing a full sweep experiment

```python
import numpy as np
from mc.mc_validation import sweep_x, plot_gaps_vs_x

coeffs = [0.0, -3.0, 0.0, 1.0]   # f = x³ − 3x
result = sweep_x(coeffs, s=0.5, x_grid=np.linspace(-3, 3, 50),
                 N=200_000, seed=0)
fig = plot_gaps_vs_x(result)      # saves figures/gaps_vs_x.pdf
```

Every driver function accepts an explicit `seed` argument. Using the same seed
on the same grid reproduces identical output to within floating-point identity.

---

## Experiments 3a, 3b and S — cubic noise threshold and unsafe bands

These three figures all concern the same underlying question: for a cubic
`f(q) = a·q³ + b·q² + c·q + d` under Laplace noise, at which noise scales and
evaluation points does the unbiased estimator lose to the naive plug-in?

### Background — the safety threshold

The MSE gap and variance gap for a cubic are both proportional to the inner
polynomial:

```
I_MSE(x) = 54a²s² + 27a²x² + 18abx + 6ac + b²
I_Var(x) = 54a²s² + 18a²x²  + 12abx + 6ac
```

Both share the same minimum value (attained at `x* = −b/(3a)`):

```
K(s) = 54a²s² + 6ac − 2b²
```

The unbiased estimator wins everywhere (globally safe) iff `K(s) ≥ 0`, which
holds for all `s ≥ s*` where the noise threshold is:

```
s* = sqrt((b² − 3ac) / (27a²))    if b² > 3ac,  else 0  (always safe)
```

When `s < s*`, the cubic has an "unsafe band" centred on `x*` where the naive
estimator has lower MSE or lower variance.  The half-widths of those bands are:

```
MSE-unsafe half-width: sqrt(−K / (27a²))
Var-unsafe half-width: sqrt(−K / (18a²))   =  MSE half-width × sqrt(3/2) ≈ 1.2247
```

### `gap_vs_s_at_vertex` (Figure 2) — Monte Carlo

**What it shows:** the empirical MSE gap at the worst-case point `x* = −b/(3a)`,
plotted as a function of the noise scale `s`, with the theoretical curve
`cubic_mse_gap(a, b, c, x*, s)` overlaid.

The vertex `x*` is where `K(s)` is attained, so it is the hardest point for the
unbiased estimator to win.  Below the threshold `s*` the MSE gap at `x*` is
positive (naive wins there); above `s*` it becomes negative (unbiased wins
everywhere including `x*`).  The zero-crossing is marked with a vertical line.

**Method:** `sweep_s_at_vertex` builds one estimator per `s` value (since the
correction term `−s²f''(x)` depends on `s`) and draws fresh Laplace noise at
each `s`.  The default polynomial is `f = x³ − 3x` (a=1, b=0, c=−3), which has
`s* ≈ 0.577`.  Error bars are bootstrap SEs with B=200 resamples.

**What to look for:** the empirical points should track the theory curve closely
and cross zero near the marked `s*`.  The tight agreement validates the
closed-form `cubic_mse_gap` formula across the full range of noise scales.

### `unsafe_bands` (Figure 3) — Monte Carlo

**What it shows:** for a fixed noise scale `s < s*` (chosen so that unsafe bands
exist), the empirical var-gap and MSE-gap are plotted against `x` across a wide
range.  The measured unsafe interval (where the empirical gap > 0, meaning naive
wins) is shaded in red; the theoretically predicted band edges are marked in
green.

The plot makes two claims visible simultaneously:
1. The measured band width matches the theoretical half-width formula.
2. The variance-unsafe band is wider than the MSE-unsafe band by the factor
   `sqrt(3/2) ≈ 1.2247` (reported numerically below the plot).

**Method:** `measure_band` calls `sweep_x` internally (Monte Carlo at each x
value) and then identifies the unsafe region from the sign of the empirical gap.
The default uses `s = 0.3 < s* ≈ 0.577` for `f = x³ − 3x`.

**What to look for:** the shaded region should align closely with the green
dashed theory edges.  The printed width ratio should be within ~10 % of
`sqrt(3/2)` (the small discrepancy is due to the discrete x grid).

### `S_coincidence` (Figure 10) — **theory only, no Monte Carlo**

**What it shows:** three curves plotted on the same axes as a function of `s`:

1. `K(s) = 54a²s² + 6ac − 2b²`  (analytical formula, closed form)
2. `min_x I_Var(x)`  (numerical minimum of the variance inner polynomial over a
   fine grid of x values)
3. `min_x I_MSE(x)`  (numerical minimum of the MSE inner polynomial)

All three curves coincide exactly.  This is not a coincidence: the shared
minimum `K(s)` is the same for both inner polynomials by construction.  The
figure makes this structural identity visible and marks the zero-crossing at
`s*` with an annotated vertical line.

**Method:** purely analytical — no noise samples are drawn.  The minima are
computed numerically over `x ∈ [−10, 10]` with 4 000 points.  The analytical
and numerical curves should be indistinguishable to plotting precision.

**What to look for:** all three lines should overlap perfectly.  The zero of
`K(s)` is the threshold `s*`; below it all three quantities are negative
(unsafe bands exist), above it they are positive (globally safe).  This plot
provides visual proof that "MSE-safe" and "Var-safe" are the same condition,
i.e. the two safety regions coincide.

---

## Experiment Q — Exhaustive quadratic validation

### What it tests

For a quadratic `f(q) = a·q² + b·q + c` under Laplace noise with scale `s = Δ/ε`,
the unbiased estimator is `g(X) = h(X) − 2·a·s²` (a constant shift of the naive
plug-in `h(X) = f(X)`).  The closed-form results are:

| Quantity | Formula | Depends on |
|---|---|---|
| Var-gap = Var(g) − Var(h) | **0** | nothing — exact zero for any (a,b,c,s,x) |
| MSE-gap = MSE(g) − MSE(h) | **−4a²s⁴** | only `a` and `s` |

The MSE-gap is strictly negative for all `a ≠ 0` and `s > 0`, so the unbiased
estimator always has lower MSE than the naive estimator for every quadratic and
every evaluation point.  Experiment Q tests these two claims exhaustively across
a dense grid of polynomial parameters.

### Parameter grid

`experiment_Q` sweeps all combinations of:

| Parameter | Default values | Count |
|-----------|---------------|-------|
| `a` (leading coefficient) | {−5,…,5} \ {0} | 10 |
| `b` (linear coefficient) | {−5,−4,−2,0,2,4,5} | 7 |
| `c` (constant term) | {−5,−4,−2,0,2,4,5} | 7 |
| `x` (evaluation point) | linspace(−10, 10, 17) | 17 |
| `s` (noise scale) | {0.1, 0.3, 0.5, 0.75, 1.0, 1.5} | 6 |

Total: 10 × 7 × 7 × 17 × 6 = **49 980 configurations**.

Noise is drawn once per `(a, s)` pair and reused for all `(b, c, x)` combinations
at that pair, so the inner loop is pure arithmetic with no extra sampling.
Default `N = 50 000` samples per `(a, s)` pair.

### Standard errors

The MSE-gap is estimated as the sample mean of the paired differences
`Z_i = (g(X_i) − f(x))² − (h(X_i) − f(x))²`.  The SE is computed from the
CLT:  `SE = std(Z) / √N`.  No bootstrap is needed because the pairing already
controls for the correlation between the two estimators.

### PASS/FAIL criterion in `run_all()`

With ~50 000 configurations, roughly 0.3 % are expected to exceed 3 SE by chance
(false positives under the null).  The criterion is therefore:
- ≥ 99 % of configurations within 3 SE
- maximum residual ≤ 5 SE

A configuration with `emp_mse_gap > 0` and `z > 2` from zero (i.e. the empirical
value is significantly positive, not just a sign-flip near zero) is flagged as a
genuine violation.  The expected count is 0.

### The five Q-plots

**`Q_residuals` (figures/Q_residuals)** — Four panels, one per parameter
(a, b, c, x).  The y-axis is the standardised residual
`(emp_mse_gap − theory) / SE`.  Reference lines at 0 and ±3.
What to look for:
- All panels should show a horizontal cloud centred on 0, spread ≈ 1.
- The b, c, x panels should show no slope or structure, confirming the formula
  is truly independent of those parameters.
- Nearly all points should lie within ±3.

**`Q_var_residuals` (figures/Q_var_residuals)** — Single panel: empirical
Var-gap residual (= empirical Var-gap − 0) across all configuration indices.
All residuals should be numerically negligible (≈ 1e-12 level floating-point
noise), confirming that Var-gap = 0 exactly.

**`Q_loglog` (figures/Q_loglog)** — Log-log plot of |MSE-gap| vs s for a
single quadratic (a = 1, x = 0).  The empirical points should align with the
theoretical line `4a²s⁴` and the fitted log-log slope should be ≈ 4, confirming
the quartic scaling in s.

**`Q_collapse` (figures/Q_collapse)** — Log-log scatter of |MSE-gap| vs a²s⁴
for all 49 980 configurations, coloured by s.  Because MSE-gap = −4a²s⁴ the
points from all different (a, b, c, x, s) combinations should collapse onto the
single line `y = 4x` regardless of b, c, x.  Any off-line points are due to
sampling noise near a²s⁴ ≈ 0.  The plot also prints a report of sign-flipped
configurations and how many are statistically significant (z > 2 from zero).

**`Q_gap_vs_a` (figures/Q_gap_vs_a)** — Signed MSE-gap vs a for all
configurations, with the theory parabola −4a²s⁴ overlaid per s value.  All
empirical points (jittered for visibility) should lie on or near the downward
parabola for their respective s.  No point should be significantly above zero.

---

## Experiment G — Laplace vs Gaussian comparison (summary)

### Part 0 — `plot_G_loglog` (existing)

Uses `sweep_epsilon_noise` with Monte Carlo.  Sweeps ε for a degree-4 polynomial
at a fixed evaluation point `x=1`, showing variance and MSE on log-log axes with
crossover annotation.  Saves `figures/G_loglog.{pdf,png}`.

### Part A — `experiment_G_multidelta` + `plot_G_multidelta`

**Method:** closed form, `q=0`, no Monte Carlo.

For a fixed quartic (default: `f(q) = q⁴+q²+1`) and ε ∈ [0.1, 10], computes
the exact MSE of the unbiased estimator for:
- Laplace with `s = Δ/ε` (single reference curve, δ-independent)
- Gaussian calibrated via Balle–Wang for each of
  δ ∈ {1e-10, 1e-8, 1e-6, 1e-4, 1e-2}

Plot: log-log MSE vs ε.  Laplace = bold red.  Gaussian family = Blues gradient
(light = small δ, dark = large δ).  Crossovers with Laplace marked by vertical
dotted lines.

### Part B — `experiment_G_quartic_sweep` + `plot_G_quartic_distribution`

**Method:** closed form, `q=0`, no Monte Carlo.

Evaluates R(ε) = MSE_Gaussian / MSE_Laplace over a grid of degree-4 polynomials:
- a ∈ {−10,…,10} \ {0}  (20 values)
- b, c, d, e ∈ {−10, −5, 0, 5, 10}  (5 values each)
- Total: 20 × 5⁴ = 12 500 quartics

plus four named quartics highlighted individually:
T₄ = 8q⁴−8q²+1, q⁴, q⁴+q²+1, 10q⁴−10q².

Plot: log-log R vs ε.  Shows median, 10th-90th percentile band, min/max thin
dashed lines, R=1 parity reference, named quartics as individual curves.

**Key finding:** with δ=1e-10 the analytic Gaussian mechanism requires
σ/s ≈ 6.8 for all large ε, so the leading MSE term scales as
∝ (σ/s)⁸ ≈ 6.8⁸ ≈ 4.7×10⁶ relative to Laplace.  Consequently Laplace
dominates for quartics at q=0 across the entire ε grid at this δ.
The crossover seen in the Monte Carlo `G_loglog` experiment (at x=1) is
evaluation-point dependent, not a universal property of the mechanisms.

---

## Running in the terminal

Run from the **project root** (the directory that contains `dp_estimators.py`):

```bash
python mc/mc_validation.py
```

This executes `run_all()` via the `if __name__ == "__main__"` block and prints
a full PASS/FAIL table to stdout. Expected runtime is **2–4 minutes** with the
default `N = 200_000` samples per experiment point.

To reduce runtime for a quick sanity check, pass a smaller `N` directly in
Python (there is no CLI flag):

```python
# quick_check.py  — place in project root
from mc.mc_validation import run_all
run_all(N=10_000, seed=0)
```

```bash
python quick_check.py
```

Figures are saved automatically to `figures/` (created if absent) whenever a
`plot_*` function is called. The validation harness itself does not produce
figures — call the plot functions separately after running the drivers.
