# Monte Carlo Plots - Reproducibility Documentation

**Last Updated**: May 11, 2026  
**Repository State**: After systematic cleanup and archived obsolete runs

This document describes which Monte Carlo analysis outputs we have retained for thesis reproducibility.

## Retained Plot Outputs

### 1. **Symbolic Presentation Outputs** (`q_range_*_delta_*` folders)
- **Location**: `plots/monte_carlo/q_range_*/`
- **Folders**:
  - `q_range_1_5_delta_1e10/` - Wider q-range visualization
  - `q_range_15_delta_1e10/` - Main symbolic analysis (15-point q-grid, δ=1e-10)
  - `q_range_15_delta_1e6/` - Alternative δ specification (δ=1e-6)
- **Source**: `plotting/plot_symbolic_presentation.py`
- **Purpose**: Publication-quality symbolic/presentation figures showing how the naive and unbiased estimator ratios vary with q for representative polynomials
- **Contents**: Eight PNG figures per output directory
- **Interpretation**: The folder name does not encode monomial-vs-Chebyshev information; that distinction is inside the fixed panel examples in the script. The lower-degree figures compare quadratic, cubic, a coefficient-varied cubic, and Chebyshev T3. The higher-degree figures compare a degree-10 baseline, a coefficient-varied degree-10 polynomial, and Chebyshev T9. This gives a basis-sensitive view plus edge-case coefficient variations, rather than a monomial-only picture
- **Key Papers**: Used for thesis presentation of theoretical noise properties

### 2. **Systematic Degree Sweep** 
- **Location**: `plots/monte_carlo/systematic_degree_summary/`
- **Source Data**: `reports/monte_carlo/monte_carlo_systematic_degree_q15_20260511_010914.csv` (3.2M)
- **Source Script**: `plotting/plot_monte_carlo_study.py --plot-mode degree-summary`
- **Contents**:
  - `systematic_degree_heatmaps.png` - Degree-by-q heatmaps (MSE/variance by noise model)
  - `systematic_degree_trends.png` - Trend analysis across degrees
  - `systematic_degree_summary/gaussian/` and `systematic_degree_summary/laplace/` - Organized by noise model
- **Purpose**: One-dimensional degree sweep showing how polynomial degree affects noise scaling
- **Polynomials Tested**: 6,720 degree-varied polynomials (degrees 2-10, multiple coefficients)
- **Parameters**:
  - Q-grid: 15 points (-14 to 14)
  - Epsilons: 7 values (0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0)
  - Samples per config: 20,000

### 3. **Leading Order Noise Analysis**
- **Location**: `plots/monte_carlo/leading_order_summary.png`
- **Source Data**: `reports/analysis/leading_order_noise_terms.csv/.md`
- **Source Script**: `plotting/plot_leading_order_summary.py`
- **Computed By**: `scripts/leading_order_noise_terms.py`
- **Purpose**: Theoretical leading-order noise term visualization
- **Plot Convention**: Polynomial names are shown only on the leftmost subplot column to reduce repetition; the title now states the asymptotic limit as ε→∞ with Δ/ε→0 for Laplace and σ→0 for Gaussian
- **Status**: Active thesis research—kept for discussion of noise scaling theory

### 4. **Pairwise Criss-Cross Coefficient Sweep** (In Progress)
- **Status**: Sweep launched May 11, 2026 12:26:43 (PID 343163)
- **Location**: `reports/monte_carlo/monte_carlo_cubic_quartic_criss_cross_20260511_122643.csv` (being written)
- **Source Script**: `scripts/run_monte_carlo_study.py --study-mode cubic_quartic`
- **Purpose**: Capture pairwise coefficient interactions (beyond one-factor-at-a-time)
- **Design**:
  - Cubic polynomials: 97 (1 baseline + 21 one-factor + 75 pairwise pairs)
  - Quartic polynomials: 179 (1 baseline + 28 one-factor + 150 pairwise pairs)
  - Total: 276 polynomial specifications
- **Parameters**:
  - Coefficient pairs: {-2, -1, 0, 1, 2} (sparse grid—not full Cartesian)
  - Q-grid: 15 points (-14 to 14)
  - Epsilons: 7 values (0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0)
  - Samples per config: 20,000
- **Expected Plots** (after completion):
  - `plots/coefficient_criss_cross/{gaussian,laplace}/{cubic,quartic}/{ab,ac,ad,bc,bd,cd}/`
  - Per-pair heatmaps (MSE ratio + naive-win %) for each epsilon and noise model

### 5. **Older Coefficient Sweep** (Reference Only)
- **Location**: `plots/monte_carlo/monte_carlo_summary/`
- **Source**: `reports/monte_carlo/monte_carlo_cubic_quartic_ratios_20260511_114249.csv` (5.5M)
- **Status**: One-factor-at-a-time design (limited, superseded by criss-cross)
- **Purpose**: Reference for comparison; kept for historical context
- **Note**: This will be archived after criss-cross analysis is complete

## Archived Outputs (Not Used)

The following obsolete runs have been moved to `reports/monte_carlo/archive_20260511/` for safe recovery if needed:
- Pre-May 11 full sweeps (May 7–10)
- Legacy coefficient and degree analyses

## Reproducibility Notes

### For Thesis Narrative
1. Start with **symbolic presentation** (q_range folders) for theoretical foundation
2. Follow with **systematic degree sweep** for empirical degree dependence
3. Include **leading order analysis** for noise term interpretation
4. Use **pairwise criss-cross** (once complete) for coefficient interaction safety analysis

### Replication Commands
```bash
# Systematic degree sweep (already complete)
python3 scripts/run_monte_carlo_study.py --study-mode systematic_degree --samples 20000 --seed 1337

# Pairwise criss-cross sweep (in progress; restart if needed)
python3 scripts/run_monte_carlo_study.py --study-mode cubic_quartic --samples 20000 --seed 1337 --output reports/monte_carlo/monte_carlo_cubic_quartic_criss_cross_TIMESTAMP.csv

# Generate plots
python3 plotting/plot_monte_carlo_study.py --input reports/monte_carlo/monte_carlo_STUDY.csv --output-dir plots/monte_carlo --plot-mode [degree-summary|coefficient-criss-cross]
```

### Key Scripts
- **Main orchestrator**: `scripts/run_monte_carlo_study.py`
- **Plotting engine**: `plotting/plot_monte_carlo_study.py`
- **Symbolic analysis**: `plotting/plot_symbolic_presentation.py`
- **Leading order analysis**: `scripts/leading_order_noise_terms.py`
- **Calibration utilities**: `dp_calibration/{gaussian.py, SigmaFromEpsilon.py}`

### Dependencies
- Core simulation: `monte_carlo_simulation.py`
- Noise models: `noise_models.py`
- Estimators: `dp_estimators.py`
