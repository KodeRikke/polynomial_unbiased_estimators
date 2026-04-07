import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

"""OBS! the value b needs to be substituted with epsilon instead, 
as it is more common to use epsilon as the parameter for noise level in general. 
This means, b = \Delta / \epsilon, where \Delta is the sensitivity of the estimator.
For good measure I will change this to epsilon, if I need the plots."""

"""Create matrices of variance gaps used for heatmaps of polynomials of different degrees,
first used for Chebyshev polynomials, but can be used for any family of polynomials."""
def make_gap_matrix(q_val, b_grid, rows, for_type="gap"):
    degrees = [sp.degree(r["polynomial"], gen=q) for r in rows]
    G = np.zeros((len(b_grid), len(degrees)), dtype=float)
    R = np.zeros_like(G)

    for j, r in enumerate(rows):
        vn_expr = r["naive"]["variance"]
        vu_expr = r["unbiased"]["variance"]
        for i, b_val in enumerate(b_grid):
            vn = vn_expr.subs({q: q_val, b: b_val})
            vu = vu_expr.subs({q: q_val, b: b_val})
            if for_type == "gap":
                gap = float((vu - vn).evalf())
                G[i, j] = gap
            elif for_type == "ratio":
                ratio = float((vu / vn).evalf()) if vn != 0 else np.nan
                G[i, j] = ratio
            elif for_type == "relative_gap":
                rel_gap = float(((vu - vn) / vn).evalf()) if vn != 0 else np.nan
                G[i, j] = rel_gap
            else :
                raise ValueError("for_type must be 'gap', 'ratio', or 'relative_gap'")

    return np.array(b_grid, dtype=float), np.array(degrees, dtype=int), G

"""Create heatmaps of variance gaps for a given family of polynomials,
with log scale and sign, i.e., log10(|var_unbiased - var_naive|) and sign"""
"""OBS! Something is wrong with the log scale; the range is of, very high numbers 
and no negative sign, which is not what I expected."""
def plot_heatmap_log(q_vals, b_grid, rows):
    for q_val in q_vals:
        b_vals, degs, G = make_gap_matrix(q_val, b_grid, rows, for_type="gap")

        eps = 1e-300
        logabs = np.log10(np.abs(G) + eps)
        sign = np.sign(G)

        plt.figure(figsize=(10, 5))
        plt.imshow(logabs, aspect="auto", origin="lower")
        plt.xticks(range(len(degs)), degs)
        plt.yticks([0, len(b_vals)//2, len(b_vals)-1],
                   [f"{b_vals[0]:.2g}", f"{b_vals[len(b_vals)//2]:.2g}", f"{b_vals[-1]:.2g}"])
        plt.colorbar(label=r"$\log_{10}(|\mathrm{VarGap}|)$")
        plt.title(f"Log-magnitude of variance gap at q={q_val}")
        plt.xlabel("degree")
        plt.ylabel("b (log grid)")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.imshow(sign, aspect="auto", origin="lower", vmin=-1, vmax=1)
        plt.xticks(range(len(degs)), degs)
        plt.yticks([0, len(b_vals)//2, len(b_vals)-1],
                   [f"{b_vals[0]:.2g}", f"{b_vals[len(b_vals)//2]:.2g}", f"{b_vals[-1]:.2g}"])
        plt.colorbar(label="sign (−1 negative, +1 positive)")
        plt.title(f"Sign of variance gap at q={q_val}")
        plt.xlabel("degree")
        plt.ylabel("b (log grid)")
        plt.show()

"""Create heatmaps of variance ratios for a given family of polynomials,
with log scale, i.e., log10(|var_unbiased / var_naive|)"""
def make_ratio_plot(q_values, b_grid, rows):
    for q_val in q_values:
        b_vals, degs, R = make_gap_matrix(q_val, b_grid, rows, for_type="ratio")

        plt.figure(figsize=(10, 5))
        plt.imshow(np.log10(np.abs(R)), aspect="auto", origin="lower")
        plt.xticks(range(len(degs)), degs)
        plt.yticks([0, len(b_vals)//2, len(b_vals)-1],
                   [f"{b_vals[0]:.2g}", f"{b_vals[len(b_vals)//2]:.2g}", f"{b_vals[-1]:.2g}"])
        plt.colorbar(label=r"$\log_{10}(|\mathrm{VarRatio}|)$")
        plt.title(f"Log scale of variance ratio at q={q_val}")
        plt.xlabel("degree")
        plt.ylabel("b (log grid)")
        plt.show()

"""Create heatmaps of relative variance gaps for a given family of polynomials,
with log scale, i.e., log10(|(var_unbiased - var_naive) / var_naive|)"""
def plot_relative_gap_heatmap(q_vals, b_grid, rows):
    for q_val in q_vals:
        b_vals, degs, G = make_gap_matrix(q_val, b_grid, rows, for_type="relative_gap")

        plt.figure(figsize=(10, 5))
        plt.imshow(np.log10(np.abs(G) + 1e-300), aspect="auto", origin="lower")
        plt.xticks(range(len(degs)), degs)
        plt.yticks([0, len(b_vals)//2, len(b_vals)-1],
                   [f"{b_vals[0]:.2g}", f"{b_vals[len(b_vals)//2]:.2g}", f"{b_vals[-1]:.2g}"])
        plt.colorbar(label=r"$\log_{10}(|\mathrm{RelativeVarGap}|)$")
        plt.title(f"Log scale of relative variance gap at q={q_val}")
        plt.xlabel("degree")
        plt.ylabel("b (log grid)")
        plt.show()

# Example usage:
# q values to consider
q_values = [0.1] #[-1, -0.5, -0.1, 0, 0.1, 0.5, 1]
# b values to consider (log scale)
b_grid = np.logspace(-3, 3, 50)  # from 10^-3 to 10^3
# rows = chebyshev_for_heatmap  # this should be defined elsewhere, containing the variance expressions for the polynomials

plot_heatmap_log(q_values, b_grid, chebyshev_for_heatmap)
make_ratio_plot(q_values, b_grid, chebyshev_for_heatmap)
plot_relative_gap_heatmap(q_values, b_grid, chebyshev_for_heatmap)
