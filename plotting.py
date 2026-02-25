import sympy as sp
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from dp_estimators import EstimatorSystem, LaplaceNoiseModel

"""
This file contains code for plotting the variance gap of the naive and unbiased estimators.
It is done in the following ways:
1. A plot for each degree of Chebychev polynomials, d in {chebyshev_degrees}, we have:
    - x-axis: epsilon 
    - y-axis: the variance ratio, i.e. (var. of unbiased estimator) / (var. of naive estimator)
    - a line for each value of q in {q_values}
2. A plot for each degree of Chebychev polynomials, d in {chebyshev_degrees}, we have:
    - x-axis: epsilon 
    - y-axis: the relative variance, i.e. ((var. of unbiased estimator) - (var. of naive estimator)) / (var. of naive estimator)
    - a line for each value of q in {q_values}
3. A plot for each value of q in {q_values}, we have: 
    - x-axis: epsilon 
    - y-axis: the variance ratio, i.e. (var. of unbiased estimator) / (var. of naive estimator)
    - a line for each Chebyshev polynomial with d in {chebyshev_degrees}
4. A plot for each value of q in {q_values}, we have: 
    - x-axis: epsilon 
    - y-axis: the relative variance, i.e. ((var. of unbiased estimator) - (var. of naive estimator)) / (var. of naive estimator)
    - a line for each Chebyshev polynomial with d in {chebyshev_degrees}
"""
def main():
    # Define the values of q and the degrees of Chebychev polynomials:
    # This is the only place you should change values!
    #q_values = [-1, -sp.Rational(3,4), -sp.Rational(1,2), -sp.Rational(1,4), 0, sp.Rational(1,2), sp.Rational(3,4), 1]
    q_values = [0, sp.Rational(1,4), sp.Rational(1,2), sp.Rational(3,4), 1]
    chebyshev_degrees = [3, 6, 9] #, 10, 12]
    eps_range = (0.1, 10) # range of epsilon values for plotting
    Delta_values = [0.5, 1, 2] # fix Delta for plotting

    which = ["variance_ratio", "relative_variance"] # which metric to plot
    path_for_plots = "plots" # path to save the plots
    both = "by_degree" # either "both", "by_degree" or "by_q" to control which plots to create

# -----------------------------------------------------------------------------------------------------------------------
    # Create the plot data
    data = create_plot_data(q_values, chebyshev_degrees)
    eps_grid = np.linspace(start=eps_range[0], stop=eps_range[1], num=300) # grid of epsilon values for plotting

    if both in ("both", "by_degree"):
        by_degree_plots = {}
        # Create the plots for each degree of Chebyshev polynomials:
        for metric in which:
            for n in chebyshev_degrees:
                for Delta_value in Delta_values:
                    print(f"Evaluating {metric} for Chebyshev degree {n}...")
                    by_degree_plots.setdefault(metric, {})[n] = eval_for_fixed_degree(
                        data["by_degree"][n],
                        q_symbol=data["symbols"]["q"],
                        q_values=q_values,
                        base_subs={data["symbols"]["Delta"]: Delta_value},
                        epsilon_symbol=data["symbols"]["epsilon"],
                        eps_grid=eps_grid,
                        which_metric=metric,
                    )
                    print(f"Plotting {metric} for Chebyshev degree {n}...")
                    plot(
                        x_label=f"$\\epsilon \\smallin [{eps_range[0]}, {eps_range[1]}]$",
                        y_label=metric.replace("_", " ").title(),
                        x_grid=eps_grid,
                        y_data=by_degree_plots[metric][n],
                        title=f"{metric.replace('_', ' ').title()} for Chebyshev Degree {n}, $\\Delta$ = {Delta_value}",
                        path=path_for_plots
                    )
    
    if both in ("both", "by_q"):
        by_q_plots = {}
        # Create the plots for each value of q:
        for metric in which:
            for q_val in q_values:
                for Delta_value in Delta_values:
                    print(f"Evaluating {metric} for q = {q_val}...")
                    by_q_plots.setdefault(metric, {})[q_val] = eval_for_fixed_q(
                        data["by_q"][q_val],
                        degree_values=chebyshev_degrees,
                        base_subs={data["symbols"]["Delta"]: Delta_value},
                        epsilon_symbol=data["symbols"]["epsilon"],
                        eps_grid=eps_grid,
                        which_metric=metric,
                    )
                    print(f"Plotting {metric} for q = {q_val}...")
                    plot(
                        x_label=f"$\\epsilon \\smallin [{eps_range[0]}, {eps_range[1]}]$",
                        y_label=metric.replace("_", " ").title(),
                        x_grid=eps_grid,
                        y_data=by_q_plots[metric][q_val],
                        title=f"{metric.replace('_', ' ').title()} for q = {q_val}, $\\Delta$ = {Delta_value}",
                        path=path_for_plots
                    )
    
def build_system():
    # Defining symbolic values:
    Delta = sp.Symbol("Delta", real=True, positive=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)
    q = sp.Symbol("q", real=True)
    X = sp.Symbol("X", real=True)

    system = EstimatorSystem(
        noise_model=LaplaceNoiseModel(delta=Delta, epsilon=epsilon),
        q=q,
        x=X
    )
    return system, (Delta, epsilon, q, X)

def symbolic_chebyshev(system, q_symbol, chebyshev_degrees, simplify=False):
    # Compute the symbolic comparison for the Chebyshev polynomials:
    reports_degree = {}
    for n in chebyshev_degrees:
        f = sp.chebyshevt(n, q_symbol)
        report = system.compare(f=f, simplify=simplify)
        reports_degree[n] = report

    return reports_degree

def substitute_q(reports_degree, q_symbol, q_value):
    """
    For a fixed numeric q_value, substitute it into the the symbolic expression. 
    Return a dict keyed by degree containg the epsilon/Delta expression.
    """
    subs = {}
    for n, report in reports_degree.items():
        #variance_naive = report["naive"]["variance"]
        #variance_unbiased = report["unbiased"]["variance"]
        #variance_gap = report["variance_gap"]
        variance_ratio = report["variance_ratio"]
        relative_variance = report["relative_variance"]

        subs[n] = {
            #"variance_naive": variance_naive.subs(q_symbol, q_value),
            #"variance_unbiased": variance_unbiased.subs(q_symbol, q_value),
            #"variance_gap": variance_gap.subs(q_symbol, q_value),
            "variance_ratio": variance_ratio.subs(q_symbol, q_value),
            "relative_variance": relative_variance.subs(q_symbol, q_value)
        }
    return subs

def create_plot_data(q_values, chebyshev_degrees):
    system, (Delta, epsilon, q_symbol, X) = build_system()
    reports_degree = symbolic_chebyshev(system, q_symbol, chebyshev_degrees, simplify=False)

    # Define datastructure for the 4 kinds of plots to create:
    by_degree = {n: {} for n in chebyshev_degrees} # by_degree[n][q_val] -> expressions
    by_q = {q_val: {} for q_val in q_values} # by_q[q_val][n] -> expressions

    for q_val in q_values:
        subs_q = substitute_q(reports_degree, q_symbol, q_val)
        by_q[q_val] = subs_q
        for n in chebyshev_degrees:
            by_degree[n][q_val] = subs_q[n]

    return {
        "symbols": {"Delta": Delta, "epsilon": epsilon, "q": q_symbol, "X": X},
        "by_degree": by_degree,
        "by_q": by_q
    }

def eval_for_fixed_degree(curves_by_q, q_symbol, q_values, base_subs, epsilon_symbol, eps_grid, which_metric="variance_ratio"):
    """
    curves_by_q[q_val][which_metric] is a SymPy expr.
    Returns: out[q_val] -> numpy array
    """
    out = {}
    for q_val in q_values:
        expr = curves_by_q[q_val][which_metric]
        subs = dict(base_subs)   # copy
        subs[q_symbol] = q_val
        y = eval_over_epsilon(expr, epsilon_symbol, eps_grid, subs)
        out[str(q_val)] = y   # label as string for legend
    return out

def eval_for_fixed_q(curves_by_degree, degree_values, base_subs, epsilon_symbol, eps_grid, which_metric="variance_ratio"):
    """
    curves_by_degree[n][which_metric] is a SymPy expr.
    Returns: out[n] -> numpy array
    """
    out = {}
    for n in degree_values:
        expr = curves_by_degree[n][which_metric]
        subs = dict(base_subs)   # copy
        subs["n"] = n
        y = eval_over_epsilon(expr, epsilon_symbol, eps_grid, subs)
        out[f"degree {n}"] = y   # label as string for legend
    return out

def plot(x_label, y_label, x_grid, y_data, title=None, path=None):
    plt.figure(figsize=(10, 6))
    for label, y_values in y_data.items():
        # plot 2 text labels indicating on which part of the 
        # x-axis the unbiased estimator has lower variance 
        # (if variance ratio < 1) or higher variance (if variance ratio > 1)
        # same with the relative variance plot, but the threshold is 0 instead of 1
        if y_label == "Variance Ratio":
            threshold = 1
            plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1)
            plt.text(x_grid[len(x_grid)//23], threshold*1.05, 'Naive has lower variance', color='black', fontsize=9)
            plt.text(x_grid[len(x_grid)//23], threshold*0.95, 'Unbiased has lower variance', color='black', fontsize=9)
        elif y_label == "Relative Variance":
            threshold = 0
            plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1)
            plt.text(x_grid[len(x_grid)//23], 0.05, 'Naive has lower variance', color='black', fontsize=9)
            plt.text(x_grid[len(x_grid)//23], -0.05, 'Unbiased has lower variance', color='black', fontsize=9)
        plt.plot(x_grid, y_values, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{path}/plot_{title.replace('$', '').replace('\\', '').replace('=', '').replace(',', '').replace(' ', '_').replace('Variance', 'Var').replace('Chebyshev_Degree', 'Cheb_D').replace('/', 'over')}.png", 
        dpi=200, 
        bbox_inches='tight'
    )
    plt.close()

def eval_over_epsilon(expr: sp.Expr, epsilon_symbol: sp.Symbol, eps_grid: np.ndarray, subs_dict: dict) -> np.ndarray:
    """
    expr: SymPy expression in epsilon
    epsilon_symbol: the SymPy symbol representing epsilon in the expression
    eps_grid: a numpy array of epsilon values to evaluate on
    subs_dict: e.g {q: sp.Rational(1, 2), Delta: 1}
    """
    eval_expr = sp.simplify(expr.subs(subs_dict))
    eval_func = sp.lambdify(epsilon_symbol, eval_expr, modules=["numpy"])
    y = eval_func(eps_grid)

    y = np.asarray(y, dtype=float) # force to be the same shape as eps_grid

    if y.shape == (): # scalar output, make it an array of the same shape as eps_grid
        y = np.full_like(eps_grid, fill_value=float(y), dtype=float)
    elif y.shape == (1,): # length - 1 array
        y = np.full_like(eps_grid, fill_value=float(y[0]), dtype=float)

    y[~np.isfinite(y)] = np.nan # clean inf/nan for plotting
    return y

if __name__ == "__main__":
    main()

