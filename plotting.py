import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from dp_estimators import EstimatorSystem, LaplaceNoiseModel

"""
This file contains code for plotting the variance gap of the naive and unbiased estimators.
It is done in the following ways:
1. A plot for each degree of Chebychev polynomials, d in {5, 10, 15, 20}, we have:
    - x-axis: epsilon 
    - y-axis: the variance ratio, i.e. (var. of unbiased estimator) / (var. of naive estimator)
    - a line for each value of q in {-1, -0.5, 0, 0.5, 1}
2. A plot for each degree of Chebychev polynomials, d in {5, 10, 15, 20}, we have:
    - x-axis: epsilon 
    - y-axis: the relative variance, i.e. ((var. of unbiased estimator) - (var. of naive estimator)) / (var. of naive estimator)
    - a line for each value of q in {-1, -0.5, 0, 0.5, 1}
3. A plot for each value of q in {-1, -0.5, 0, 0.5, 1}, we have: 
    - x-axis: epsilon 
    - y-axis: the variance ratio, i.e. (var. of unbiased estimator) / (var. of naive estimator)
    - a line for each Chebyshev polynomial with d in {5, 10, 15, 20}
4. A plot for each value of q in {-1, -0.5, 0, 0.5, 1}, we have: 
    - x-axis: epsilon 
    - y-axis: the relative variance, i.e. ((var. of unbiased estimator) - (var. of naive estimator)) / (var. of naive estimator)
    - a line for each Chebyshev polynomial with d in {5, 10, 15, 20}
"""
def main():
    # Define the values of q and the degrees of Chebychev polynomials:
    q_values = [-sp.Rational(9,10), -sp.Rational(1, 2), 0, sp.Rational(1,10),sp.Rational(1, 3), 1] # same as [-1, -0.5, 0, 0.5, 1]
    chebyshev_degrees = [5, 10, 15, 20]
    eps_grid = np.linspace(0.1, 10, 300) # grid of epsilon values for plotting
    which = ["variance_ratio", "relative_variance"] # which metric to plot
    Delta_value = 1 # fix Delta for plotting

    # Create the plot data
    data = create_plot_data(q_values, chebyshev_degrees)

    by_degree_plots = {}
    # Create the plots for each degree of Chebyshev polynomials:
    # OBS! Will not plot multiple plots at a time, 
    # the terminal will just keep running;;; Fix this...!!
    for metric in which:
        for n in chebyshev_degrees:
            print(f"Evaluating {metric} for Chebyshev degree {n}...")
            by_degree_plots[n] = eval_for_fixed_degree(
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
                x_label="epsilon",
                y_label=metric.replace("_", " ").title(),
                x_grid=eps_grid,
                y_data=by_degree_plots[n],
                title=f"{metric.replace('_', ' ').title()} for Chebyshev Degree {n}, Delta = {Delta_value}"
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

def eval_by_q(expr_dict_by_q, q_symbol, chebyshev_degrees, subs_dict, epsilon_symbol, eps_grid, which_metric="variance_ratio"):
    eval_data = {}
    for q_val, degree_exprs in expr_dict_by_q.items():
        eval_data[q_val] = {}
        for n in chebyshev_degrees:
            expr = degree_exprs[which_metric]
            subs_dict[q_symbol] = q_val
            eval_data[q_val][n] = eval_over_epsilon(expr, epsilon_symbol, eps_grid, subs_dict)
    return eval_data

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
        # clean inf/nan for plotting
        y = np.array(y, dtype=float)
        y[~np.isfinite(y)] = np.nan
        out[str(q_val)] = y   # label as string for legend
    return out


def plot(x_label, y_label, x_grid, y_data, title=None):
    plt.figure(figsize=(10, 6))
    for label, y_values in y_data.items():
        plt.plot(x_grid, y_values, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def eval_over_epsilon(expr: sp.Expr, epsilon_symbol: sp.Symbol, eps_grid: np.ndarray, subs_dict: dict) -> np.ndarray:
    """
    expr: SymPy expression in epsilon
    epsilon_symbol: the SymPy symbol representing epsilon in the expression
    eps_grid: a numpy array of epsilon values to evaluate on
    subs_dict: e.g {q: sp.Rational(1, 2), Delta: 1}
    """
    eval_expr = sp.simplify(expr.subs(subs_dict))
    eval_func = sp.lambdify(epsilon_symbol, eval_expr, modules=["numpy"])
    return eval_func(eps_grid)

if __name__ == "__main__":
    main()

