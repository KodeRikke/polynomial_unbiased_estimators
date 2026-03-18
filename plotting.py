import sympy as sp
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from dp_estimators import EstimatorSystem
from noise_models import LaplaceNoiseModel

# -------------------------------------------------------
# NOT WORKING
# -------------------------------------------------------
"""
TO DO:
- Add documentation for the functions in this file.
- Plot for each value of epsilon and then value of q in the x-axis (instead of opposite).
- Add description of the file.
- Maybe make a clean script to remove all tex, aux, pdf files (and plots?) - the folder gets messy.
- Consider making it into a class? And then a script for running the plots?
- Add / change functions to accommodate plotting over q instead of epsilon, for different values of epsilon.
"""

def main():
    # Define the values:  this is the only place you should change values!

    # The values of q to plot / the range of q values. 
    q_values = [0, sp.Rational(1,4), sp.Rational(1,2), sp.Rational(3,4), 1] # values of q to plot (if we want to plot over epsilon instead of q)
    q_range = (0, 1) # range of q values for plotting (if we want to plot over q instead of epsilon)

    # The range of epsilon on the x-axis / the values of epsilon to plot.
    eps_range = (0.1, 5) # range of epsilon values for plotting
    eps_values = [sp.Rational(1,10), sp.Rational(1,2), 1, sp.Rational(3,2), 4] # [0.1, 0.5, 0.9, 1, 1.5, 4]

    # The degrees of the Chebyshev polynomials. Note that the variance and MSE gap grows with the degree, so the range of the plots might need adjusting for higher degrees.
    chebyshev_degrees = [2, 3, 6, 9] 

    # The fixed values of Delta for plotting; often Dela = 1 is used for simplicity, w.o. loss of generality, since the noise parameter Beta = Delta / epsilon. 
    Delta_values = [0.5, 1, 2]

    # Which metrics to plot The "ratio" means the unbiased / naive and the "relative" means (unbiased - naive) / naive. 
    metrics = ["variance_ratio", "variance_relative", "mse_ratio", "mse_relative"]

    # What each single plot should contain. 
    # The first part indicates the curves to plot together, and the second part indicates the fixed paramter for the curves. 
    # - "q_by_degree" means that for each degree of Chebyshev polynomials, we make one plot with the curves for different values of q, with epsilon on the x-axis.
    # - "degree_by_q" means that for each value of q, we plot the curves for different degrees of Chebyshev polynomials in the same plot, with epsilon on the x-axis.
    # - "epsilion_by_degree" means that for each degree of Chebyshev polynomials, we plot the curves for different values of epsilon in the same plot, with q on the x-axis.
    each_plot = ["q_by_degree", "degree_by_q", "epsilon_by_degree"] 

    # The path to save the plots. The code will create subfolders for variance and MSE plots.
    path_for_plots = "plots"

    plot_plots(
        q_values=q_values,
        q_range=q_range,
        eps_values=eps_values,
        eps_range=eps_range,
        chebyshev_degrees=chebyshev_degrees,
        Delta_values=Delta_values,
        metrics=metrics,
        each_plot=each_plot,
        path_for_plots=path_for_plots
    )

# -----------------------------------------------------------------------------------------------------------------------

def plot_plots(*, q_values, q_range, eps_values, eps_range, chebyshev_degrees, Delta_values, metrics, each_plot, path_for_plots):

    if "q_by_degree" in each_plot or "degree_by_q" in each_plot:
        # Create the plot data
        data_d_q = create_q_cheb_data(q_values, chebyshev_degrees)
        # Grid of epsilon values for plotting
        eps_grid = np.linspace(start=eps_range[0], stop=eps_range[1], num=300)

    if "q_by_degree" in each_plot:
        # create path for the plots:
        # - with curves for each q, 
        # - for each degree of Chebyshev polynomials, 
        # - with epsilon on the x-axis
        path_q_by_degree = f"{path_for_plots}/q_by_degree"

        by_degree_plots = {}
        # Create the plots for each degree of Chebyshev polynomials:
        for metric in metrics:
            if metric == "mse_ratio" or metric == "mse_relative":
                path = f"{path_q_by_degree}/mse"
            else:
                path = f"{path_q_by_degree}/variance"
            for n in chebyshev_degrees:
                for Delta_value in Delta_values:
                    print(f"Evaluating {metric} for Chebyshev degree {n} for different values of q...")
                    by_degree_plots.setdefault(metric, {})[n] = eval_q_for_fixed_degree(
                        curves_by_q=data_d_q["by_q"][q_val],
                        q_symbol=data_d_q["symbols"]["q"],
                        q_values=q_values,
                        base_subs={data_d_q["symbols"]["Delta"]: Delta_value},
                        epsilon_symbol=data_d_q["symbols"]["epsilon"],
                        eps_grid=eps_grid,
                        which_metric=metric,
                    )
                    print(f"Plotting {metric} for Chebyshev degree {n} for different values of q...")
                    plot(
                        x_label=f"$\\epsilon \\smallin [{eps_range[0]}, {eps_range[1]}]$",
                        y_label=metric.replace("_", " ").title(),
                        x_grid=eps_grid,
                        y_data=by_degree_plots[metric][n],
                        title=f"{metric.replace('_', ' ').title()} for Chebyshev Degree {n}, $\\Delta$ = {Delta_value}",
                        path=path
                    )
    
    if "degree_by_q" in each_plot:
        # create path for the plots:
        # - with curves for each q,
        # - for each degree of Chebyshev polynomials,
        # - with epsilon on the x-axis
        path_degree_by_q = f"{path_for_plots}/degree_by_q"

        by_q_plots = {}
        # Create the plots for each value of q:
        for metric in metrics:
            if metric == "mse_ratio" or metric == "mse_relative":
                path = f"{path_degree_by_q}/mse"
            else:
                path = f"{path_degree_by_q}/variance"
            for q_val in q_values:
                for Delta_value in Delta_values:
                    print(f"Evaluating {metric} for q = {q_val} for different degrees of Chebyshev polynomials...")
                    by_q_plots.setdefault(metric, {})[q_val] = eval_degree_for_fixed_q(
                        curves_by_degree=data_d_q["by_degree"][n],
                        degree_values=chebyshev_degrees,
                        base_subs={data_d_q["symbols"]["Delta"]: Delta_value},
                        epsilon_symbol=data_d_q["symbols"]["epsilon"],
                        eps_grid=eps_grid,
                        which_metric=metric,
                    )
                    print(f"Plotting {metric} for q = {q_val} for different degrees of Chebyshev polynomials...")
                    plot(
                        x_label=f"$\\epsilon \\smallin [{eps_range[0]}, {eps_range[1]}]$",
                        y_label=metric.replace("_", " ").title(),
                        x_grid=eps_grid,
                        y_data=by_q_plots[metric][q_val],
                        title=f"{metric.replace('_', ' ').title()} for q = {q_val}, $\\Delta$ = {Delta_value}",
                        path=path
                    )
    
    if "epsilon_by_degree" in each_plot:

        # create path for the plots:
        # - with curves for each epsilon,
        # - for each degree of Chebyshev polynomials,
        # - with q on the x-axis
        path_epsilon_by_degree = f"{path_for_plots}/epsilon_by_degree"

        # Create the data for plotting over q instead of epsilon, for different values of epsilon:
        data_eps = create_epsilon_cheb_data(eps_values, chebyshev_degrees)
        by_degree_plots_eps = {}

        for metric in metrics:
            if metric == "mse_ratio" or metric == "mse_relative":
                path = f"{path_epsilon_by_degree}/mse"
            else:
                path = f"{path_epsilon_by_degree}/variance"

            for n in chebyshev_degrees:
                for Delta_value in Delta_values:
                    print(f"Evaluating {metric} for Chebyshev degree {n} over q, for different values of epsilon...")
                    by_degree_plots_eps.setdefault(metric, {})[n] = eval_epsilon_for_fixed_degree(
                        curves_by_eps=data_eps["by_eps"][eps_val],
                        degree_values=chebyshev_degrees,
                        base_subs={data_eps["symbols"]["Delta"]: Delta_value},
                        q_symbol=data_eps["symbols"]["q"],
                        q_grid=np.linspace(start=q_range[0], stop=q_range[1], num=300),
                        which_metric=metric,
                    )
                    print(f"Plotting {metric} for Chebyshev degree {n} over q, for different values of epsilon...")
                    plot(
                        x_label=f"$q \\smallin [{q_range[0]}, {q_range[1]}]$",
                        y_label=metric.replace("_", " ").title(),
                        x_grid=np.linspace(start=q_range[0], stop=q_range[1], num=300),
                        y_data=by_degree_plots_eps[metric][n],
                        title=f"{metric.replace('_', ' ').title()} for Chebyshev Degree {n}, $\\Delta$ = {Delta_value}, for different $\\epsilon$",
                        path=path
                    )

"""
The build_system function initializes the EstimatorSystem with a Laplace noise model, using symbolic variables for Delta, epsilon, q, and X.
This allows for symbolic analysis of the estimators and their properties as functions of these parameters. 
Input: 
- None (the symbolic variables are defined within the function)
Output:
- system: an instance of EstimatorSystem initialized with the Laplace noise model and symbolic variables
- (Delta, epsilon, q, X): a tuple of the symbolic variables used in the system, which can be used for further symbolic computations and plotting.
"""
def build_system():
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

"""
The symbolic_chebyshev function computes the symbolic comparison of the naive and unbiased estimators for Chebyshev polynomials.
It computes it according to the initialized EstimatorSystem (using the compare-method) for specified degrees of Chebyshevs polynomials. 
Input:
- system: an instance of EstimatorSystem initialized with a noise model and symbolic variables
- q_symbol: the SymPy symbol representing q in the system
- chebyshev_degrees: a list of integers representing the degrees of the Chebyshev polynomials to analyze
Output:
- reports_degree: a dictionary keyed by the degree of the Chebyshev polynomial
"""
def symbolic_chebyshev(system, q_symbol, chebyshev_degrees):
    reports_degree = {}
    for n in chebyshev_degrees:
        f = sp.chebyshevt(n, q_symbol)
        report = system.compare(f=f)
        reports_degree[n] = report

    return reports_degree

"""
The subustitute function substitues a fixed NUMERIC value of either q or epsilon into the symbolic expression. 
It does this for all the expressions in the report keyed by Chebyshev degrees, and returns a dict keyed by degree containing the substituted expressions.

Input:
- subs_var: a string indicating which variable to substitute, either "q" or "epsilon"
- reports_degree: a dictionary keyed by degree containing the symbolic comparison report for each degree of Chebyshev polynomial
- q_symbol: the SymPy symbol representing q in the system, which is the variable to substitute
- q_value: the fixed numeric value to substitute for q in the expressions
- epsilon_symbol: the SymPy symbol representing epsilon in the system, which is the variable to substitute
- eps_value: the fixed numeric value to substitute for epsilon in the expressions
Output:
- subs: a dictionary keyed by degree containing the expressions with either q or epsilon substituted by the fixed numeric value
"""
def substitute(subs_var, reports_degree, q_symbol=None, q_value=None, epsilon_symbol=None, eps_value=None):
    if subs_var == "q":
        subs = {}
        for n, report in reports_degree.items():
            #variance_naive = report["naive"]["variance"]
            #variance_unbiased = report["unbiased"]["variance"]
            #variance_gap = report["variance_gap"]
            variance_ratio = report["variance_ratio"]
            variance_relative = report["variance_relative"]
            #mse_naive = report["naive"]["mse"]
            #mse_unbiased = report["unbiased"]["mse"]
            #mse_gap = report["mse_gap"]
            mse_ratio = report["mse_ratio"]
            mse_relative = report["mse_relative"]

            subs[n] = {
                #"variance_naive": variance_naive.subs(q_symbol, q_value),
                #"variance_unbiased": variance_unbiased.subs(q_symbol, q_value),
                #"variance_gap": variance_gap.subs(q_symbol, q_value),
                "variance_ratio": variance_ratio.subs(q_symbol, q_value),
                "variance_relative": variance_relative.subs(q_symbol, q_value),
                #"mse_naive": mse_naive.subs(q_symbol, q_value),
                #"mse_unbiased": mse_unbiased.subs(q_symbol, q_value),
                #"mse_gap": mse_gap.subs(q_symbol, q_value),
                "mse_ratio": mse_ratio.subs(q_symbol, q_value),
                "mse_relative": mse_relative.subs(q_symbol, q_value)
            }
        return subs
    elif subs_var == "epsilon":
        subs = {}
        for n, report in reports_degree.items():
            variance_ratio = report["variance_ratio"]
            variance_relative = report["variance_relative"]
            mse_ratio = report["mse_ratio"]
            mse_relative = report["mse_relative"]

            subs[n] = {
                "variance_ratio": variance_ratio.subs(epsilon_symbol, eps_value),
                "variance_relative": variance_relative.subs(epsilon_symbol, eps_value),
                "mse_ratio": mse_ratio.subs(epsilon_symbol, eps_value),
                "mse_relative": mse_relative.subs(epsilon_symbol, eps_value)
            }
        return subs
    else:
        raise ValueError("subs_var must be either 'q' or 'epsilon'")

"""
The create_q_cheb_data function creates the data needed for plotting the variance and MSE comparisons for different values of q and degrees of Chebyshev polynomials.
It builds the EstimatorSystem, using build_system(), computes the comparison reports using symbolic_chebushev(),
and then substitutes the fixed numeric values of q into the symbolic expressions using substitute().
Input:
- q_values: a list of numeric values of q to substitute into the expressions for plotting
- chebyshev_degrees: a list of integers representing the degrees of the Chebyshev polynomials to analyze
Output:
- A dictionary containing the substituted expressions, organized by Chebyshev degree and q value.
    The dictionary has the following structure:
{
    "symbols": {"Delta": Delta, "epsilon": epsilon, "q": q_symbol, "X": X},
    "by_degree": {
        n: {
            "variance_ratio": expression with q substituted,
            "variance_relative": expression with q substituted,
            "mse_ratio": expression with q substituted,
            "mse_relative": expression with q substituted
        }, 
        ...
    },
    "by_q": {
        q_val: {
            "variance_ratio": expression with q substituted,
            "variance_relative": expression with q substituted,
            "mse_ratio": expression with q substituted,
            "mse_relative": expression with q substituted
        },
        ...
    }
}
"""
def create_q_cheb_data(q_values, chebyshev_degrees):
    system, (Delta, epsilon, q_symbol, X) = build_system()
    reports_degree = symbolic_chebyshev(system, q_symbol, chebyshev_degrees)

    by_degree = {n: {} for n in chebyshev_degrees} # by_degree[n][q_val] -> expressions
    by_q = {q_val: {} for q_val in q_values} # by_q[q_val][n] -> expressions

    for q_val in q_values:
        subs_q = substitute("q", reports_degree, q_symbol=q_symbol, q_value=q_val)
        by_q[q_val] = subs_q
        for n in chebyshev_degrees:
            by_degree[n][q_val] = subs_q[n]

    return {
        "symbols": {"Delta": Delta, "epsilon": epsilon, "q": q_symbol, "X": X},
        "by_degree": by_degree,
        "by_q": by_q
    }

def create_epsilon_cheb_data(eps_values, chebyshev_degrees):
    system, (Delta, epsilon, q_symbol, X) = build_system()
    reports_degree = symbolic_chebyshev(system, q_symbol, chebyshev_degrees)

    by_degree = {n: {} for n in chebyshev_degrees} # by_degree[n][eps_val] -> expressions
    by_eps = {eps_val: {} for eps_val in eps_values} # by_eps[eps_val][n] -> expressions

    for eps_val in eps_values:
        subs_eps = substitute("epsilon", reports_degree, epsilon_symbol=epsilon, eps_value=eps_val)
        by_eps[eps_val] = subs_eps
        for n in chebyshev_degrees:
            by_degree[n][eps_val] = subs_eps[n]

    return {
        "symbols": {"Delta": Delta, "epsilon": epsilon, "q": q_symbol, "X": X},
        "by_degree": by_degree,
        "by_eps": by_eps
    }

"""
The eval_q_for_fixed_degree function plots the variance and MSE comparisons for different values of q, for a fixed degree of Chebyshev polynomial.
It takes the substituted expressions for a fixed degree of Chebyshev polynomial, evaluates them over a grid of epsilon values, and returns the evaluated data for plotting.
Input:
- curves_by_q: a dictionary keyed by q value containing the expressions for the variance and MSE comparisons for a fixed degree of Chebyshev polynomial
- q_symbol: the SymPy symbol representing q in the system, which is the variable that was substituted by a fixed numeric value in the expressions
- q_values: a list of numeric values of q that were substituted into the expressions for plotting
- base_subs: a dictionary containing the base substitutions for the other symbolic variables (e.g., Delta) that are needed to evaluate the expressions
- epsilon_symbol: the SymPy symbol representing epsilon in the system, which is the variable to evaluate over a grid of values for plotting
- eps_grid: a numpy array of epsilon values to evaluate the expressions on for plotting
- which_metric: a string indicating which metric to evaluate and plot, either "variance_ratio", "variance_relative", "mse_ratio", or "mse_relative" 
Output:
- A dictionary mapping each q value to the evaluated data for the specified metric.
"""
def eval_q_for_fixed_degree(curves_by_q, q_symbol, q_values, base_subs, epsilon_symbol, eps_grid, which_metric="variance_ratio"):
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
        out[f"q = {str(q_val)}"] = y   # label as string for legend
    return out

"""
The eval_degree_for_fixed_q function plots the variance and MSE comparisons for different degrees of Chebyshev polynomials, for a fixed value of q.
It takes the substituted expressions for a fixed value of q, evaluates them over a grid of epsilon values, and returns the evaluated data for plotting.
Input:
- curves_by_degree: a dictionary keyed by degree of Chebyshev polynomial containing the expressions for the variance and MSE comparisons for a fixed value of q
- degree_values: a list of integers representing the degrees of Chebyshev polynomials that were substituted into the expressions for plotting
- base_subs: a dictionary containing the base substitutions for the other symbolic variables (e.g., Delta) that are needed to evaluate the expressions
- epsilon_symbol: the SymPy symbol representing epsilon in the system, which is the variable to evaluate over a grid of values for plotting
- eps_grid: a numpy array of epsilon values to evaluate the expressions on for plotting
- which_metric: a string indicating which metric to evaluate and plot, either "variance_ratio", "variance_relative", "mse_ratio", or "mse_relative" 
Output:
- A dictionary mapping each degree of Chebyshev polynomial to the evaluated data for the specified metric.
"""
def eval_degree_for_fixed_q(curves_by_degree, degree_values, base_subs, epsilon_symbol, eps_grid, which_metric="variance_ratio"):
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

def eval_epsilon_for_fixed_degree(curves_by_degree, degree_values, base_subs, q_symbol, q_grid, which_metric="variance_ratio"):
    """
    curves_by_degree[n][which_metric] is a SymPy expr.
    Returns: out[n] -> numpy array
    """
    out = {}
    for n in degree_values:
        expr = curves_by_degree[n][which_metric]
        subs = dict(base_subs)   # copy
        subs["n"] = n 
        y = eval_over_q(expr, q_symbol, q_grid, subs)
        out[f"degree {n}"] = y   # label as string for legend
    return out

def plot(x_label, y_label, x_grid, y_data, title=None, path=None):
    plt.figure(figsize=(10, 6))
    for label, y_values in y_data.items():
        # plot 2 text labels indicating on which part of the 
        # x-axis the unbiased estimator has lower variance 
        # (if variance ratio < 1) or higher variance (if variance ratio > 1)
        # same with the relative variance plot, but the threshold is 0 instead of 1
        if y_label == "Variance Ratio" or y_label == "Mse Ratio":
            threshold = 1
            plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1)
            plt.text(x_grid[len(x_grid)//23], threshold*1.05, 'Naive has lower variance', color='black', fontsize=9)
            plt.text(x_grid[len(x_grid)//23], threshold*0.95, 'Unbiased has lower variance', color='black', fontsize=9)
        elif y_label == "Relative Variance" or y_label == "Relative Mse":
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

    # create path if it doesn't exist
    if path and not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(
        f"{path}/{title.replace('$', '').replace('\\', '').replace('=', '').replace(',', '').replace(' ', '_').replace('Variance', 'Var').replace('Chebyshev_Degree', 'Cheb_D').replace('/', 'over')}.png", 
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

def eval_over_q(expr: sp.Expr, q_symbol: sp.Symbol, q_grid: np.ndarray, subs_dict: dict) -> np.ndarray:
    """
    expr: SymPy expression in q
    q_symbol: the SymPy symbol representing q in the expression
    q_grid: a numpy array of q values to evaluate on
    subs_dict: e.g {epsilon: 1, Delta: 1}
    """
    eval_expr = sp.simplify(expr.subs(subs_dict))
    eval_func = sp.lambdify(q_symbol, eval_expr, modules=["numpy"])
    y = eval_func(q_grid)

    y = np.asarray(y, dtype=float) # force to be the same shape as q_grid

    if y.shape == (): # scalar output, make it an array of the same shape as q_grid
        y = np.full_like(q_grid, fill_value=float(y), dtype=float)
    elif y.shape == (1,): # length - 1 array
        y = np.full_like(q_grid, fill_value=float(y[0]), dtype=float)

    y[~np.isfinite(y)] = np.nan # clean inf/nan for plotting
    return y

if __name__ == "__main__":
    main()

