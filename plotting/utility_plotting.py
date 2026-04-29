import sympy as sp
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg") # Use the TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
from pathlib import Path


__all__ = [
    "format_value",
    "metric_expr",
    "metric_expr_estimators",
    "evaluate_on_grid", 
    "add_reference_line",
    "save_filename",
    "plot_curves",
    "metric_folder"
]

# ----------------------------------------------------------------------
# --- Utility functions ---
def format_value(value):
    """
    Utility function to format a value for display in plot labels.
    If the value is a sympy expression, it converts it to a string.
    If the value is a float, it formats it to 3 significant digits.
    Otherwise, it converts it to a string directly.
    Input:
    - value: the value to format, which can be a sympy expression, a float,
    or any other type.
    Output:
    - A string representation of the value, formatted appropriately for 
    display in plot labels. 
    """
    if isinstance(value, sp.Basic):
        return f"{float(value):.3g}"
    if isinstance(value, float):
        return f"{value:.3g}"
    else: 
        return str(value)

# ----------------------------------------------------------------------
# --- Symbolic utilities ---
def metric_expr(reports_by_label, label, metric):
    """
    Function to extract the symbolic expression for a given metric (e.g., 
    "variance_ratio") from the report for a specific polynomial, 
    sorted by some "label", for example degree of Chebyshev polynomial 
    or name of the polynomial.
    Input:
    - reports_by_label: a dictionary mapping each label (e.g., degree of Chebyshev polynomial) 
        to the corresponding report (which is itself a dictionary mapping metric names to symbolic expressions).
    - label: the label of the report to extract from (e.g., degree of Chebyshev polynomial or name of polynomial).
    - metric: a string indicating which metric to extract from the report (e.g., "variance_ratio", "mse_relative", etc.)
    Output:
    - The symbolic expression corresponding to the specified metric for the given polynomial.
    """
    return reports_by_label[label][metric]

def metric_expr_estimators(reports_by_label, label, estimator, metric):
    """
    Extract the variance or MSE expression from the estimator.
    
    Input:
    - reports_by_label: a dict mapping noise dists to their respective symbolic reports, as returned by build_reports
    - label: the name of the polynomial to extract the metric for
    - estimator: the name of the estimator to extract the metric for
        (either "unbiased" or "naive" )
    - metric: the name of the metric to extract 
        (e.g., "variance", "mse")
    Output:
    - The symbolic expression for the requested metric
    """
    return reports_by_label[label][estimator][metric]



def evaluate_on_grid(expr, x_symbol, x_grid, subs_dict):
    """
    Function to evaluate a symbolic expression over a grid of values for 
    a specific variable.
    Input:
    - expr: the symbolic expression to evaluate
    - x_symbol: the symbolic variable in the expression, that are to be the x-axis
    - x_grid: a list of values to evaluate the expression at
    - subs_dict: a dictionary of other variable substitutions to apply
    Output:
    - A dictionary mapping each value in the grid to the evaluated 
    result of the expression.
    """
    eval_expr = sp.simplify(expr.subs(subs_dict))
    
    # Check that all symbols in the evaluated expression are substituted.
    # For easier interpretation of error-messages.
    leftover = eval_expr.free_symbols - {x_symbol}
    if leftover:
        raise ValueError(
            f"Expression still contains symbols {leftover}. "
            f"x_symbol={x_symbol}, subs_dict={subs_dict}, expr={eval_expr}"
        )
    
    eval_func = sp.lambdify(x_symbol, eval_expr, modules=["numpy"])
    y = eval_func(x_grid)

    y = np.asarray(y, dtype=float)

    if y.shape == (): 
        # scalar output, make it an array of the same shape as x_grid
        y = np.full_like(x_grid, fill_value=float(y), dtype=float)
    elif y.shape == (1,):
        # length - 1 array
        y = np.full_like(x_grid, fill_value=float(y[0]), dtype=float)
    # clean inf/nan for plotting
    y[~np.isfinite(y)] = np.nan
    return y

# ----------------------------------------------------------------------
# --- Plotting utilities ---
def add_reference_line(y_label, ax=None):
    """
    Function to add a reference line to the plot, indicating the threshold 
    for the metric being plotted. For the ratio metrics, the threshold is 1, 
    indicating where the unbiased estimator has the same variance/MSE as 
    the naive estimator. For the relative metrics, the threshold is 0, 
    indicating where the unbiased estimator has the same variance/MSE as 
    the naive estimator. The function also adds text labels to indicate 
    which side of the line corresponds to which estimator being better.
    Input:
    - x_grid: the grid of x values for the plot, used to position the text
    - y_label: the label for the y-axis, indicating the metric being plotted
    Output:
    - None 
    """
        # check if only one of the text-boxes stays within the y-limits of the plot
        # (hence the threshold line is either close to the top or bottom of the plot, 
        # or not in the plot at all, and only one of the two sides of the line is visible), 
        # in this case, add only the text-box, that corresponds to the side of the line that is visible, 
        # and let plt decide where to place it (so it won't be cut off by the limits of the plot).
    if ax is None:
        ax = plt.gca()
    
    if y_label in {"Variance Ratio", "MSE Ratio"}:
        threshold = 1.0
    elif y_label in {"Relative Variance", "Relative MSE", "Variance Relative", "MSE Relative"}:
        threshold = 0.0
    else:
        return

    ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
    ymin, ymax = ax.get_ylim()
    x_frac = 0.08

    # threshold position measured in the visible axis box
    y_frac = (threshold - ymin) / (ymax - ymin)

    # threshold below visible area
    if y_frac < 0:
        ax.text(
            x_frac, 0.15, "Naive lower",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="black"
        )
        return

    # threshold above visible area
    if y_frac > 1:
        ax.text(
            x_frac, 0.85, "Unbiased lower",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="black"
        )
        return

    # threshold is visible: add only labels that fit
    pad_frac = 0.04

    if y_frac + 2*pad_frac < 1:
        ax.text(
            x_frac, y_frac + pad_frac, "Naive lower",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color="black"
        )

    if y_frac - 2*pad_frac > 0:
        ax.text(
            x_frac, y_frac - pad_frac, "Unbiased lower",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="black"
        )

def plot_curves(x_label, y_label, x_grid, y_data, title=None, save_path=None, yscale="linear"):
    """
    Plot the curves for the given x and y data.
    """
    plt.figure(figsize=(10, 6))
    for label, y_values in y_data.items():
        plt.plot(x_grid, y_values, label=label)
    add_reference_line(y_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yscale(yscale)
    plt.title(title)
    plt.legend()
    plt.grid()
 
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, save_filename(title) + ".png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

# ----------------------------------------------------------------------
# --- File naming and organization utilities ---
def save_filename(text):
    """
    Generate a safe filename from the given text.
    """
    return (
        text.replace("$", "")
        .replace("\\", "")
        .replace("=", "")
        .replace(",", "")
        .replace(" ", "_")
        .replace("/", "_over_")
        .replace("Variance", "Var")
        .replace("Relative", "Rel")
        .replace("Chebyshev", "Cheb")
    )

def metric_folder(metric):
    """
    Generate a folder name for the given metric.
    """
    return "mse" if "mse" in metric else "variance"