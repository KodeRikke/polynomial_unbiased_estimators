import sympy as sp
from report_formatter import ReportFormatter
from print_LaTeX import build_latex_document
from noise_models import NoiseModel
from typing import Union


__all__ = ["EstimatorSystem", 
           "ComparisonReport", 
           "EstimatorContext", 
           "EstimatorAnalyzer"
           ]

"""
This module defines the main class for the estimator system.

The four main classes are:
- EstimatorSystem: the main interface for users to interact with the system, providing methods for getting estimators and comparing them.
- ComparisonReport: responsible for generating a report comparing the naive and unbiased estimators for a given polynomial function.
- EstimatorContext: responsible for managing different noise/estimation strategies and providing methods for getting naive and unbiased estimators.
- EstimatorAnalyzer: responsible for analyzing the properties of estimators, such as calculating mean and variance, using the noise model's moment calculations.

It follows the Facade design pattern, where the EstimatorSystem class provides a simplified interface to the complex subsystem of noise models, estimation strategies, and analysis.
Further it follows the Strategy design pattern, where the EstimatorContext class manages different estimation strategies (naive, unbiased, which type of noise), and it maintains a reference to one of the concrete strategies,
and interacts with this object only through the interface defined by the abstract strategy (NoiseModel).
The EstimatorAnalyzer class manages the analysis of the estimators (mean, variance, MSE). It serves the same purpose as the "Context" in the Strategy pattern, only here the estimator and analysis of the estimator are split into two classes,
The ComparisonReport class is responsible for generating a report comparing the naive and unbiased estimators for a given polynomial function, by calculating their means, variances, MSE and the gaps between them.

The three modules, "noise_models", "print_LaTeX", and "report_formatter" are separate from the main estimator system.
"""

# --- EstimatorSystem ---
class EstimatorSystem:
    """
    The EstimatorSystem class serves as a facade for interacting with the estimator and analyzer. 
    It initializes the context and analyzer, and provides methods for getting estimators and comparing them.
    When initializing the EstimatorSystem, the following needs to be provided:
        noise_model: an instance of a NoiseModel (e.g., LaplaceNoiseModel)
        q: a variable s.a. an int, float or sympy symbol representing the original statistic
        x: a variable s.a. an int, float or sympy symbol representing the observed statistic, x = q + noise 
    It changes q and x into strings, s.t they can be changed to sympy symbols in the context and analyzer."""
    def __init__(self, noise_model: NoiseModel, q: Union[int, float, sp.Symbol], x: Union[int, float, sp.Symbol]):
        self.noise_model = noise_model
        self.q = sp.Symbol(str(q), real=True)
        self.x = sp.Symbol(str(x), real=True)
        self.context = EstimatorContext(noise_model, str(q), str(x)) # initialize context
        self.analyzer = EstimatorAnalyzer(noise_model, str(q), str(x)) # initialize analyzer

    """
    The estimator method takes a polynomial function f in q and returns the corresponding estimator g in x, depending on the specified biasedness.

    Input: f is a polynomial function in q, biasedness is either "naive" or "unbiased".
    Output: the corresponding estimator g in x.
    """
    def estimator(self, f, biasedness="naive"):
        if biasedness == "naive":
            return self.context.naive(f)
        elif biasedness == "unbiased":
            return self.context.unbiased(f)
        else:
            raise ValueError("biasedness must be either 'naive' or 'unbiased'")
    
    """
    The compare method takes a polynomial function f in q and generates a report comparing the naive and unbiased estimators for f, by calculating their means, variances, MSE and the gaps between them.

    Input: f is a polynomial function in q.
    Output: a report comparing the naive and unbiased estimators for f.
    """
    def compare(self, f):
        return ComparisonReport(self, f=f).report()

    """
    The compare_more method takes a list of polynomial functions fs in q and generates a report comparing the naive and unbiased estimators for each function in fs.

    Input: fs: a list of polynomial functions in q,
           name_fn: an optional function that takes a polynomial and its index and returns a name for it.
    Output: a list of comparison reports for each polynomial in fs, optionally with names.
    """
    def compare_more(self, fs, name_fn=None):
        rows = []
        for i, f in enumerate(fs):
            res = self.compare(f=f)
            if name_fn:
                res["name"] = name_fn(f, i)
            rows.append(res)
        return rows
    
    """
    The summary_report method takes a polynomial function f in q and generates a summary report 
    comparing the naive and unbiased estimators for f, by calculating their means, variances, MSE and the gaps between them.
    It uses the class ReportFormatter from the module report_formatter and the method render_summary for formatting the report
    for display in the terminal, with the specified notation and compactness.

    Input: f: a polynomial function in q,
           notation: either "beta" or "grouped", for either keeping the noise parameters as beta = Delta/epsilon (best for readability in the terminal), or substituting back to Delta / epsilon,
           compact: a boolean indicating whether to use compact formatting.
    Output: a formatted summary report comparing the naive and unbiased estimators for f, suitable for display in the terminal.
    """
    def summary_report(self, f, *, notation="beta", compact=False):
        report = self.compare(f)
        noise_model = self.context.noise_model
        if hasattr(noise_model, "delta") and hasattr(noise_model, "epsilon"):
            delta = noise_model.delta
            epsilon = noise_model.epsilon
            fmt = ReportFormatter(delta=delta, epsilon=epsilon)
        else:
            raise ValueError("summary_compare currently expect a Laplace-like model.")
        return fmt.render_summary(report, notation=notation, compact=compact)

    """
    The latex_compare method takes a polynomial function f in q and generates a LaTeX report comparing the naive and unbiased estimators for f, by calculating their means, variances, MSE and the gaps between them.
    It uses the class ReportFormatter from the module report_formatter and the method render_latex for formatting the report for display in a LaTeX document.

    Input: f: a polynomial function in q,
           notation: either "beta" or "grouped", for either keeping the noise parameters as beta = Delta/epsilon, or substituting back to Delta and epsilon for better readability in the LaTeX document,
           compact: a boolean indicating whether to use compact formatting.
    Output: a formatted LaTeX report comparing the naive and unbiased estimators for f, suitable for inclusion in a LaTeX document.
    """
    def latex_compare(self, f, notation="grouped", compact=False):
        report = self.compare(f)
        noise_model = self.context.noise_model
        if hasattr(noise_model, "delta") and hasattr(noise_model, "epsilon"):
            delta = noise_model.delta
            epsilon = noise_model.epsilon
            fmt = ReportFormatter(delta=delta, epsilon=epsilon)
        else:
            raise ValueError("latex_compare currently expect a Laplace-like model.")
        return fmt.render_latex(report, notation=notation, compact=compact)

    """
    The pdf_report method takes a polynomial function f in q and generates a PDF report comparing the naive and unbiased estimators for f, by calculating their means, variances, MSE and the gaps between them.
    It uses the class ReportFormatter from the module report_formatter and the method render_pdf for formatting the report for display in a PDF document.

    Input: f: a polynomial function in q,
           output_stem: a string representing the stem of the output file name (without extension) for the PDF report,
           notation: either "beta" or "grouped", for either keeping the noise parameters as beta = Delta/epsilon  or substituting back to Delta / epsilon,
           compact: a boolean indicating whether to use compact formatting.
    Output: a formatted PDF report comparing the naive and unbiased estimators for f, suitable for inclusion in a PDF document.
    """
    def pdf_report(self, f, output_stem, *, notation="grouped", title=None, compact=False):
        latex_body = self.latex_compare(f, notation=notation, compact=compact)
        build_latex_document(latex_body, output_stem, title=title)

# --- ComparisonReport ---
class ComparisonReport:
    """ The ComparisonReport class is responsible for generating a report comparing the naive and unbiased estimators for a given polynomial function.
    It computes the estimators, their means and variances, and the gaps between them and the Mean Squared Error (MSE).
    When initializing the ComparisonReport, the following needs to be provided:
        system: an instance of EstimatorSystem to use for calculations
        f: a polynomial function in q of which to compare the estimators """
    def __init__(self, system: EstimatorSystem, f: sp.Expr):
        self.system = system
        self.f = sp.sympify(f)

    """ 
    The report method generates a report comparing the naive and unbiased estimators for the polynomial function f, by calculating their means, variances, MSE and the gaps between them.
    
    Input: none (the polynomial function f is provided at initialization).
    Output: a report comparing the naive and unbiased estimators for f. 
            The report is in the form of a RAW dictionary containing the polynomial f, the naive and unbiased estimators, 
            their means, variances, MSE and the gaps between them, the ratio and relative variance.
    """
    def report(self):
        system = self.system
        f = self.f

        g_naive = system.estimator(f, biasedness="naive")
        g_unbiased = system.estimator(f, biasedness="unbiased")

        mean_naive = system.analyzer.mean(g_naive)
        mean_unbiased = system.analyzer.mean(g_unbiased)
        variance_naive = system.analyzer.variance(g_naive)
        variance_unbiased = system.analyzer.variance(g_unbiased)
        mse_naive = system.analyzer.mse(g_naive, f)
        mse_unbiased = system.analyzer.mse(g_unbiased, f)

        bias_naive = system.analyzer.bias(g_naive, f)

        return {
            "polynomial": f,
            "naive": {
                "estimator": g_naive,
                "mean": mean_naive,
                "variance": variance_naive,
                "mse": mse_naive,
                "bias": bias_naive
            },
            "unbiased": {
                "estimator": g_unbiased,
                "mean": mean_unbiased,
                "variance": variance_unbiased,
                "mse": mse_unbiased
            },
            "mean_gap": mean_unbiased - mean_naive,
            "variance_gap": variance_unbiased - variance_naive, # if NEGATIVE, then unbiased has lower variance
            "variance_ratio": sp.oo if variance_naive.is_zero else variance_unbiased / variance_naive, # if LESS than 1, then unbiased has lower variance 
            "variance_relative": sp.Integer(0) if variance_naive.is_zero else (variance_unbiased - variance_naive) / variance_naive, # if negative, then unbiased has lower variance
            "mse_gap": mse_unbiased - mse_naive,
            "mse_ratio": sp.oo if mse_naive.is_zero else mse_unbiased / mse_naive,
            "mse_relative": sp.Integer(0) if mse_naive.is_zero else (mse_unbiased - mse_naive) / mse_naive,
            "bias_naive_squared": bias_naive**2
        }

# --- Context EstimatorContext ---
class EstimatorContext:
    """The context class EstimatorContext is responsible for managing different noise /
    estimation strategies. It passess the estimation task to the selected strategy.
    When initializing the EstimatorContext, the following needs to be provided:
        noise_model: an instance of a NoiseModel (e.g., LaplaceNoiseModel)
        q: a variable s.a. an int, float or sympy symbol representing the original statistic
        x: a variable s.a. an int, float or sympy symbol representing the observed statistic, x = q + noise 
    It changes q and x into sympy symbols, s.t they can be used in the noise model and analyzer."""
    def __init__(self, noise_model: NoiseModel, q: str, x: str):
        self.noise_model = noise_model
        self.q = sp.Symbol(q, real=True)
        self.x = sp.Symbol(x, real=True)

    """ 
    The naive estimator is a plug-in estimator. It is calculated by substituting q with x in the polynomial function f, s.t the function stays the same, i.e. g = f.

    Input: f is a polynomial function in q.
    Output: the plug-in estimator g(x) = f(q), which is just the substitution of q with x, s.t the function stays the same, i.e. g = f.
    """
    def naive(self, f):
        f = sp.sympify(f)
        return f.subs(self.q, self.x) # naive is just substitution
        # .subs is used to substitute a variable or expression with a specified 
        # value or another expression in a symbolic mathematical expression

    """ 
    The unbiased estimator is calculated using the noise model's unbiased_transform method.

    Input: f is a polynomial function in q.
    Output: the unbiased estimator g(x), which is calculated using the noise model's unbiased_transform method, 
    implemented according to the specific noise distribution."""
    def unbiased(self, f):
        f = sp.sympify(f)
        f_in_x = f.subs(self.q, self.x) # substitute q with x in f to turn f(q) into f(x) = f(q + noise)
        g = self.noise_model.unbiased_transform(f_in_x, self.x)
        return sp.expand(g)

# --- Context EstimatorAnalyzer ---
class EstimatorAnalyzer:
    """The analyzer class EstimatorAnalyzer is responsible for analyzing the properties
    of estimators, such as calculating variance.
    When initializing the EstimatorAnalyzer, the following needs to be provided:
        noise_model: an instance of a NoiseModel (e.g., LaplaceNoiseModel)
        q: a variable s.a. an int, float or sympy symbol representing the original statistic
        x: a variable s.a. an int, float or sympy symbol representing the observed statistic, x = q + noise 
    It changes q and x into sympy symbols, s.t they can be used in the noise model and analyzer."""
    def __init__(self, noise_model: NoiseModel, q: str, x: str):
        self.noise_model = noise_model
        self.q = sp.Symbol(q, real=True)
        self.x = sp.Symbol(x, real=True)

    """
    The mean method calculates the mean of the estimator, using the noise model's moment method, implemented according to the specific noise distribution.

    Input: the estimator as a polynomial function in x.
    Output: the mean of the estimator, which is calculated using the noise model's moment method,
    implemented according to the specific noise distribution.
    """
    def mean(self, estimator):
        estimator = sp.sympify(estimator)

        # Exception handling: 

        # 1) If the estimator does not depend on x, 
        # then it is a constant and its mean is just itself.
        if not estimator.has(self.x):
            return estimator
        
        # 2) If the estimator it NOT a polynomial in x, 
        # then the moments method for calculating the mean does not apply.
        # forcing polynomial to be univariate in x
        try:
            poly = sp.Poly(estimator, self.x, domain="EX") 
            # setting domain to EX allows the coeffs to be arbitrary expressions
        except sp.PolynomialError:
            raise ValueError("Estimator must be a polynomial in x.")
        
        # 3) If there is a zero polynomial, then degree = -oo is avoided.
        if poly.is_zero:
            return sp.Integer(0)
        
        # Functionality:
        coeffs = poly.all_coeffs() 
        degree = poly.degree()

        mu = [self.noise_model.moment(n, self.q) for n in range(degree + 1)]
        expr = 0
        for i in range(degree + 1):
            a_i = coeffs[i]
            mu_i = mu[degree - i]
            expr += a_i * mu_i
        return sp.expand(expr) # leave "simplify" to higher level classes

    """ 
    The variance method calculates the variance of the estimator, using the mean method.

    Input: estimator as a polynomial function in x.
    Output: the variance of the estimator, which is calculated using the mean method.
    """
    def variance(self, estimator):
        estimator = sp.sympify(estimator)
        Eg = self.mean(estimator)
        Eg2 = self.mean(sp.expand(estimator**2))
        return  sp.simplify(sp.expand(Eg2 - Eg**2)) 
         # leave "simplify" to higher level classes
         # do simplify here...
         # or ?
    
    """
    The Mean Squared Error (MSE) of the estimator, depending on the unknown target statistic.
    Thus this is an a priori property of the estimator, as it depends on the true value of the statistic, which is unknown at the time of estimation.
    For f(q) as the target statistic and g(x) as the estimator, it is calculated as:
    E[(g(x) - f(q))^2], or 
    Var_[q](x) + Bias(x,f(q))^2.

    Input:
        estimator: a polynomial function in x representing the estimator
        target_statistic: a variable representing the true statistic, which can be f(q) for some polynomial f, or just q itself.
    Output: the MSE of the estimator, which is calculated using the variance, mean and bias methods.
    """
    def mse(self, estimator, target_statistic):
        estimator = sp.sympify(estimator)
        target_statistic = sp.sympify(target_statistic)

        mse_direct = self.mean(sp.expand((estimator - target_statistic)**2))

        Eg = self.mean(estimator)
        Eg2 = self.mean(sp.expand(estimator**2))
        bias = self.bias(estimator, target_statistic)
        variance = Eg2 - Eg**2
        mse = sp.expand(variance + bias**2)

        check = sp.simplify(sp.expand(mse - mse_direct))
        if check != 0:
            raise ValueError(f"MSE decomposition check failed: {check}")
        return mse
    
    """
    The Bias of the estimator, depending on the unknown target statistic.
    Thus this is an a priori property of the estimator, as it depends on the true value of the statistic, which is unknown at the time of estimation.
    For f(q) as the target statisti and h(x) as the estimator:
    Bias(h(x),f(q)) = E[h(x)] - f(q).
    
    Input:
        estimator: a polynomial function in x representing the estimator (often only the naive estimator).
        target_statistic: a variable representing the true statistic, which can be f(q) for some polynomial f, or just q itself.
    Output: the bias of the estimator, which is calculated using the mean method.
    """
    def bias(self, estimator, target_statistic):
        estimator = sp.sympify(estimator)
        target_statistic = sp.sympify(target_statistic)

        Eg = self.mean(estimator)
        bias = sp.expand(Eg - target_statistic)
        return bias