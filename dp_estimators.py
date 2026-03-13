import sympy as sp
from functools import lru_cache
from report_formatter import ReportFormatter
from print_LaTeX import build_latex_document
from typing import Union


__all__ = ["EstimatorSystem", 
           "ComparisonReport", 
           "NoiseModel", 
           "LaplaceNoiseModel", 
           "GaussianNoiseModel", 
           "EstimatorContext", 
           "EstimatorAnalyzer"
           ]

"""
Explaining the code design: 
The code is designed using the Strategy design pattern, which allows for flexibility in choosing different noise models and estimation strategies.
Further the Facade design pattern is used to provide a simple interface for users to interact with the system, hiding the complexities of the underlying implementations.
It is the EstimatorSystem class that serves as the facade, providing methods for getting estimators and comparing them, while the other classes implement the specific strategies and analysis.
The main classes are:
- EstimatorSystem: the main interface for users to interact with the system, providing methods for getting estimators and comparing them.
- ComparisonReport: responsible for generating a report comparing the naive and unbiased estimators for a given polynomial function.
- NoiseModel: an abstract class defining the interface for different noise models, with methods for calculating moments and unbiased transforms.
- LaplaceNoiseModel: a concrete implementation of the NoiseModel for Laplace noise, providing methods for calculating moments and unbiased transforms specific to Laplace noise.
- GaussianNoiseModel: a concrete implementation of the NoiseModel for Gaussian noise, which can be implemented similarly to LaplaceNoiseModel if needed.
- EstimatorContext: responsible for managing different noise/estimation strategies and providing methods for getting naive and unbiased estimators.
- EstimatorAnalyzer: responsible for analyzing the properties of estimators, such as calculating mean and variance, using the noise model's moment calculations.
This design allows for easy extension to other noise models and estimation strategies by simply implementing new classes that adhere to the defined interfaces, without needing to modify the existing codebase.
The name "EstimatorContext" is beause it reflects the role of "context" in the Strategy pattern, where it maintains a reference to one of the concrete strategies, 
and interacts with this object only through the interface defined by the abstract strategy (NoiseModel).
The name "EstimatorAnalyzer" is chosen to reflect its role in analyzing the properties of estimators, such as calculating mean and variance,
but it serves the same purpose as the "Context" in the Strategy pattern, only here the estimator and analysis of the estimator are split into two classes, 
where the EstimatorContext is responsible for managing the estimation strategies and the EstimatorAnalyzer is responsible for analyzing the properties of the estimators.
"""
# --- Strategy NoiseModel ---
class NoiseModel:
    """The strategy "NoiseModel" defines the interface for different noise models, 
    passing to concrete strategies s.a. Laplace Noise or Gaussian Noise.
    OBS! The noise model can only be of distributions with known moments.
    It requires the implementation of the following methods:
        moment(n, q): calculates the n-th moment of the noise distribution given q.
        unbiased_transform(f, x): returns a function g(x) such that E[g(q + noise)] = f(q) for the given polynomial f."""
    
    def moment(self, n, q):
        # \mu_n(q) = E[(q + noise)^n] for x = q + noise
        raise NotImplementedError("Subclasses must implement this method")
    
    def unbiased_transform(self, f, x):
        # Returns g(x) s.t. E[g(q + noise)] = f(q)
        raise NotImplementedError("Subclasses must implement this method")
    
    def clear_cache(self):
        pass

    def cache_info(self):
        pass

# --- Concrete Strategy LaplaceNoiseModel ---
class LaplaceNoiseModel(NoiseModel):
    """The concrete stategy Laplace Noise Model is resposible for 
    calculating moments and the unbiased transform for Laplace noise.
    When initializing the LaplaceNoiseModel, the following needs to be provided:
        delta: a variable s.a. an int, float or sympy symbol representing the sensitivity of the query
        epsilon: a variable s.a. an int, float or sympy symbol representing the privacy budget / noise scale"""
    def __init__(self, delta, epsilon):
        #self.b = sp.sympify(b)
        self.delta = sp.sympify(delta)
        self.epsilon = sp.sympify(epsilon)

    """ Helper method to calculate the k-th moment of the Laplace noise distribution, E[noise^k], 
        which is used in the central moment calculation. Because Laplace noise has mean 0, the k-th 
        moment about zero is the same as the k-th central moment.
        Input: k is a non-negative integer representing the order of the moment
        Output: the k-th moment of the noise distribution, which is (delta/epsilon)^k * k! if k is even, and 0 if k is odd."""
    def _moment_about_zero(self, k):
        if k % 2 == 0: # even k
            return (self.delta / self.epsilon)**k * sp.factorial(k) 
        else: # odd k
            return 0 # if not caught by caller

    """ Input: i is a non-negative integer representing the order of the moment, q is a variable representing the original statistic.
        Output: the i-th CENTRAL moment of the released statistic, E[(q + noise)^i], which is calculated using the binomial expansion and the moments of the noise distribution.
        This method implements the formular from equation (12) in the thesis prep-project.
        It is the central moment because it is the moment of the released statistic x = q + noise, which is centered around q, as the noise has mean 0."""
    @lru_cache(maxsize=4096) # maybe set roof
    def moment(self, i, q):
        i = int(i) # for caching
        expr = 0
        for k in range(0, i + 1, 2): # only even k contributes
            expr += sp.binomial(i, k) * q**(i - k) * self._moment_about_zero(k)
        return expr
    
    """ Input: f is a polynomial function in q, x is the variable representing the observed statistic.
        Output: the unbiased estimator g(x) = f(x) - (delta/epsilon)^2 f''(x), s.t. E[g(x)] = f(q)."""
    def unbiased_transform(self, f, x):
        f_dd = sp.diff(f, x, 2)  # second derivative
        g = f - (self.delta / self.epsilon)**2 * f_dd
        return g
    
    def clear_cache(self):
        self.moment.cache_clear()

    def cache_info(self):
        return self.moment.cache_info()

# --- Concrete Strategy GaussianNoiseModel ---
class GaussianNoiseModel(NoiseModel):
    def __init__(self, sigma):
        self.sigma = sp.sympify(sigma)

    # To be implemented if needed
    # def gaussian_moment(self, k):
    #     pass

    # @lru_cache(maxsize=None) # maybe set roof
    # def moment_of_gaussian(self, i, q):
    #     pass   

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
    Input: f is a polynomial function in q, biasedness is either "naive" or "unbiased".
    Output: the corresponding estimator g in x."""
    def estimator(self, f, biasedness="naive"):
        if biasedness == "naive":
            return self.context.naive(f)
        elif biasedness == "unbiased":
            return self.context.unbiased(f)
        else:
            raise ValueError("biasedness must be either 'naive' or 'unbiased'")
    
    """
    Input: f is a polynomial function in q.
    Output: a report comparing the naive and unbiased estimators for f."""
    def compare(self, f):
        return ComparisonReport(self, f=f).report()

    """
    Input: fs: a list of polynomial functions in q,
           name_fn: an optional function that takes a polynomial and its index and returns a name for it.
    Output: a list of comparison reports for each polynomial in fs, optionally with names."""
    def compare_more(self, fs, name_fn=None):
        rows = []
        for i, f in enumerate(fs):
            res = self.compare(f=f)
            if name_fn:
                res["name"] = name_fn(f, i)
            rows.append(res)
        return rows
    
    def summary_report(self, f, *, notation="noise_scale", compact=False):
        report = self.compare(f)
        noise_model = self.context.noise_model
        if hasattr(noise_model, "delta") and hasattr(noise_model, "epsilon"):
            delta = noise_model.delta
            epsilon = noise_model.epsilon
            fmt = ReportFormatter(delta=delta, epsilon=epsilon)
        else:
            raise ValueError("summary_compare currently expect a Laplace-like model.")
        return fmt.render_summary(report, notation=notation, compact=compact)

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

    """ Input: f is a polynomial function in q, simplify is a boolean indicating whether to simplify the results.
        Output: a report comparing the naive and unbiased estimators for f. 
                The report is in the form of a RAW dictionary containing the polynomial f, the naive and unbiased estimators, 
                their means, variances, MSE and the gaps between them, the ratio and relative variance."""
    def report(self):
        system = self.system
        f = self.f

        g_naive = system.estimator(f, biasedness="naive")
        g_unbiased = system.estimator(f, biasedness="unbiased")

        mean_naive = system.analyzer.mean(g_naive)
        mean_unbiased = system.analyzer.mean(g_unbiased)
        variance_naive = system.analyzer.variance(g_naive)
        variance_unbiased = system.analyzer.variance(g_unbiased)
        mse_naive = system.analyzer.mse_of_estimator(g_naive, f)
        mse_unbiased = system.analyzer.mse_of_estimator(g_unbiased, f)

        return {
            "polynomial": f,
            "naive": {
                "estimator": g_naive,
                "mean": mean_naive,
                "variance": variance_naive,
                "mse": mse_naive
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
        }

    #""" Input: result is a comparison report generated by the compute method.
    #    Output: a simplified version of the comparison report, where all expressions are simplified using sympy."""
    #@staticmethod # static because it does not depend on instance
    #def _simplify_result(result):
    #    simplified_result = {}
    #    for key, value in result.items():
    #        if isinstance(value, dict):
    #            simplified_result[key] = ComparisonReport._simplify_result(value)
    #        else:
    #            simplified_result[key] = sp.simplify(value)
    #    return simplified_result

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

    """ Input: f is a polynomial function in q.
        Output: the plug-in estimator g(x) = f(q), which is just the substitution of q with x, s.t the function stays the same, i.e. g = f."""
    def naive(self, f):
        f = sp.sympify(f)
        return f.subs(self.q, self.x) # naive is just substitution
        # .subs is used to substitute a variable or expression with a specified 
        # value or another expression in a symbolic mathematical expression
    
    """ Input: f is a polynomial function in q.
        Output: the unbiased estimator g(x), which is calculated using the noise model's unbiased_transform method, 
        implemented according to the specific noise distribution."""
    def unbiased(self, f):
        f = sp.sympify(f)
        f_in_x = f.subs(self.q, self.x) # substitute q with x in f to turn f(q) into f(x) = f(q + noise)
        g = self.noise_model.unbiased_transform(f_in_x, self.x)
        return sp.expand(g)

# --- Analyzer ---
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

    """ Input: estimator is a polynomial function in x.
        Output: the mean of the estimator, which is calculated using the noise model's moment method, 
        implemented according to the specific noise distribution."""
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

    """ Input: estimator is a polynomial function in x.
        Output: the variance of the estimator, which is calculated using the mean method."""
    def variance(self, estimator):
        estimator = sp.sympify(estimator)
        Eg = self.mean(estimator)
        Eg2 = self.mean(sp.expand(estimator**2))
        return  sp.simplify(sp.expand(Eg2 - Eg**2)) 
    # leave "simplify" to higher level classes
    # do simplify here...
    
    def mse_of_estimator(self, estimator, target_statistic):
        """The Mean Squared Error (MSE) of the estimator, depending on the unknown target statistic.
        Thus this is an a priori property of the estimator, as it depends on the true value of the statistic, which is unknown at the time of estimation.
        For f(q) as the target statistic and g(x) as the estimator, it is calculated as:
        E[(g(x) - f(q))^2], or 
        Var_[q](x) + Bias(x,f(q))^2.

        Input:
            estimator: a polynomial function in x representing the estimator
            target_statistic: a variable representing the true statistic, which can be f(q) for some polynomial f, or just q itself.
        Output: the MSE of the estimator, which is calculated using the variance and mean methods."""
        estimator = sp.sympify(estimator)
        target_statistic = sp.sympify(target_statistic)

        mse_direct = self.mean(sp.expand((estimator - target_statistic)**2))

        Eg = self.mean(estimator)
        Eg2 = self.mean(sp.expand(estimator**2))
        bias = Eg -target_statistic
        variance = Eg2 - Eg**2
        mse = sp.expand(variance + bias**2)

        check = sp.simplify(sp.expand(mse - mse_direct))
        if check != 0:
            raise ValueError(f"MSE decomposition check failed: {check}")
        return mse