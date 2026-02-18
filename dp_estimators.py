import sympy as sp
from functools import lru_cache
from report_formatter import ReportFormatter

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
    def __init__(self, noise_model, q, x):
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
    Input: f is a polynomial function in q, simplify is a boolean indicating whether to simplify the results.
    Output: a report comparing the naive and unbiased estimators for f."""
    def compare(self, f, simplify=False):
        return ComparisonReport(self, f).compute(simplify=simplify)
    
    """ 
    Input: fs is a list of polynomial functions in q, simplify is a boolean indicating whether to simplify the results, 
           name_fn is an optional function that takes a polynomial and its index and returns a name for it.
    Output: a list of comparison reports for each polynomial in fs, optionally with names."""
    def compare_more(self, fs, simplify=False, name_fn=None):
        rows = []
        for i, f in enumerate(fs):
            res = self.compare(f, simplify=simplify)
            if name_fn:
                res["name"] = name_fn(f, i)
            rows.append(res)
        return rows

    """ Input: comparison_report is a report generated by the compare method, latex is a boolean indicating whether to format the output in LaTeX.
        Output: a pretty-printed version of the comparison report, optionally in LaTeX format."""
    def pretty_print(self, comparison_report, mode="grouped", latex=False, simplify_level=2, indent=0):
        noise_model = self.context.noise_model

        delta = noise_model.delta
        epsilon = noise_model.epsilon

        fmt = ReportFormatter(delta=delta, epsilon=epsilon)#, ratio_name="b")
        return fmt.render(
            comparison_report,
            mode=mode, 
            latex=latex,
            simplify_level=simplify_level,
            indent=indent
        )
    
    def pretty_compare(self, f, *, latex=False, **fmt_kwargs):
        return self.pretty_print(self.compare(f), latex=latex, **fmt_kwargs)

# --- ComparisonReport ---
class ComparisonReport:
    """ The ComparisonReport class is responsible for generating a report comparing the naive and unbiased estimators for a given polynomial function.
    It computes the estimators, their means and variances, and the gaps between them.
    When initializing the ComparisonReport, the following needs to be provided:
        system: an instance of EstimatorSystem to use for calculations
        f: a polynomial function in q of which to compare the estimators """
    def __init__(self, system, f):
        self.system = system
        self.f = sp.sympify(f)

    """ Input: f is a polynomial function in q, simplify is a boolean indicating whether to simplify the results.
        Output: a report comparing the naive and unbiased estimators for f. 
                The report is in the form of a dictionary containing the polynomial f, the naive and unbiased estimators, 
                their means and variances, and the gaps between them, optionally simplified."""
    def compute(self, simplify=False):
        system = self.system
        f = self.f

        g_naive = system.estimator(f, biasedness="naive")
        g_unbiased = system.estimator(f, biasedness="unbiased")

        mean_naive = system.analyzer.mean(g_naive)
        mean_unbiased = system.analyzer.mean(g_unbiased)
        variance_naive = system.analyzer.variance(g_naive)
        variance_unbiased = system.analyzer.variance(g_unbiased)

        result = {
            "polynomial": f,
            "naive": {
                "estimator": g_naive,
                "mean": mean_naive,
                "variance": variance_naive
            },
            "unbiased": {
                "estimator": g_unbiased,
                "mean": mean_unbiased,
                "variance": variance_unbiased
            },
            "mean_gap": mean_unbiased - mean_naive,
            "variance_gap": variance_unbiased - variance_naive,
        }
        if simplify:
            result = ComparisonReport._simplify_result(result)
        return result

    """ Input: result is a comparison report generated by the compute method.
        Output: a simplified version of the comparison report, where all expressions are simplified using sympy."""
    @staticmethod # static because it does not depend on instance
    def _simplify_result(result):
        simplified_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                simplified_result[key] = ComparisonReport._simplify_result(value)
            else:
                simplified_result[key] = sp.simplify(value)
        return simplified_result

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
        self.delta = sp.Symbol(delta, positive=True, real=True)
        self.epsilon = sp.Symbol(epsilon, positive=True, real=True)

    """ Helper method to calculate the k-th central moment of the noise distribution, E[noise^k], which is used in the moment calculation.
        Input: k is a non-negative integer representing the order of the moment
        Output: the k-th central moment of the noise distribution, which is (delta/epsilon)^k * k! if k is even, and 0 if k is odd."""
    def _central_moment(self, k):
        if k % 2 == 0: # even k
            return (self.delta / self.epsilon)**k * sp.factorial(k) 
        else: # odd k
            return 0 # if not caught by caller

    """ Input: i is a non-negative integer representing the order of the moment, q is a variable representing the original statistic.
        Output: the i-th moment of the noise distribution, E[(q + noise)^i], which is calculated using the binomial expansion and the central moments of the noise distribution.
        This method implements the formular from equation (12) in the thesis prep-project.
"""
    @lru_cache(maxsize=4096) # maybe set roof
    def moment(self, i, q):
        i = int(i) # for caching
        expr = 0
        for k in range(0, i + 1, 2): # only even k contributes
            expr += sp.binomial(i, k) * q**(i - k) * self._central_moment(k)
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
        # forcing polynomial to be univariate in x
        try:
            poly = sp.Poly(estimator, self.x, domain="EX") 
            # setting domain to EX allows the coeffs to be arbitrary expressions
        except sp.PolynomialError:
            raise ValueError("Estimator must be a polynomial in x.")
        
        coeffs = poly.all_coeffs() 
        degree = poly.degree()
        print(f"Estimator: {estimator}, degree: {degree}, coeffs: {coeffs}")

        mu = []
        for n in range(degree + 1):
            print(f"Checking coeff {n}: {coeffs[n]}")
            print(f"Free symbols in coeff {n}: {coeffs[n].free_symbols}")
            print(f"Does coeff {n} depend on x? {coeffs[n].has(self.x)}")
            print(f"Range of loop: n={n}, degree={degree}")
            if not coeffs[n].free_symbols.issubset({self.q}):
                raise ValueError("Coefficients must be functions of q only.")
            if not coeffs[n].has(self.x):
                raise ValueError("Coefficients must depend on x.")
            if not coeffs[n].has(self.q):
                raise ValueError("Coefficients must depend on q.")
            else:
                mu.append(self.noise_model.moment(n, self.q))

        #mu = [self.noise_model.moment(n, self.q) for n in range(degree + 1)]
        expr = 0
        for i in range(degree + 1):
            a_i = coeffs[i]
            mu_i = mu[degree - i]
            expr += a_i * mu_i
        return expr # leave "simplify" to higher level classes
    
    """ Input: estimator is a polynomial function in x.
        Output: the variance of the estimator, which is calculated using the mean method."""
    def variance(self, estimator):
        estimator = sp.sympify(estimator)
        Eg = self.mean(estimator)
        Eg2 = self.mean(sp.expand(estimator**2))
        return  Eg2 - Eg**2 # leave "simplify" to higher level classes