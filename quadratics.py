import sympy as sp
from dp_estimators import EstimatorSystem, LaplaceNoiseModel

# Define the SymPy symbols used in the polynomials
q = sp.Symbol("q", real=True)
a, b, c = sp.symbols("a b c", real=True)

# initialize the EstimatorSystem with the Laplace noise model and the symbolic variables
sys = EstimatorSystem(
    noise_model=LaplaceNoiseModel(delta="Delta", epsilon="epsilon"),
    q="q",
    x="X"
)

# define the quadratic function f(q) = a*q^2 + b*q + c
# (as a sympy expression, not a Python function, since we want to analyze it symbolically)
f = a*q**2 + b*q + c

prettyyy = sys.pretty_compare(f, latex=False, mode="grouped", simplify_level=2)
print("Pretty comparison report for f(q) = a*q^2 + b*q + c under Laplace noise:")
print(prettyyy)

"""
# compute the naive and unbiased estimators for f
naive_estimator = sys.context.naive(f)
unbiased_estimator = sys.context.unbiased(f)

# compute the variance and mean of the naive and unbiased estimators
naive_variance = sys.analyzer.variance(naive_estimator)
unbiased_variance = sys.analyzer.variance(unbiased_estimator)
naive_mean = sys.analyzer.mean(naive_estimator)
unbiased_mean = sys.analyzer.mean(unbiased_estimator)

print("Naive estimator:", naive_estimator)
print("Unbiased estimator:", unbiased_estimator)
print("Naive variance:", naive_variance)
print("Unbiased variance:", unbiased_variance)
print("Naive mean:", naive_mean)
print("Unbiased mean:", unbiased_mean)
"""
