import sympy as sp
from dp_estimators import EstimatorSystem
from noise_models import LaplaceNoiseModel
# Define the SymPy symbols used in the polynomials
q = sp.Symbol("q", real=True)
a, b, c, d = sp.symbols("a b c d", real=True)

# initialize the EstimatorSystem with the Laplace noise model and the symbolic variables
sys = EstimatorSystem(
    noise_model=LaplaceNoiseModel(delta="Delta", epsilon="epsilon"),
    q="q",
    x="X"
)

# define the cubic / quadratic function f(q) = a*q**3 + b*q**2 + c*q + d / f(q) = a*q**2 ...
# (as a sympy expression, not a Python function, since we want to analyze it symbolically)
f_cubic = a*q**3 + b*q**2 + c*q + d
f_quadratic = a*q**2 + b*q + c

f_cubic_basecase = f_cubic.subs({a:1, b: 0, c: 0, d: 0}) 

summary_cubic = sys.summary_report(f_cubic, notation="beta") # use the beta mode for terminal summary, where beta = Delta/epsilon is easy to read
print(summary_cubic)
summary_quadratic = sys.summary_report(f_quadratic, notation="beta")
print(summary_quadratic)
summary_cubic_basecase = sys.summary_report(f_cubic_basecase, notation="beta")
print(summary_cubic_basecase)

sys.pdf_report(f_cubic, "cubic_report", title="Cubic Function Report", compact=True) # use compact mode for the LaTeX report, to remove paranthesis basically
sys.pdf_report(f_quadratic, "quadratic_report", title="Quadratic Function Report", compact=True)
sys.pdf_report(f_cubic_basecase, "cubic_basecase_report", title="Cubic Base Case Report", compact=True)