import sympy as sp
from dp_estimators import EstimatorSystem, LaplaceNoiseModel

""" THis script demonstrates how to use the EstimatorSystem to compare the variance of 
    naive and unbiased estimators for the first 20 Chebyshev polynomials under Laplace 
    noise. The script defines the noise model, the symbolic variables, and then uses 
    the compare_more method of the EstimatorSystem to compute the variance of the naive 
    and unbiased estimators for each Chebyshev polynomial. Finally, it prints the 
    variance gap for each polynomial."""

# Define the SymPy symbol used in the polynomials
q = sp.Symbol("q", real=True)

sys = EstimatorSystem(
    noise_model=LaplaceNoiseModel(delta="Delta", epsilon="epsilon"),
    q="q",
    x="q_tilde"
)

# First 20 Chebyshev polynomials T_1(q) ... T_20(q)
fs = [sp.chebyshevt(n, q) for n in range(1, 21)]

rows = sys.compare_more(
    fs,
    simplify=False,
    name_fn=lambda f, i: f"T_{i+1}"
)

for r in rows:
    vn = r["naive"]["variance"]
    vu = r["unbiased"]["variance"]
    gap = sp.simplify(vu - vn)
    print(r.get("name", "?"), "variance gap =", gap)
