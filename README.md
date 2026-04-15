To be continued...

------------------------ Dependencies ------------------------------
SymPy

dp_estimators.py:
    typing.Union

noise_models.py:
    functools.lru_cache

print_LaTeX.py:
    subprocess, pathlib.Path

plot_mse_var.py:
    numpy, os, matplotlib, pathlib.Path
----------------------- Dependencies --------------------------------

To Do: 
-- Write README.
-- Write explanaintion in all scripts.

Observations:
1. When using the library, remember that SymPy uses SYMBOLIC expressions and functions. Therefor the symbols needs to be defined as "symbolic", i.e.:

    Delta = sp.Symbol("Delta", real=True, positive=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)
    q = sp.Symbol("q", real=True)
    X = sp.Symbol("X", real=True)

And then later substituted with real values. 

2. Dependencies; maybe the packages for LaTeX, breqn, cause trouble, but needed for breaking expressions when compiling PDF.

3. Fix TypeError: 'NegativeInfinity' object cannot be interpreted as an integer in mu in EstimatorAnalyzer (maybe not relevant).
