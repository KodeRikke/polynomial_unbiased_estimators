To be continued...

------------------------ Dependencies ------------------------------
SymPy

dp_estimators.py:
    functools.lru_cache, typing.Union

print_LaTeX.py:
    subprocess, pathlib.Path

plotting.py:
    numpy, os, matplotlib
--------------------------------------------------------------------

To Do: 
1. Write README
2. Write documentation for everything
3. Fix TypeError: 'NegativeInfinity' object cannot be interpreted as an integer in mu in EstimatorAnalyzer (maybe not relevant)
4. Dependencies; maybe the packages for LaTeX, breqn, cause trouble, but needed for breaking expressions when compiling PDF. 
5. When printing plots (Cheb), the text indicating "unbiased is lower" vs "naive is lower" is hardcoded on the y-axis and might end up weird places.
6. Maybe make a clean script to remove all tex, aux, pdf files (and plots?) - the folder gets messy.

Observations:
1. When using the library, remember that SymPy uses SYMBOLIC expressions and functions. Therefor the symbols needs to be defined as "symbolic", i.e.:

    Delta = sp.Symbol("Delta", real=True, positive=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)
    q = sp.Symbol("q", real=True)
    X = sp.Symbol("X", real=True)

And then later substituted with real values. 
