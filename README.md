To be continued...

Dependencies: 
Sympy, functools, 
numpy, matplotlib, 
os

To Do: 
1. Write README
2. Finish report_formatter to make it print out the expressions with Delta and epsilon grouped. It is currently not working optimal.
3. Fix TypeError: 'NegativeInfinity' object cannot be interpreted as an integer in mu in EstimatorAnalyzer.
4. Fix printing out chebycheb Pretty Printer. 
5. When printing plots, the text indicating "unbiased is lower" vs "naive is lower" is hardcoded on the y-axis and might end up weird places.

Important observations:
1. When using the library, remember that SymPy uses SYMBOLIC expressions and functions. Therefor the symbols needs to be defined as "symbolic", i.e.:
    Delta = sp.Symbol("Delta", real=True, positive=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)
    q = sp.Symbol("q", real=True)
    X = sp.Symbol("X", real=True)
And then later substituted with real values. 
