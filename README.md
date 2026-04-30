As the contributor of this repo is, in her heart, a Machine Learner (MLer), and NOT a Computer Scientist, the basics of "good code" was not a fundament of her professional education. Rather, when a MLer does their work; they code, they patch-up, they get their result, and then they throw their code in the trash,  never to look at it again. 
Though sufficient for some task, this contributor deemed this approach inadequate for this project and leaned on her problem-solving skills, her structure (lol) and her stubbornness to come up with a better plan.
Thus the structure of this code might not live up to the professionel expectations from an inherent Computer Sciencetist, but please, bear with this author, as her ML-DNA is showing within the lines of this code base.
True to the heart of DIKU, some of this code has also been created while drunk.

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

When using the library, remember that SymPy uses SYMBOLIC expressions and functions. Therefor the symbols needs to be defined as "symbolic", i.e.:

    Delta = sp.Symbol("Delta", real=True, positive=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)
    q = sp.Symbol("q", real=True)
    X = sp.Symbol("X", real=True)

And then later substituted with real values. This property is fundamental for this code base. 