As the contributor of this repo is, in her heart, a Machine Learner (MLer), and NOT a Computer Scientist, the basics of "good code" was not a fundament of her professional education. Rather, when a MLer does their work; they code, they patch-up, they get their result, and then they throw their code in the trash,  never to look at it again. 
Though sufficient for some task, this contributor deemed this approach inadequate for this project and leaned on her problem-solving skills, her structure (lol) and her stubbornness to come up with a better plan.
Thus the structure of this code might not live up to the professionel expectations from an inherent Computer Sciencetist, but please, bear with this author, as her ML-DNA is showing within the lines of this code base.
True to the heart of DIKU, some of this code has also been created while drunk.

------------------------ Dependencies ------------------------------

    All files: 
        sympy
        numpy
        os
        matplotlib
        typing.Union
        functools.lru_cache
        subprocess 
        pathlib.Path
        math 
        scipy.special
    
    List of dependencies for individual files (excluding scripts):

    # General Architecture
    noise_nodels.py:
        sympy, typing.Union
    
    dp_estimators.py:
        sympy, functools.lru_cache, 
        report_formatter.ReportFormatter, print_LaTeX.build_latex_document, noise_models.NoiseModel
    
    # Utility
    print_LaTeX.py:
        subprocess, pathlib.Path

    report_formatter.py:
        sympy
    
    gaussian.py:
        math.exp, math.sqrt, scipy.special.erf
    
    SigmaFromEpsilon.py:
        sympy
        gaussian.calibrateAnalyticGaussianMechanism

    utility_plotting.py:
        sympy, numpy, os, matplotlib.pyplot, pathlib.Path
        SigmaFromEpsilon.SigmaFromEpsilon
    
    plot_mse_var.py:
        numpy, os, matplotlib, pathlib.Path
    
    # Plotting
    relative_laplace_vs_gaussian.py
        sympy, numpy, os.path
        dp_estimators.EstimatorSystem, noise_models.LaplaceNoiseModel, noise_models.GaussianNoiseModel,
        utility_plotting.format_value, utility_plotting.metric_expr, utility_plotting.evaluate_on_grid, utility_plotting.plot_curves, utility_plotting.metric_folder
        SigmaFromEpsilon.SigmaFromEpsilon

    direct_laplace_vs_gaussian.py
        sympy, numpy, os.path
        dp_estimators.EstimatorSystem, noise_models.LaplaceNoiseModel, noise_models.GaussianNoiseModel,
        utility_plotting.format_value, utility_plotting.metric_expr, utility_plotting.evaluate_on_grid, utility_plotting.plot_curves, utility_plotting.metric_folder
        SigmaFromEpsilon.SigmaFromEpsilon

    relative_laplace_chebyshev.py
        sympy, numpy, os.path
        dp_estimators.EstimatorSystem, noise_models.LaplaceNoiseModel
        utility_plotting.format_value, utility_plotting.metric_expr, utility_plotting.evaluate_on_grid, utility_plotting.plot_curves, utility_plotting.metric_folder


----------------------- Dependencies --------------------------------

When using the library, remember that SymPy uses SYMBOLIC expressions and functions. Therefor the symbols needs to be defined as "symbolic", i.e.:

    Delta = sp.Symbol("Delta", real=True, positive=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)
    q = sp.Symbol("q", real=True)
    X = sp.Symbol("X", real=True)

And then later substituted with real values. This property is fundamental for this code base. 


New files:

    monte_carlo/simulation.py
    scripts/run_monte_carlo_study.py
    plotting/plot_monte_carlo_study.py

Purpose:

    Empirically compare naive vs unbiased estimators for high-order polynomials
    under Laplace and Gaussian noise, while validating against symbolic theory.
     The study now includes coefficient sweeps, so the empirical plots can be
     compared directly against the quadratic and cubic derivations and then
     extended to higher-degree families.

Run:

    python3 scripts/run_monte_carlo_study.py

    # quick run (smaller sample size / faster)
    python3 scripts/run_monte_carlo_study.py --quick --samples 15000 --seed 1337
     # coefficient-only study
     python3 scripts/run_monte_carlo_study.py --study-mode coefficients --samples 15000 --seed 1337

What it does:

    - Builds symbolic estimators using EstimatorSystem
    - Substitutes all non-swept symbols numerically before runtime
    - Samples noise and computes empirical mean/variance/MSE/bias
    - Prints symbolic metrics and empirical-vs-symbolic errors
     - Sweeps polynomial coefficients for quadratic, cubic, and higher-degree
        families so coefficient sensitivity is visible in the Monte Carlo data
    - Saves a CSV file at reports/monte_carlo/monte_carlo_results.csv

Plot summaries:

    python3 plotting/plot_monte_carlo_study.py

    # focus degree-trend plots at q=1.0 (default)
    python3 plotting/plot_monte_carlo_study.py --q-focus 1.0
     The plotting script also produces per-family coefficient-sweep summaries.

Presentation figure:

    python3 plotting/plot_symbolic_presentation.py --output-dir plots/presentation

    # recommended epsilon set for the main presentation figure
    python3 plotting/plot_symbolic_presentation.py --output-dir plots/presentation --epsilons 0.5 1.0 2.0

This creates a compact 2x2 symbolic summary for the baseline quadratic,
transition cubic, Chebyshev exception, and a coefficient-sensitive cubic.
The MSE figure is the best candidate for the main text; the variance figure
is useful as a supporting or appendix graphic.

Notes:

    - For Gaussian runs, sigma is calibrated from (epsilon, delta, Delta)
      using dp_calibration/SigmaFromEpsilon.py.
    - For high-order polynomials, variance can be extremely large; empirical
      convergence may require many more Monte Carlo samples than low-order cases.
     - The coefficient sweeps are intentionally small and structured so the
        quadratic and cubic empirical results can be checked against the theory
        before interpreting higher-degree behavior.