# Preface
As the contributor of this repo is, in her heart, a Machine Learner (MLer), and NOT a Computer Scientist, the basics of "good code" was not a fundament of her professional education. Rather, when a MLer does their work; they code, they patch-up, they get their result, and then they throw their code in the trash,  never to look at it again. 
Though sufficient for some task, this contributor deemed this approach inadequate for this project and leaned on her problem-solving skills, her structure (lol) and her stubbornness to come up with a better plan.
Thus the structure of this code might not live up to the professionel expectations from an inherent Computer Sciencetist, but please, bear with this author, as her ML-DNA is showing within the lines of this code base.
True to the heart of DIKU, some of this code has also been created while drinking.

# Welcome! 
This codebase exist as a product of the thesis "Bias-Variance Analysis of Polynomial Estimators under Differential Privacy" and serve as both a tool and a deliverable for the final project. It builds on the closed form expression of unbiased estimators from Theorem 10 by Calmon et al. and the linear system for general noise distributions from Theorem 22 by Calmon et al. 

In the main folder, the main architecture lies; dp_estimators.py and noise_models.py. These files implements the main functionality of this repo, which includes:
for any given univariate polynomial function, it is possible to: 
\begin{itemize}

    \item Compute the unbiased estimator under Laplace noise using the closed-form formula from Theorem 10 (the \texttt{LaplaceNoiseModel} class).

    \item Compute the unbiased estimator under Gaussian noise by solving the linear system from the proof of Theorem 22 (the \texttt{GaussianNoiseModel} class inheriting from the \texttt{NoiseModel} class).

    \item Compute the naive estimator.

    \item Compute analysis of all estimators including mean, bias (for the naive), variance and MSE. 

    \item Compare different estimators for said function by computing gaps between analysis measures.
    
\end{itemize} 

# User Guide


Important!
When using the library, remember that SymPy uses SYMBOLIC expressions and functions. Therefor the symbols needs to be defined as "symbolic", i.e.:

    Delta = sp.Symbol("Delta", real=True, positive=True)
    epsilon = sp.Symbol("epsilon", real=True, positive=True)
    q = sp.Symbol("q", real=True)
    X = sp.Symbol("X", real=True)

And then later substituted with real values. This property is fundamental for this code base. 

# Dependencies
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

# Design
The main modules are in the files dp_estimators.py and noise_models.py. 
The four classes in \texttt{dp\_estimators.py} are:
\begin{itemize}
    \item EstimatorSystem: the main interface for users to interact with the system, providing methods for getting estimators and comparing them.
    \item ComparisonReport: responsible for generating a report for a given polynomial function.
    \item EstimatorContext: responsible for managing different noise/estimation strategies and providing methods for getting naive and unbiased estimators.
    \item EstimatorAnalyzer: responsible for analyzing the properties of estimators, such as calculating mean, variance etc, using the noise model's moment calculations.
\end{itemize}
The three classes in \texttt{noise\_models.py} are:
\begin{itemize}
    \item NoiseModel: an abstract base class defining the interface for the possible noise distributions. Further it implements the linear system from Section \ref{section_GeneralNoiseDistributions} as a fallback method for the unbiased transform.
    \item LaplaceNoiseModel: a concrete strategy class calculating the unbiased transformation of a polynomial under Laplace noise.
    \item GaussianNoiseModel: a concrete strategy class that inherits the fallback method from the NoiseModel class.
\end{itemize}

The code structure follows the Facade design pattern, where the EstimatorSystem class provides a simplified interface to the complex subsystem of noise models, estimation strategies, and analysis. The user of the code base only interacts with the class EstimatorSystem, and instances of the other classes are initiated through this class, except for the ComparisonReport which though still needs and instance of EstimatorSystem as input. EstimatorSystem is also responsible for delegating the comparison of estimator of a given polynomial function to the ComparisonReport class. 

Further it follows the Strategy design pattern,where the EstimatorContext class manages different estimation strategies. This includes which type of noise the unbiased estimator should employ. The class NoiseModel defines a common abstract interface for noise models, thus supporting different noise distributions. The two classes, LaplaceNoiseModel and GaussianNoiseModel inherits NoiseModel and provide concrete implementations of this interface. When EstimatorContext is initiated, it maintains a reference to the object created from one of the concrete strategies. EstimatorContext interacts with this object only through the interface defined by NoiseModel. 
Further EstimatorContext also produces the naive estimator. 

The EstimatorAnalyzer class manages the \textit{analysis} of any of the estimators, i.e. it creates corresponding mean, variance, MSE and bias. It serves the same purpose as the "Context" in the Strategy pattern, only here the estimator and analysis of the estimator are split into two classes to avoid overloading the EstimatorContext class. 

The ComparisonReport class is responsible for generating a report comparing the naive and unbiased estimators for a given polynomial function, by calculating their means, variances, MSE, bias and the gaps between them. Different specifications are possible, depending on whether the user wants to get a quick overview of the estimators of a polynomial in the terminal or rather wants to get a report printed on PDF.
