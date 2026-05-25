import sys
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_estimators import EstimatorSystem, ComparisonReport
from noise_models import LaplaceNoiseModel, GaussianNoiseModel
from utility.report_formatter import ReportFormatter
from utility.print_LaTeX import build_latex_document
# Define the SymPy symbols used in the monomials
q = sp.Symbol("q", real=True)
a = sp.symbols("a", real=True)
Delta = sp.Symbol("Delta", real=True)
epsilon = sp.Symbol("epsilon", real=True)
sigma = sp.Symbol("sigma", real=True)

# initialize the EstimatorSystem with the Laplace noise model and the symbolic variables
laplace_sys = EstimatorSystem(
    noise_model=LaplaceNoiseModel(Delta="Delta", epsilon="epsilon"),
    q="q",
    x="X"
)
gaussian_sys = EstimatorSystem(
    noise_model=GaussianNoiseModel(sigma="sigma"),
    q="q",
    x="x"
)
list_of_sys = [laplace_sys, gaussian_sys]
def main():
    for sys in list_of_sys:
        # In terminal
        print(f"Noise model: {sys.noise_model.__class__.__name__}")
        for i in range(2, 10):
            f = a*q**i
            report = ComparisonReport(sys, f).report()
            monomial = report["polynomial"]
            mse_gap = report["mse_gap"]
            var_gap = report["variance_gap"]
            print(f"Monomial: {monomial}, MSE gap: {mse_gap}, Variance gap: {var_gap}")
            # Create LaTeX report for each monomial
            sys.pdf_report(f, f"monomials/monomial_{i}_{sys.noise_model.__class__.__name__}", title=f"Monomial f(q) = a*q^{i} Report", compact=True)

"""
def main():
    latex_body = []
    fmt = ReportFormatter(Delta, epsilon)
    for sys in list_of_sys:
        latex_body += [
            r"\section*{NoiseModel}",
        ]
        for i in range(2, 20):
            # define monomial
            f = a*q**i
            # extract the monomial and the mean- and var-gap from the comparison report
            report = ComparisonReport(sys, f).report()
            monomial = report["polynomial"]
            mse_gap = report["mse_gap"]
            var_gap = report["variance_gap"]

            # format into LaTeX:
            monomial_latex = fmt._latex_expr(expr=monomial, notation="grouped", compact=True)
            mse_latex = fmt._latex_expr(expr=mse_gap, notation="grouped", compact=True)
            var_latex = fmt._latex_expr(expr=var_gap, notation="grouped", compact=True)

            # insert into LaTeX-body
            latex_body += [
                    r"\begin{dmath*}",
                    rf"f(q) = {monomial_latex}",
                    r"\end{dmath*}",
                    r"\begin{dmath*}",
                    rf"\text{{variance gap}} = {var_latex}",
                    r"\end{dmath*}",
                    r"\begin{dmath*}",
                    rf"\text{{MSE gap}} = {mse_latex}",
                    r"\end{dmath*}",
                ]
    build_latex_document(latex_body="\n".join(latex_body), output_stem="monomials", title="Monomials")
"""
if __name__ == "__main__":
    main()