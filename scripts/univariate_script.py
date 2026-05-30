import sympy as sp
from dp_estimators import EstimatorSystem
from noise_models import LaplaceNoiseModel, GaussianNoiseModel

def main():
    # ---- USER INPUTS ----
    polynomial_str = "3*q**3 + 7*q**2 + 2*q + 1"  # change this
    noise_str = "laplace"  # or "gaussian"
    name = "polynomial_report"  # name for the generated report PDF
    # ---------------------


    # Parse polynomial and symbols
    poly_expr, symbols = parse_polynomial(polynomial_str)

    # Ensure variable 'q' exists; user might use x or something else
    if "q" not in symbols:
        raise ValueError("Polynomial must use the variable 'q' as the input variable.")

    # Instantiate noise model
    noise_model = create_noise_model(noise_str)

    # Choose name for the report PDF
    if name is None:
        name = "polynomial_report"

    # Run estimator system
    run_analysis(
        poly_expr=poly_expr,
        noise_model=noise_model,
        report_name=name
    )

def create_noise_model(name: str):
    """Return instantiated noise model from string."""
    name = name.lower()
    if name == "laplace":
        return LaplaceNoiseModel(Delta=1.0, epsilon=3.0)
    if name == "gaussian":
        return GaussianNoiseModel(sigma="sigma")
    raise ValueError(f"Unknown noise model '{name}'. Use 'laplace' or 'gaussian'.")


def parse_polynomial(poly_str: str):
    """
    Convert a user‑provided polynomial string to a SymPy expression.
    Automatically identifies and declares all used symbols.
    """
    # collect symbol names by scanning the expression
    symbol_names = sorted({str(s) for s in sp.sympify(poly_str).free_symbols})

    # define sympy symbols dynamically
    symbols_dict = {name: sp.Symbol(name, real=True) for name in symbol_names}

    # parse the expression using the symbol table
    poly_expr = sp.sympify(poly_str, locals=symbols_dict)

    return poly_expr, symbols_dict


def run_analysis(poly_expr, noise_model, report_name):
    sys = EstimatorSystem(
        noise_model=noise_model,
        q="q",
        x="X"
    )

    summary = sys.summary_report(poly_expr, notation="beta")
    print(summary)

    sys.pdf_report(poly_expr, report_name)


if __name__ == "__main__":
    main()
