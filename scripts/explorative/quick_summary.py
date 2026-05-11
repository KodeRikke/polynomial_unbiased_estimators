import sys
import sympy as sp
from dp_estimators import EstimatorSystem
from noise_models import LaplaceNoiseModel, GaussianNoiseModel


def create_noise_model(name: str):
    name = name.lower()
    if name == "laplace":
        return LaplaceNoiseModel(Delta="Delta", epsilon="epsilon")
    if name == "gaussian":
        return GaussianNoiseModel(Delta="Delta", epsilon="epsilon")
    raise ValueError(f"Unknown noise model '{name}'. Use 'laplace' or 'gaussian'.")


def parse_polynomial(poly_str: str):
    """
    Convert a user-provided polynomial string to a SymPy expression.
    Accepts arbitrary whitespace.
    """

    normalized = " ".join(poly_str.strip().split())

    # first pass: detect free symbols
    tmp_expr = sp.sympify(normalized)
    symbol_names = sorted(str(s) for s in tmp_expr.free_symbols)

    symdict = {name: sp.Symbol(name, real=True) for name in symbol_names}

    expr = sp.sympify(normalized, locals=symdict)
    return expr, symdict


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m scripts.quick_summary <noise> <univariate> <polynomial...>")
        sys.exit(1)

    noise_str = sys.argv[1]
    mode = sys.argv[2].lower()

    if mode != "univariate":
        raise ValueError("This script only supports 'univariate' mode.")

    # join all remaining arguments into one polynomial string
    poly_str = " ".join(sys.argv[3:])

    # parse polynomial
    poly_expr, symbols = parse_polynomial(poly_str)

    # enforce univariate: must have q as the single input variable
    if "q" not in symbols:
        raise ValueError("Univariate mode requires the polynomial to use 'q' as the input variable.")

    noise_model = create_noise_model(noise_str)

    sys_est = EstimatorSystem(noise_model=noise_model, q="q", x="X")
    summary = sys_est.summary_report(poly_expr, notation="beta")
    print(summary)


if __name__ == "__main__":
    main()
