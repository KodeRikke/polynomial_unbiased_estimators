import numpy as np 
import sympy as sp

# Define the SymPy symbols used in the polynomials
q = sp.Symbol("q", real=True)
a, b, c, d, e = sp.symbols("a b c d e", real=True)
B = sp.Symbol("B", real=True, positive=True)  # Scale parameter for the Laplace distribution

# Define the polynomial function we wish to test this with: 
f_fourth_degree = a*q**4 + b*q**3 + c*q**2 + d*q + e

# Define the moments of the distribution (Laplace)
#mu_0 = 1  # Zeroth moment (total mass)
#mu_1 = 0  # First moment (mean)
#mu_2 = 2*B**2  # Second moment (variance)
#mu_3 = 0  # Third moment (skewness)
#mu_4 = 24*B**4  # Fourth moment (kurtosis)

moment_values = {
    sp.Symbol("mu_0"): 1,
    sp.Symbol("mu_1"): 0,
    sp.Symbol("mu_2"): 2*B**2,
    sp.Symbol("mu_3"): 0,
    sp.Symbol("mu_4"): 24*B**4,
}

# Construct the moment matrix M, which for a p'th degree polynomial is a (p+1)x(p+1) matrix, 
# where the first row is [mu_0, mu_1, ..., mu_p], 
# the second row is [0, (1 choose 1)*mu_0, (2 choose 1)*mu_1, ..., (p choose 1)*mu_{p-1}],
# the third row is [0, 0, (2 choose 2)*mu_0, (3 choose 2)*mu_1, ..., (p choose 2)*mu_{p-2}],
# and so on, s.t. last row is [0, 0, ..., 0, (p choose p)*mu_0  = mu_0]
def moment_matrix(p):
    M = sp.zeros(p+1, p+1)

    for i in range(p+1):
        for j in range(i, p+1):
            M[i, j] = sp.binomial(j, i) * sp.symbols(f"mu_{j-i}")

    return M

# define inverse
def moment_matrix_inverse(p):
    M = moment_matrix(p)
    return M.inv()

# define the vector of coefficients for the polynomial, given the polynomial 
# and the degree p (this needs to match ofcause, even though it is a bit hardcoded), 
# the vector is ordered as the vector t = [t_0, t_1, ..., t_p], 
# so the vector will be [e, d, c, b, a] for the fourth degree polynomial, and so on.
def coefficient_vector(p, polynomial):
    coeffs = sp.Poly(polynomial, q).all_coeffs()  # Get coefficients in descending order
    coeffs = [sp.simplify(coeff) for coeff in coeffs]  # Simplify coefficients
    return sp.Matrix(coeffs)

# printing functions for printing the moment matrix and its inverse in a nice format
# s.t. I can visually verify: 
def print_moment_matrix(p):
    M = moment_matrix(p)
    print(f"Moment Matrix M for p={p}:")
    sp.pprint(M)

def print_moment_matrix_inverse(p):
    M_inv = moment_matrix_inverse(p).subs(moment_values)
    print(f"Inverse of Moment Matrix M^-1 for p={p}:")
    sp.pprint(M_inv)

def print_coefficient_vector(p, polynomial):
    coeffs = coefficient_vector(p, polynomial)
    print(f"Coefficient vector t for p={p} and polynomial {polynomial}:")
    sp.pprint(coeffs)

def main():
    p = 4  # Degree of the polynomial
    print_moment_matrix(p)
    print_moment_matrix_inverse(p)
    print_coefficient_vector(p, f_fourth_degree)

if __name__ == "__main__":
    main()