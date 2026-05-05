import sympy as sp

from plotting.direct_laplace_vs_gaussian import plot_plots


def shared_polynomials():
    q = sp.Symbol("q", real=True)

    return {
        "q^2": q**2,
        "q^3": q**3,
        "chebyshev T_3": sp.chebyshevt(3, q),
    }

# the value of delta for the Gaussian noise. We can set it to a very small value to approximate pure DP, since the Laplace mechanism is pure DP.
delta_value = 1e-10 # must be smaller than 1e-5

def experiment_a_fixed_q_vary_epsilon(polynomials):
    """
    Experiment A:
    Fixed q, vary epsilon.

    Purpose:
    Shows how privacy strength / noise scale affects variance and MSE.

    x-axis = epsilon
    one plot per polynomial and q value
    curves = Laplace naive, Laplace unbiased, Gaussian naive, Gaussian unbiased
    """
    plot_plots(
        q_values=[-10, -5, 0, 5, 10],
        q_range=(-10, 10),  # not central for this experiment, but harmless if required
        epsilon_values=[sp.Rational(1, 2), 1, 2],  # not central here
        epsilon_range=(0.5, 5.0),
        polynomials=polynomials,
        Delta_values=[1],
        delta_value=delta_value,
        metrics=[
            "variance",
            "mse",
        ],
        each_plot=[
            "direct_by_q_and_poly",
        ],
        path_for_plots="plots/laplace_vs_gaussian_direct/experiment_a_fixed_q_vary_epsilon",
    )


def experiment_b_fixed_epsilon_vary_q(polynomials):
    """
    Experiment B:
    Fixed epsilon, vary q.

    Purpose:
    Shows how estimator variance/MSE changes with the true query value q.

    x-axis = q
    one plot per polynomial and epsilon value
    curves = Laplace naive, Laplace unbiased, Gaussian naive, Gaussian unbiased
    """
    plot_plots(
        q_values=[-5, 0, 5],  # not central here
        q_range=(-10, 10),
        epsilon_values=[
            sp.Rational(1, 2),
            1,
            2,
            5,
        ],
        epsilon_range=(0.5, 5.0),  # not central here
        polynomials=polynomials,
        Delta_values=[1],
        delta_value=delta_value,
        metrics=[
            "variance",
            "mse",
        ],
        each_plot=[
            "direct_by_epsilon_and_poly",
        ],
        path_for_plots="plots/laplace_vs_gaussian_direct/experiment_b_fixed_epsilon_vary_q",
    )


def experiment_c_focused_noise_comparison(polynomials):
    """
    Experiment C:
    Focused Laplace-vs-Gaussian comparison.

    Purpose:
    Uses a smaller, more readable range to compare the noise distributions directly.

    This is still direct comparison, but with less extreme values so the plots
    are easier to interpret.
    """
    plot_plots(
        q_values=[-2, -1, 0, 1, 2],
        q_range=(-2, 2),
        epsilon_values=[
            sp.Rational(1, 2),
            1,
            2,
        ],
        epsilon_range=(0.75, 4.0),
        polynomials=polynomials,
        delta_value=delta_value,
        Delta_values=[1],
        metrics=[
            "variance",
            "mse",
        ],
        each_plot=[
            "direct_by_q_and_poly",
            "direct_by_epsilon_and_poly",
        ],
        path_for_plots="plots/laplace_vs_gaussian_direct/experiment_c_focused_comparison",
    )


def main():
    polynomials = shared_polynomials()

    experiment_a_fixed_q_vary_epsilon(polynomials)
    experiment_b_fixed_epsilon_vary_q(polynomials)
    experiment_c_focused_noise_comparison(polynomials)


if __name__ == "__main__":
    main()