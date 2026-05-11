import sympy as sp

from plotting.plot_relative_laplace_vs_gaussian import plot_plots

def main():
    q = sp.Symbol("q", real=True)

    polynomials = {
        "q^2": q**2,
        "chebyshev T_3": sp.chebyshevt(3, q),
        "q^5 + q^4 + q^3 + q^2": q**5 + q**4 + q**3 + q**2,
    }
    # The values of q to plot / the range of q values. 
    q_values = [-10, -5, 0, 5, 10] # values of q to plot (if we want to plot over epsilon instead of q)
    q_range = (-10, 10) # range of q values for plotting (if we want to plot over q instead of epsilon)

    # The range of epsilon on the x-axis / the values of epsilon to plot.
    epsilon_range = (0.1, 5.0)#(0.1, 5.0) # range of epsilon values for plotting
    
    # (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.0)
    epsilon_values = [sp.Rational(3,2)]
    #epsilon_values = [sp.Rational(1,10), sp.Rational(1,4), sp.Rational(1,2), sp.Rational(3,4), 1, sp.Rational(3,2), 2] # [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
    # epsilon_values = [sp.Rational(1,10), sp.Rational(1,2), 1, sp.Rational(3,2), 4] # [0.1, 0.5, 0.9, 1, 1.5, 4]

    # The fixed values of Delta for plotting; often Dela = 1 is used for simplicity, w.o. loss of generality, since the noise parameter Beta = Delta / epsilon. 
    Delta_values = [1] # [0.5, 1, 2]

    # the value of delta for the Gaussian noise. We can set it to a very small value to approximate pure DP, since the Laplace mechanism is pure DP.
    delta_value = 1e-10 # must be smaller than 1e-5

    # Which metrics to plot The "ratio" means the unbiased / naive and the "relative" means (unbiased - naive) / naive. 
    metrics = [
        "variance_ratio", 
        "variance_relative", 
        "mse_ratio", 
        "mse_relative"
    ]

    # What each single plot should contain. 
    # The first part indicates the curves to plot together, and the second part indicates the fixed paramter for the curves. 
    # - "q_by_degree" means that for each polynomial f, we make one plot with the curves for different values of q, with epsilon on the x-axis.
    # - "poly_by_q" means that for each value of q, we plot the curves for different polynomials f in the same plot, with epsilon on the x-axis.
    # - "epsilon_by_poly" means that for each polynomial, we plot the curves for different values of epsilon in the same plot, with q on the x-axis.
    each_plot = [
        "q_by_poly", 
        "poly_by_q", 
        "epsilon_by_poly"
    ] 

    # The path to save the plots. The code will create subfolders for variance and MSE plots.
    path_for_plots = "plots/laplace_vs_gaussian"

    plot_plots(
        q_values=q_values,
        q_range=q_range,
        epsilon_values=epsilon_values,
        epsilon_range=epsilon_range,
        polynomials=polynomials,
        Delta_values=Delta_values,
        delta_value=delta_value,
        metrics=metrics,
        each_plot=each_plot,
        path_for_plots=path_for_plots
    )

if __name__ == "__main__":
    main()