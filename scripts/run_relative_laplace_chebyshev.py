import sympy as sp

from plotting.plot_relative_laplace_chebyshev import plot_plots

def main():
    # The values of q to plot / the range of q values. 
    q_values = [0, sp.Rational(1,4), sp.Rational(1,2), sp.Rational(3,4), 1] # values of q to plot (if we want to plot over epsilon instead of q)
    q_range = (0.0, 1.0) # range of q values for plotting (if we want to plot over q instead of epsilon)

    # The range of epsilon on the x-axis / the values of epsilon to plot.
    epsilon_range = (0.1, 2.0)#(0.1, 5.0) # range of epsilon values for plotting
    
    # (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.0)
    epsilon_values = [sp.Rational(1,10), sp.Rational(1,4), sp.Rational(1,2), sp.Rational(3,4), 1, sp.Rational(3,2), 2] # [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
    # epsilon_values = [sp.Rational(1,10), sp.Rational(1,2), 1, sp.Rational(3,2), 4] # [0.1, 0.5, 0.9, 1, 1.5, 4]

    # The degrees of the Chebyshev polynomials. Note that the variance and MSE gap grows with the degree, so the range of the plots might need adjusting for higher degrees.
    chebyshev_degrees = [2, 3, 4, 5, 6] 
    # The fixed values of Delta for plotting; often Dela = 1 is used for simplicity, w.o. loss of generality, since the noise parameter Beta = Delta / epsilon. 
    Delta_values = [0.9]# [0.5, 1, 2]

    # Which metrics to plot The "ratio" means the unbiased / naive and the "relative" means (unbiased - naive) / naive. 
    metrics = [
        "variance_ratio", 
        "variance_relative", 
        "mse_ratio", 
        "mse_relative"
    ]

    # What each single plot should contain. 
    # The first part indicates the curves to plot together, and the second part indicates the fixed paramter for the curves. 
    # - "q_by_degree" means that for each degree of Chebyshev polynomials, we make one plot with the curves for different values of q, with epsilon on the x-axis.
    # - "degree_by_q" means that for each value of q, we plot the curves for different degrees of Chebyshev polynomials in the same plot, with epsilon on the x-axis.
    # - "epsilion_by_degree" means that for each degree of Chebyshev polynomials, we plot the curves for different values of epsilon in the same plot, with q on the x-axis.
    each_plot = [

        "q_by_degree", 
        "degree_by_q", 
        "epsilon_by_degree"
    ] 

    # The path to save the plots. The code will create subfolders for variance and MSE plots.
    path_for_plots = "plots"

    plot_plots(
        q_values=q_values,
        q_range=q_range,
        epsilon_values=epsilon_values,
        epsilon_range=epsilon_range,
        chebyshev_degrees=chebyshev_degrees,
        Delta_values=Delta_values,
        metrics=metrics,
        each_plot=each_plot,
        path_for_plots=path_for_plots
    )

if __name__ == "__main__":
    main()