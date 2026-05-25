# Color scheme #1 
colors_poly = {
    "Quadratic basecase": "tab:blue",
    "Cubic basecase": "tab:red",
    "Cubic coefficients": "tab:pink",
    "Chebyshev T3": "tab:cyan",
}

import matplotlib.pyplot as plt
import numpy as np

epsilons = [0.1, 0.5, 1, 2, 5]

# Color scheme #2
colors_eps = plt.cm.viridis(np.linspace(0.15, 0.85, len(epsilons)))

for eps, color in zip(epsilons, colors_eps):
    ax.plot(q, laplace_values[eps],
            color=color,
            linestyle="-",
            label=f"Laplace ε={eps}")

    ax.plot(q, gaussian_values[eps],
            color=color,
            linestyle="--",
            label=f"Gaussian ε={eps}")