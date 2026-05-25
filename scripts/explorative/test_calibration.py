"""
Testing the calibration of the Gaussian noise model, found in
    dp_calibration/gaussian.py, 
by comparing the calibrated sigma to the original formula for Gaussian noise in differential privacy.
The difference in sigma SHOULD be big, and the calibrated sigma should be much
smaller than the original sigma, especially for smaller values of epsilon.
"""

from dp_calibration.gaussian import analytic_gaussian_sigma, original_gaussian_sigma

epsilons = [0.1, 0.5,1.0, 2.0, 5.0]
delta = 1e-6
Delta = 1

for epsilon in epsilons:
    sigma_calibrated = analytic_gaussian_sigma(epsilon, delta, Delta)
    sigma_original = original_gaussian_sigma(epsilon, delta, Delta)
    print(f"original sigma for ε={epsilon} =", sigma_original)
    print(f"calibrated sigma for ε={epsilon} =", sigma_calibrated)
