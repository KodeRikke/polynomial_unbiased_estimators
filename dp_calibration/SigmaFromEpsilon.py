import sympy as sp
from dp_calibration.gaussian import calibrateAnalyticGaussianMechanism

class SigmaFromEpsilon(sp.Function):
    """
    Symbolic placeholder for sigma(epsilon, delta, Delta).
    It evaluates numerically using the analytical Gaussian calibration.
    """

    @classmethod
    def eval(cls, epsilon, delta, Delta):
        # Never evaluate symbolically
        return None

    @staticmethod
    def numeric(epsilon_value, delta_value, Delta_value):
        return calibrateAnalyticGaussianMechanism(
            float(epsilon_value),
            float(delta_value),
            float(Delta_value)
        )