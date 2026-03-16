import sympy as sp
from functools import lru_cache

__all__ = ["NoiseModel", 
           "LaplaceNoiseModel", 
           "GaussianNoiseModel"
           ]

"""
This module defines the noise models used in the estimation system, following the Strategy design pattern.
The NoiseModel class is an abstract base class that defines the interface for different noise models,
the LaplaceNoiseModel and GaussianNoiseModel are concrete implementations of the NoiseModel strategy, 
responsible for calculating the moments and unbiased transforms specific to their respective noise distributions.
"""

# --- Strategy NoiseModel ---
class NoiseModel:
    """
    The strategy "NoiseModel" defines the interface for different noise models, 
    passing to concrete strategies s.a. Laplace Noise or Gaussian Noise.
    OBS! The noise model can only be of distributions with known moments.
    It requires the implementation of the following methods:
        moment(n, q): calculates the n-th moment of the noise distribution given q.
        unbiased_transform(f, x): returns a function g(x) such that E[g(q + noise)] = f(q) for the given polynomial f.
        clear_cache(): clears the cache of the moment method, if implemented with caching.
        cache_info(): returns the cache information of the moment method, if implemented with caching.
    """
    
    def moment(self, n, q):
        # \mu_n(q) = E[(q + noise)^n] for x = q + noise
        raise NotImplementedError("Subclasses must implement this method")
    
    def unbiased_transform(self, f, x):
        # Returns g(x) s.t. E[g(q + noise)] = f(q)
        raise NotImplementedError("Subclasses must implement this method")
    
    def clear_cache(self):
        pass

    def cache_info(self):
        pass

# --- Concrete Strategy LaplaceNoiseModel ---
class LaplaceNoiseModel(NoiseModel):
    """
    The concrete strategy Laplace Noise Model inherits from NoiseModel and is responsible for
    calculating moments and the unbiased transform for Laplace noise.
    When initializing the LaplaceNoiseModel, the following needs to be provided:
        delta: a variable s.a. an int, float or sympy symbol representing the sensitivity of the query
        epsilon: a variable s.a. an int, float or sympy symbol representing the privacy budget / noise scale
    """
    def __init__(self, delta, epsilon):
        self.delta = sp.sympify(delta)
        self.epsilon = sp.sympify(epsilon)

    """
    The _moment_about_zero method is a helper method to calculate the k-th moment of
    the Laplace noise distribution, E[noise^k], which is used in the central moment calculation.
    Because Laplace noise has mean 0, the k-th moment about zero is the same as the k-th central moment.

    Input: k is a non-negative integer representing the order of the moment
    Output: the k-th moment of the noise distribution, which is (delta/epsilon)^k * k! if k is even, and 0 if k is odd.
    """
    def _moment_about_zero(self, k):
        if k % 2 == 0: # even k
            return (self.delta / self.epsilon)**k * sp.factorial(k) 
        else: # odd k
            return 0 # if not caught by caller

    """ 
    The  moment method calculates the i-th central moment of the released statistic, E[(q + noise)^i], using the binomial expansion and the moments of the noise distribution.
    It implements the formula from equation (12) in the thesis prep-project.
    It is the central moment because it is the moment of the released statistic x = q + noise, which is centered around q, as the noise has mean 0.

    Input: 
    i: a non-negative integer representing the order of the moment, 
    q: a variable representing the original statistic.
    Output: the i-th CENTRAL moment of the released statistic, E[(q + noise)^i], which is calculated using the binomial expansion and the moments of the noise distribution.
    """
    @lru_cache(maxsize=4096) # maybe set roof
    def moment(self, i, q):
        i = int(i) # for caching
        expr = 0
        for k in range(0, i + 1, 2): # only even k contributes
            expr += sp.binomial(i, k) * q**(i - k) * self._moment_about_zero(k)
        return expr
    
    """ 
    The unbiased_transform method takes a polynomial function f in q and returns the unbiased estimator g(x) in x.

    Input: 
    f: a polynomial function in q, 
    x: the variable representing the observed statistic.
    Output: the unbiased estimator g(x) = f(x) - (delta/epsilon)^2 f''(x), s.t. E[g(x)] = f(q).
    """
    def unbiased_transform(self, f, x):
        f_dd = sp.diff(f, x, 2)  # second derivative
        g = f - (self.delta / self.epsilon)**2 * f_dd
        return g
    
    def clear_cache(self):
        self.moment.cache_clear()

    def cache_info(self):
        return self.moment.cache_info()

# --- Concrete Strategy GaussianNoiseModel ---
class GaussianNoiseModel(NoiseModel):
    def __init__(self, sigma):
        self.sigma = sp.sympify(sigma)

    # To be implemented if needed
    # def gaussian_moment(self, k):
    #     pass

    # @lru_cache(maxsize=None) # maybe set roof
    # def moment_of_gaussian(self, i, q):
    #     pass   
