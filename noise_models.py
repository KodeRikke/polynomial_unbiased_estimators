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
        clear_cache(): clears the cache of the moment method, if implemented with caching.
        cache_info(): returns the cache information of the moment method, if implemented with caching.
    """
    
    def moment(self, n, q):
        # \mu_n(q) = E[(q + noise)^n] for x = q + noise
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

    def _moment_about_zero(self, k):
        """
        The _moment_about_zero method is a helper method to calculate the k-th moment of
        the Laplace noise distribution, E[noise^k], which is used in the central moment calculation.
        Because Laplace noise has mean 0, the RAW k-th moment about zero is EQUAL to the k-th CENTRAL moment.
    
        Input: k is a non-negative integer representing the order of the moment
        Output: the k-th moment of the noise distribution, which is (delta/epsilon)^k * k! if k is even, and 0 if k is odd.
        """
        if k % 2 == 0: # even k
            return (self.delta / self.epsilon)**k * sp.factorial(k) 
        else: # odd k
            return 0 # if not caught by caller

    @lru_cache(maxsize=4096) # maybe set roof
    def moment(self, i, q):
        """ 
        The  moment method calculates the i-th raw moment of the released statistic, E[(q + noise)^i], 
        using the binomial expansion and the moments of the noise distribution.
        It implements the formula from equation (12) in the thesis prep-project.
    
        Note that this is a raw moment, not a central moment. Since the released
        statistic x = q + noise has mean q when the noise has mean 0, the corresponding
        i-th central moment would be E[(x - q)^i].
    
        Input: 
        i: a non-negative integer representing the order of the moment, 
        q: a variable representing the original statistic.
        Output: the i-th raw moment of the released statistic, E[(q + noise)^i], which is calculated using the binomial expansion and the moments of the noise distribution.
        """
        i = int(i) # for caching
        expr = 0
        for k in range(0, i + 1, 2): # only even k contributes
            expr += sp.binomial(i, k) * q**(i - k) * self._moment_about_zero(k)
        return expr
    
    def unbiased_transform(self, f, x):
        """ 
        The unbiased_transform method takes a polynomial function f in q and returns the unbiased estimator g(x) in x,
        s.t. E[g(q + noise)] = f(q) for the given polynomial f.

        Input: 
        f: a polynomial function in q, 
        x: the variable representing the observed statistic.
        Output: the unbiased estimator g(x) = f(x) - (delta/epsilon)^2 f''(x), s.t. E[g(x)] = f(q).
        """
        f_dd = sp.diff(f, x, 2)  # second derivative
        g = f - (self.delta / self.epsilon)**2 * f_dd
        return g
    
    def clear_cache(self):
        self.moment.cache_clear()

    def cache_info(self):
        return self.moment.cache_info()

# --- Concrete Strategy GaussianNoiseModel ---
class GaussianNoiseModel(NoiseModel):
    """
    The concrete strategy Gaussian Noise Model inherits from NoiseModel and is responsible for
    calculating moments and the unbiased transform for Gaussian noise.
    When initializing the GaussianNoiseModel, the following needs to be provided:
        sigma: a (positive) variable s.a. an int, float or sympy symbol representing the standard deviation of the noise
    """
    def __init__(self, sigma):
        self.sigma = sp.sympify(sigma)

    def _moment_about_zero(self, k):
        """
        The _moment_about_zero method is a helper method to calculate the k-th moment of
        the Gaussian noise distribution. The plain central moments of a Gaussian distribution 
        with mean 0, standard deviation sigma, and for any non-negative integer k are given by:
        - 0 if k is odd
        - sigma^k * (k-1)!! if k is even,
          where (k-1)!! is the double factorial of (k-1), defined as the product of all numbers
          from 1 to (k-1) that have the same parity (even or odd) as (k-1).

        Input: k is a non-negative integer representing the order of the moment
        Output: the k-th moment of the noise distribution, which is 0 if k is odd, and sigma^k * (k-1)!! if k is even.
        """
        if k % 2 == 0: # even k
            return (self.sigma**k * sp.factorial2(k - 1)) if k > 0 else 1 # (k-1)!! = factorial2(k-1) in sympy, and for k=0, the moment is 1 (since E[noise^0] = 1)
        else: # odd k
            return 0

    @lru_cache(maxsize=None) # maybe set roof
    def moment(self, i, q):
        """ 
        The  moment method calculates the i-th raw moment of the released statistic, E[(q + noise)^i], 
        using the binomial expansion and the moments of the noise distribution.

        Note that this is a raw moment, not a central moment. Since the released
        statistic x = q + noise has mean q when the noise has mean 0, the corresponding
        i-th central moment would be E[(x - q)^i].

        Input: 
        i: a non-negative integer representing the order of the moment, 
        q: a variable representing the original statistic.
        Output: the i-th raw moment of the released statistic, E[(q + noise)^i], which is calculated using the binomial expansion and the moments of the noise distribution.
        """
        i = int(i) # for caching
        expr = 0
        for k in range(0, i + 1, 2): # only even k contributes
            expr += sp.binomial(i, k) * q**(i - k) * self._moment_about_zero(k)
        return expr

    def clear_cache(self):
        self.moment.cache_clear()

    def cache_info(self):
        return self.moment.cache_info()