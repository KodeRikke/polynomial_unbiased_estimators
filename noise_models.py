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
        moment(n, q): calculates the n-th raw moment of the released statistics, q + noise.
        unbiased_transform(f, x): takes a polynomial function f in q and returns the unbiased 
        estimator g(x) in x, s.t. E[g(q + noise)] = f(q) for the given polynomial f.
        clear_cache(): clears the cache of the moment method, if implemented with caching.
        cache_info(): returns the cache information of the moment method, if implemented with caching.
    """
    
    def moment(self, n, q):
        # \mu_n(q) = E[(q + noise)^n] for x = q + noise
        raise NotImplementedError("Subclasses must implement this method")

    def unbiased_transform(self, f, x):
        return self._general_unbiased_transform(self, f, x)
    
    @staticmethod
    def _general_unbiased_transform(self, f, x):
        """"
        The _general_unbiased_transform method is a helper-function and a 
        general implementation of the unbiased transform for any noise model
        with known raw moments and a polynomial function f. It guilds and 
        solves the linear system from Calmon et al., THM 22, to find the 
        unbiased estimator g(x) s.t. E[g(q + noise)] = f(q).
        This is done by defining the coefficients of f as the vector b, 
        then building the matrix M from the raw moments of the noise distribution, 
        and finally solving the linear system M a = b 
        for the coefficients a of the unbiased estimator g(x).

        This is a general implememntation that works for any noise model
        with known raw moments and any polynomial function f, 
        but it is not optimized for specific noise models.

        Input: 
        - f: a polynomial function in q,
        - x: the variable representing the observed statistic.
        Output: the unbiased estimator g(x) = sum_i a_i x^i, s.t. E[g(x)] = f(q).

        """
        # symbolic symbols
        q = sp.Symbol('q', real=True)

        # extract degree from f
        f = sp.sympify(f)
        poly = sp.Poly(f, x)
        degree = poly.degree()

        # extract coefficients of f(q), from highest degree to lowest degree
        # this should hold for any polynomial, 
        # as long as the degree is correctly extracted
        # b[k] is coefficients of q^k
        f_in_q = f.subs(x, q)
        f_poly = sp.Poly(f_in_q, q)
        b = [
            f_poly.coeff_monomial(q**k) for k in range(degree + 1)
        ]
        # M[k, i] is the coefficient of q^k in E[(q + noise)^i]
        M = sp.zeros(degree + 1, degree + 1)

        # for each COLUMN in M
        for i in range(degree + 1): 
            # calculate the raw moment, E[(q + noise)^i]
            mu = self.moment(i, q)
            # express mu as a polynomial in q and extract coefficients from highest to lowest degree
            mu_poly = sp.Poly(mu, q) 

            # for each ROW in M
            for k in range(degree + 1):
                # fill M with the coefficients of the raw moment
                M[k, i] = mu_poly.coeff_monomial(q**k)

        # solve the linear system for a
        a = M.LUsolve(sp.Matrix(b))

        # build the unbiased estimator g(x) = sum_i a_i x^i
        g = sum(a[i] * x**i for i in range(degree + 1))
        
        return sp.expand(g)
    
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

    def _noise_central_moment(self, k):
        """
        _moment_about_zero(k) returns the k-th central moment of the noise.

        The _noise_central_moment method is a helper method to calculate the k-th central 
        moment of the Laplace noise distribution, E[noise^k], which is used in the 
        raw moment calculation for the RELEASED STATISTICS, E[(q + noise)^i].
        Because the noise has mean 0, this equals the k-th raw moment E[noise^k].    

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
        The  moment method calculates the i-th 
        RAW moment of the RELEASED STATISTICS, E[(q + noise)^i], 
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
            expr += sp.binomial(i, k) * q**(i - k) * self._noise_central_moment(k)
        return expr
    
    def unbiased_transform(self, f, x):
        """ 
        This Laplace-specific implementation of the unbiased_transform method 
        replaces the general linear system solution with the (faster)
        closed-form solution for the unbiased estimator under Laplace noise.

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

    def _noise_central_moment(self, k):
        """
        The _noise_central_moment method is a helper method to calculate the k-th 
        central moment of the Gaussian noise distribution, E[noise^k]. 
        The plain central moments of a Gaussian distribution with mean 0, 
        standard deviation sigma, and for any non-negative integer k are given by:
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
        The  moment method calculates the i-th 
        RAW moment of the RELEASED STATISTIC, E[(q + noise)^i], 
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
            expr += sp.binomial(i, k) * q**(i - k) * self._noise_central_moment(k)
        return expr

    def clear_cache(self):
        self.moment.cache_clear()

    def cache_info(self):
        return self.moment.cache_info()