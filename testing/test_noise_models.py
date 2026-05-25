"""Unit tests for noise_models.py.

Coverage:
- Central moments for Laplace and Gaussian (even/odd, symbolic)
- Raw moments E[(q+noise)^i] via binomial expansion
- Laplace unbiased transform (closed-form oracle)
- Gaussian unbiased transform (general linear system via base class)
- Cross-implementation consistency: Laplace closed-form vs general
- Unbiasedness invariant: E[g(q+noise)] == f(q)
- Cache interface (clear_cache, cache_info)
- Edge cases: degree-0 polynomial, symbolic parameters
"""
import pytest
import sympy as sp

from noise_models import LaplaceNoiseModel, GaussianNoiseModel, NoiseModel

q = sp.Symbol("q", real=True)
x = sp.Symbol("x", real=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def laplace_unit():
    """Laplace with Delta=1, epsilon=1 → s=1."""
    return LaplaceNoiseModel(Delta=1, epsilon=1)


@pytest.fixture
def laplace_half():
    """Laplace with Delta=1, epsilon=2 → s=0.5."""
    return LaplaceNoiseModel(Delta=1, epsilon=2)


@pytest.fixture
def laplace_sym():
    """Laplace with symbolic positive Delta and epsilon."""
    D = sp.Symbol("D", positive=True)
    e = sp.Symbol("e", positive=True)
    return LaplaceNoiseModel(Delta=D, epsilon=e)


@pytest.fixture
def gaussian_unit():
    """Gaussian with sigma=1."""
    return GaussianNoiseModel(sigma=1)


@pytest.fixture
def gaussian_sym():
    """Gaussian with symbolic positive sigma."""
    sigma = sp.Symbol("sigma", positive=True)
    return GaussianNoiseModel(sigma=sigma)


# ── Laplace central moments ───────────────────────────────────────────────────


class TestLaplaceCentralMoments:
    @pytest.mark.parametrize("k,expected", [
        (0, 1),
        (2, 2),    # (1/1)^2 * 2! = 2
        (4, 24),   # (1/1)^4 * 4! = 24
        (6, 720),  # (1/1)^6 * 6! = 720
    ])
    def test_even(self, laplace_unit, k, expected):
        result = laplace_unit._noise_central_moment(k)
        assert sp.simplify(result - expected) == 0

    @pytest.mark.parametrize("k", [1, 3, 5, 7])
    def test_odd_is_zero(self, laplace_unit, k):
        assert laplace_unit._noise_central_moment(k) == 0

    def test_symbolic_k2(self, laplace_sym):
        D, e = laplace_sym.Delta, laplace_sym.epsilon
        result = laplace_sym._noise_central_moment(2)
        expected = (D / e) ** 2 * sp.Integer(2)
        assert sp.simplify(result - expected) == 0

    def test_symbolic_odd(self, laplace_sym):
        assert laplace_sym._noise_central_moment(3) == 0

    def test_k0_is_one(self, laplace_unit):
        assert laplace_unit._noise_central_moment(0) == 1


# ── Gaussian central moments ──────────────────────────────────────────────────


class TestGaussianCentralMoments:
    @pytest.mark.parametrize("k,expected", [
        (0, 1),
        (2, 1),    # sigma^2 * 1!! = 1
        (4, 3),    # sigma^4 * 3!! = 3
        (6, 15),   # sigma^6 * 5!! = 15
        (8, 105),  # sigma^8 * 7!! = 105
    ])
    def test_even(self, gaussian_unit, k, expected):
        result = gaussian_unit._noise_central_moment(k)
        assert sp.simplify(result - expected) == 0

    @pytest.mark.parametrize("k", [1, 3, 5, 7])
    def test_odd_is_zero(self, gaussian_unit, k):
        assert gaussian_unit._noise_central_moment(k) == 0

    def test_symbolic_k2(self, gaussian_sym):
        sigma = gaussian_sym.sigma
        result = gaussian_sym._noise_central_moment(2)
        assert sp.simplify(result - sigma**2) == 0

    def test_symbolic_k4(self, gaussian_sym):
        sigma = gaussian_sym.sigma
        result = gaussian_sym._noise_central_moment(4)
        assert sp.simplify(result - 3 * sigma**4) == 0


# ── Raw moments via binomial expansion (Laplace) ──────────────────────────────


class TestLaplaceRawMoments:
    # With s=1: E[(q+noise)^i] via binomial + Laplace moments
    @pytest.mark.parametrize("i,expected", [
        (0, sp.Integer(1)),
        (1, q),
        (2, q**2 + 2),
        (3, q**3 + 6*q),
        (4, q**4 + 12*q**2 + 24),
    ])
    def test_moment(self, laplace_unit, i, expected):
        result = laplace_unit.moment(i, q)
        assert sp.expand(result - expected) == 0

    def test_moment0_is_one(self, laplace_unit):
        """E[1] = 1 for any model."""
        assert sp.expand(laplace_unit.moment(0, q) - 1) == 0

    def test_moment1_is_q(self, laplace_unit):
        """E[x] = q because Laplace noise has mean zero."""
        assert sp.expand(laplace_unit.moment(1, q) - q) == 0

    def test_second_moment_s_half(self, laplace_half):
        """With s=0.5: E[x²] = q² + 2*(0.5)² = q² + 0.5."""
        result = laplace_half.moment(2, q)
        expected = q**2 + sp.Rational(1, 2)
        assert sp.expand(result - expected) == 0

    def test_symbolic_moment1(self, laplace_sym):
        """E[x] = q regardless of Delta/epsilon."""
        result = laplace_sym.moment(1, q)
        assert sp.expand(result - q) == 0


# ── Raw moments via binomial expansion (Gaussian) ─────────────────────────────


class TestGaussianRawMoments:
    @pytest.mark.parametrize("i,expected", [
        (0, sp.Integer(1)),
        (1, q),
        (2, q**2 + 1),
        (3, q**3 + 3*q),
        (4, q**4 + 6*q**2 + 3),
    ])
    def test_moment(self, gaussian_unit, i, expected):
        result = gaussian_unit.moment(i, q)
        assert sp.expand(result - expected) == 0

    def test_moment1_is_q(self, gaussian_unit):
        assert sp.expand(gaussian_unit.moment(1, q) - q) == 0

    def test_symbolic_moment2(self, gaussian_sym):
        sigma = gaussian_sym.sigma
        result = gaussian_sym.moment(2, q)
        expected = q**2 + sigma**2
        assert sp.expand(result - expected) == 0


# ── Laplace unbiased transform (oracle) ───────────────────────────────────────


class TestLaplaceUnbiasedTransform:
    """g = f(x) - s²·f''(x) with s=1."""

    @pytest.mark.parametrize("f,expected", [
        (x,                x),
        (x**2,             x**2 - 2),
        (x**3,             x**3 - 6*x),
        (x**4,             x**4 - 12*x**2),
        (3*x**2 + x,       3*x**2 + x - 6),
        (x**3 - x,         x**3 - 7*x),
    ])
    def test_transform(self, laplace_unit, f, expected):
        result = laplace_unit.unbiased_transform(f, x)
        assert sp.expand(result - expected) == 0

    def test_constant_unchanged(self, laplace_unit):
        """f'' of a constant is 0, so the correction vanishes."""
        result = laplace_unit.unbiased_transform(sp.Integer(7), x)
        assert sp.expand(result - 7) == 0

    def test_s_half_quadratic(self, laplace_half):
        """With s=0.5: g = x² - 2*(0.5)² = x² - 0.5."""
        result = laplace_half.unbiased_transform(x**2, x)
        expected = x**2 - sp.Rational(1, 2)
        assert sp.expand(result - expected) == 0


# ── Cross-implementation: Laplace closed-form vs general linear system ─────────


class TestCrossImplementation:
    @pytest.mark.parametrize("f", [
        x,
        x**2,
        x**3,
        2*x**2 - 3*x + 1,
        x**3 + 2*x**2 - x,
    ])
    def test_fast_vs_general(self, laplace_unit, f):
        g_fast = laplace_unit.unbiased_transform(f, x)
        g_general = NoiseModel._general_unbiased_transform(laplace_unit, f, x)
        assert sp.expand(g_fast - g_general) == 0


# ── Unbiasedness invariant via moment computation ─────────────────────────────


class TestUnbiasednessInvariant:
    """E[g(q+noise)] == f(q) for both noise models."""

    @pytest.fixture
    def laplace_analyzer(self, laplace_unit):
        from dp_estimators import EstimatorAnalyzer
        return EstimatorAnalyzer(laplace_unit, "q", "x")

    @pytest.fixture
    def gaussian_analyzer(self, gaussian_unit):
        from dp_estimators import EstimatorAnalyzer
        return EstimatorAnalyzer(gaussian_unit, "q", "x")

    @pytest.mark.parametrize("f", [
        x,
        x**2,
        x**3,
        x**4,
        2*x**2 + 3*x + 1,
    ])
    def test_laplace(self, laplace_unit, laplace_analyzer, f):
        g = laplace_unit.unbiased_transform(f, x)
        mean_g = laplace_analyzer.mean(g)
        expected = f.subs(x, q)
        assert sp.expand(mean_g - expected) == 0

    @pytest.mark.parametrize("f", [
        x,
        x**2,
        x**3,
    ])
    def test_gaussian(self, gaussian_unit, gaussian_analyzer, f):
        g = gaussian_unit.unbiased_transform(f, x)
        mean_g = gaussian_analyzer.mean(g)
        expected = f.subs(x, q)
        assert sp.expand(mean_g - expected) == 0


# ── Cache interface ───────────────────────────────────────────────────────────


class TestCacheInterface:
    def test_laplace_clear_cache_no_error(self, laplace_unit):
        laplace_unit.moment(2, q)
        laplace_unit.clear_cache()

    def test_laplace_cache_info_has_hits(self, laplace_unit):
        laplace_unit.moment(2, q)
        info = laplace_unit.cache_info()
        assert hasattr(info, "hits")

    def test_laplace_cache_warm_on_repeat(self, laplace_unit):
        laplace_unit.clear_cache()
        laplace_unit.moment(3, q)
        info_before = laplace_unit.cache_info()
        laplace_unit.moment(3, q)
        info_after = laplace_unit.cache_info()
        assert info_after.hits > info_before.hits

    def test_gaussian_clear_cache_no_error(self, gaussian_unit):
        gaussian_unit.moment(2, q)
        gaussian_unit.clear_cache()

    def test_gaussian_cache_info_has_hits(self, gaussian_unit):
        gaussian_unit.moment(2, q)
        info = gaussian_unit.cache_info()
        assert hasattr(info, "hits")
