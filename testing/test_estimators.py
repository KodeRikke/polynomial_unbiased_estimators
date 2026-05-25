"""Unit tests for dp_estimators.py.

Coverage:
- EstimatorContext: naive substitution, unbiased transform
- EstimatorAnalyzer.mean: oracle values, constant, zero polynomial, non-poly error
- EstimatorAnalyzer.variance: constant=0, known values for Laplace/Gaussian
- EstimatorAnalyzer.bias: unbiased has zero bias, naive has known bias
- EstimatorAnalyzer.mse: mse = var + bias², parametrized
- ComparisonReport: required keys, mean_gap, unbiased mean = f(q), mse gap formula
- EstimatorSystem: facade dispatch, ValueError on unknown biasedness
- Unbiasedness invariant: parametrized over polynomial forms and both noise models
- MSE decomposition: var + bias² == mse for naive estimator
"""
import pytest
import sympy as sp

from noise_models import LaplaceNoiseModel, GaussianNoiseModel
from dp_estimators import (
    EstimatorSystem,
    ComparisonReport,
    EstimatorContext,
    EstimatorAnalyzer,
)

q = sp.Symbol("q", real=True)
x = sp.Symbol("x", real=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def laplace_unit():
    return LaplaceNoiseModel(Delta=1, epsilon=1)


@pytest.fixture
def gaussian_unit():
    return GaussianNoiseModel(sigma=1)


@pytest.fixture
def sys_laplace(laplace_unit):
    return EstimatorSystem(laplace_unit, "q", "x")


@pytest.fixture
def sys_gaussian(gaussian_unit):
    return EstimatorSystem(gaussian_unit, "q", "x")


@pytest.fixture
def ctx_laplace(laplace_unit):
    return EstimatorContext(laplace_unit, "q", "x")


@pytest.fixture
def ctx_gaussian(gaussian_unit):
    return EstimatorContext(gaussian_unit, "q", "x")


@pytest.fixture
def analyzer_laplace(laplace_unit):
    return EstimatorAnalyzer(laplace_unit, "q", "x")


@pytest.fixture
def analyzer_gaussian(gaussian_unit):
    return EstimatorAnalyzer(gaussian_unit, "q", "x")


# ── EstimatorContext: naive ───────────────────────────────────────────────────


class TestEstimatorContextNaive:
    def test_linear(self, ctx_laplace):
        result = ctx_laplace.naive(q)
        assert sp.expand(result - x) == 0

    def test_quadratic(self, ctx_laplace):
        result = ctx_laplace.naive(q**2)
        assert sp.expand(result - x**2) == 0

    def test_polynomial(self, ctx_laplace):
        f = q**3 + 2*q**2 - q + 5
        expected = x**3 + 2*x**2 - x + 5
        assert sp.expand(ctx_laplace.naive(f) - expected) == 0

    def test_constant_unchanged(self, ctx_laplace):
        assert sp.expand(ctx_laplace.naive(sp.Integer(42)) - 42) == 0

    def test_gaussian_naive_same_substitution(self, ctx_gaussian):
        f = q**2 + q
        expected = x**2 + x
        assert sp.expand(ctx_gaussian.naive(f) - expected) == 0


# ── EstimatorContext: unbiased ────────────────────────────────────────────────


class TestEstimatorContextUnbiased:
    def test_linear_no_correction(self, ctx_laplace):
        """f'' of a linear is 0, so g = x."""
        result = ctx_laplace.unbiased(q)
        assert sp.expand(result - x) == 0

    def test_quadratic(self, ctx_laplace):
        """g = x² - 2*(Δ/ε)² = x² - 2 for s=1."""
        result = ctx_laplace.unbiased(q**2)
        expected = x**2 - 2
        assert sp.expand(result - expected) == 0

    def test_cubic(self, ctx_laplace):
        """g = x³ - 6x for f=q³, s=1."""
        result = ctx_laplace.unbiased(q**3)
        expected = x**3 - 6*x
        assert sp.expand(result - expected) == 0

    def test_quartic(self, ctx_laplace):
        """g = x⁴ - 12x² for f=q⁴, s=1."""
        result = ctx_laplace.unbiased(q**4)
        expected = x**4 - 12*x**2
        assert sp.expand(result - expected) == 0

    def test_constant_unchanged(self, ctx_laplace):
        result = ctx_laplace.unbiased(sp.Integer(5))
        assert sp.expand(result - 5) == 0


# ── EstimatorAnalyzer.mean ────────────────────────────────────────────────────


class TestEstimatorAnalyzerMean:
    def test_constant(self, analyzer_laplace):
        result = analyzer_laplace.mean(sp.Integer(7))
        assert sp.expand(result - 7) == 0

    def test_x_gives_q(self, analyzer_laplace):
        """E[x] = q since Laplace noise has mean zero."""
        assert sp.expand(analyzer_laplace.mean(x) - q) == 0

    def test_x2_laplace(self, analyzer_laplace):
        """E[x²] = q² + 2 for s=1."""
        result = analyzer_laplace.mean(x**2)
        assert sp.expand(result - (q**2 + 2)) == 0

    def test_x3_laplace(self, analyzer_laplace):
        """E[x³] = q³ + 6q for s=1."""
        result = analyzer_laplace.mean(x**3)
        assert sp.expand(result - (q**3 + 6*q)) == 0

    def test_x2_gaussian(self, analyzer_gaussian):
        """E[x²] = q² + σ² = q² + 1 for sigma=1."""
        result = analyzer_gaussian.mean(x**2)
        assert sp.expand(result - (q**2 + 1)) == 0

    def test_x3_gaussian(self, analyzer_gaussian):
        """E[x³] = q³ + 3q for sigma=1."""
        result = analyzer_gaussian.mean(x**3)
        assert sp.expand(result - (q**3 + 3*q)) == 0

    def test_zero_polynomial(self, analyzer_laplace):
        result = analyzer_laplace.mean(sp.Integer(0))
        assert sp.expand(result) == 0

    def test_non_polynomial_raises(self, analyzer_laplace):
        with pytest.raises((ValueError, sp.PolynomialError)):
            analyzer_laplace.mean(sp.sin(x))

    def test_linear_combination(self, analyzer_laplace):
        """E[3x² - 2x + 1] = 3(q²+2) - 2q + 1 = 3q² - 2q + 7."""
        result = analyzer_laplace.mean(3*x**2 - 2*x + 1)
        expected = 3*q**2 - 2*q + 7
        assert sp.expand(result - expected) == 0


# ── EstimatorAnalyzer.variance ────────────────────────────────────────────────


class TestEstimatorAnalyzerVariance:
    def test_constant_is_zero(self, analyzer_laplace):
        assert sp.simplify(analyzer_laplace.variance(sp.Integer(5))) == 0

    def test_x_laplace(self, analyzer_laplace):
        """Var(x) = E[x²]-E[x]² = (q²+2)-q² = 2 = 2s² for s=1."""
        result = analyzer_laplace.variance(x)
        assert sp.simplify(result - 2) == 0

    def test_x_gaussian(self, analyzer_gaussian):
        """Var(x) = σ² = 1 for sigma=1."""
        result = analyzer_gaussian.variance(x)
        assert sp.simplify(result - 1) == 0

    def test_x2_at_q0(self, analyzer_laplace):
        """Var(x²) = 8q² + 20; at q=0 this is 20."""
        var_x2 = analyzer_laplace.variance(x**2)
        val = sp.simplify(var_x2.subs(q, 0))
        assert float(val) == pytest.approx(20.0)

    def test_nonneg_at_several_q(self, analyzer_laplace):
        """Variance of any estimator must be non-negative."""
        var_x3 = analyzer_laplace.variance(x**3)
        for q_val in (-2, -1, 0, 1, 2):
            val = float(sp.simplify(var_x3.subs(q, q_val)))
            assert val >= 0


# ── EstimatorAnalyzer.bias ────────────────────────────────────────────────────


class TestEstimatorAnalyzerBias:
    @pytest.mark.parametrize("f", [q, q**2, q**3, q**3 + 3*q**2])
    def test_unbiased_has_zero_bias_laplace(self, ctx_laplace, analyzer_laplace, f):
        g = ctx_laplace.unbiased(f)
        bias = analyzer_laplace.bias(g, f)
        assert sp.expand(bias) == 0

    @pytest.mark.parametrize("f", [q, q**2, q**3])
    def test_unbiased_has_zero_bias_gaussian(self, ctx_gaussian, analyzer_gaussian, f):
        g = ctx_gaussian.unbiased(f)
        bias = analyzer_gaussian.bias(g, f)
        assert sp.expand(bias) == 0

    def test_naive_bias_quadratic(self, ctx_laplace, analyzer_laplace):
        """Bias(x², q²) = E[x²]-q² = (q²+2)-q² = 2."""
        bias = analyzer_laplace.bias(ctx_laplace.naive(q**2), q**2)
        assert sp.expand(bias - 2) == 0

    def test_naive_bias_cubic(self, ctx_laplace, analyzer_laplace):
        """Bias(x³, q³) = E[x³]-q³ = 6q."""
        bias = analyzer_laplace.bias(ctx_laplace.naive(q**3), q**3)
        assert sp.expand(bias - 6*q) == 0

    def test_naive_bias_linear_is_zero(self, ctx_laplace, analyzer_laplace):
        """Linear f=q: E[x]=q, so bias of naive is zero."""
        bias = analyzer_laplace.bias(ctx_laplace.naive(q), q)
        assert sp.expand(bias) == 0


# ── EstimatorAnalyzer.mse ─────────────────────────────────────────────────────


class TestEstimatorAnalyzerMSE:
    def test_unbiased_mse_equals_variance(self, ctx_laplace, analyzer_laplace):
        """MSE(unbiased g) = Var(g) since bias=0."""
        f = q**2
        g = ctx_laplace.unbiased(f)
        assert sp.simplify(sp.expand(
            analyzer_laplace.mse(g, f) - analyzer_laplace.variance(g)
        )) == 0

    def test_mse_decomposition_naive_linear(self, ctx_laplace, analyzer_laplace):
        f = q
        h = ctx_laplace.naive(f)
        mse = analyzer_laplace.mse(h, f)
        var = analyzer_laplace.variance(h)
        bias = analyzer_laplace.bias(h, f)
        assert sp.simplify(sp.expand(mse - var - bias**2)) == 0

    @pytest.mark.parametrize("f", [
        q,
        q**2,
        q**3,
        q**2 + 3*q,
        2*q**2 - q,
    ])
    def test_mse_decomposition_parametrized(self, ctx_laplace, analyzer_laplace, f):
        h = ctx_laplace.naive(f)
        mse = analyzer_laplace.mse(h, f)
        var = analyzer_laplace.variance(h)
        bias = analyzer_laplace.bias(h, f)
        assert sp.simplify(sp.expand(mse - var - bias**2)) == 0

    def test_mse_unbiased_quadratic_gaussian(self, ctx_gaussian, analyzer_gaussian):
        """MSE(unbiased g for q²) = Var(g) for Gaussian noise."""
        f = q**2
        g = ctx_gaussian.unbiased(f)
        mse = analyzer_gaussian.mse(g, f)
        var = analyzer_gaussian.variance(g)
        assert sp.simplify(sp.expand(mse - var)) == 0


# ── ComparisonReport ──────────────────────────────────────────────────────────


REQUIRED_TOP_KEYS = {
    "polynomial", "naive", "unbiased",
    "mean_gap", "variance_gap", "variance_ratio", "variance_relative",
    "mse_gap", "mse_ratio", "mse_relative", "bias_naive_squared",
}
NAIVE_KEYS = {"estimator", "mean", "variance", "mse", "bias"}
UNBIASED_KEYS = {"estimator", "mean", "variance", "mse"}


class TestComparisonReport:
    def test_required_keys(self, sys_laplace):
        report = sys_laplace.compare(q**2)
        assert REQUIRED_TOP_KEYS.issubset(report.keys())
        assert NAIVE_KEYS.issubset(report["naive"].keys())
        assert UNBIASED_KEYS.issubset(report["unbiased"].keys())

    def test_unbiased_mean_equals_polynomial(self, sys_laplace):
        """E[g_unbiased] must exactly equal f(q)."""
        f = q**3 + 3*q**2
        report = sys_laplace.compare(f)
        assert sp.expand(report["unbiased"]["mean"] - f) == 0

    def test_mean_gap_quadratic(self, sys_laplace):
        """E[unbiased] - E[naive] = q² - (q²+2) = -2."""
        report = sys_laplace.compare(q**2)
        assert sp.expand(report["mean_gap"] + 2) == 0

    def test_bias_naive_quadratic(self, sys_laplace):
        """Naive x² has bias=2, so bias_naive_squared=4."""
        report = sys_laplace.compare(q**2)
        assert sp.expand(report["naive"]["bias"] - 2) == 0
        assert sp.expand(report["bias_naive_squared"] - 4) == 0

    def test_mse_gap_cubic_at_q0(self, sys_laplace):
        """At q=0: MSE gap for f=q³+3q² with s=1 should be -252.

        From the closed-form: MSE_gap = -4*(27q²+54q+63); at q=0 → -252.
        """
        f = q**3 + 3*q**2
        report = sys_laplace.compare(f)
        val = float(sp.simplify(report["mse_gap"].subs(q, 0)))
        assert val == pytest.approx(-252.0)

    def test_mse_gap_cubic_symbolic(self, sys_laplace):
        """MSE gap polynomial for f=q³+3q² equals -4*(27q²+54q+63)."""
        f = q**3 + 3*q**2
        report = sys_laplace.compare(f)
        expected = -4 * (27*q**2 + 54*q + 63)
        assert sp.expand(report["mse_gap"] - expected) == 0

    def test_variance_ratio_constant_is_oo(self, sys_laplace):
        """Constant target has Var(naive)=0, so ratio is defined as oo."""
        report = sys_laplace.compare(sp.Integer(5))
        assert report["variance_ratio"] == sp.oo

    def test_mse_gap_negative_means_unbiased_wins(self, sys_laplace):
        """For f=q³+3q² (K=36>0), mse_gap < 0 everywhere — unbiased wins."""
        f = q**3 + 3*q**2
        report = sys_laplace.compare(f)
        mse_gap = report["mse_gap"]
        for q_val in (-3, -2, -1, 0, 1, 2, 3):
            val = float(sp.simplify(mse_gap.subs(q, q_val)))
            assert val < 0, f"Expected mse_gap < 0 at q={q_val}, got {val}"


# ── EstimatorSystem facade ────────────────────────────────────────────────────


class TestEstimatorSystem:
    def test_estimator_naive(self, sys_laplace):
        result = sys_laplace.estimator(q**2, biasedness="naive")
        assert sp.expand(result - x**2) == 0

    def test_estimator_unbiased(self, sys_laplace):
        result = sys_laplace.estimator(q**2, biasedness="unbiased")
        assert sp.expand(result - (x**2 - 2)) == 0

    def test_estimator_invalid_raises(self, sys_laplace):
        with pytest.raises(ValueError):
            sys_laplace.estimator(q, biasedness="something_else")

    def test_compare_returns_dict(self, sys_laplace):
        report = sys_laplace.compare(q**2)
        assert isinstance(report, dict)

    def test_compare_more_returns_list(self, sys_laplace):
        reports = sys_laplace.compare_more([q, q**2])
        assert isinstance(reports, list)
        assert len(reports) == 2

    def test_compare_more_with_name_fn(self, sys_laplace):
        polys = [q, q**2]
        reports = sys_laplace.compare_more(polys, name_fn=lambda f, i: f"poly_{i}")
        assert reports[0]["name"] == "poly_0"
        assert reports[1]["name"] == "poly_1"


# ── Unbiasedness invariant (parametrized across models and polynomials) ────────


class TestUnbiasednessInvariantParametrized:
    @pytest.mark.parametrize("f", [
        q,
        q**2,
        q**3,
        q**4,
        q**2 + 3*q + 1,
        q**3 - q,
        2*q**3 + q**2 - 3*q,
    ])
    def test_laplace(self, sys_laplace, f):
        report = sys_laplace.compare(f)
        assert sp.expand(report["unbiased"]["mean"] - f) == 0

    @pytest.mark.parametrize("f", [
        q,
        q**2,
        q**3,
        q**2 - 2*q + 1,
    ])
    def test_gaussian(self, sys_gaussian, f):
        report = sys_gaussian.compare(f)
        assert sp.expand(report["unbiased"]["mean"] - f) == 0


# ── MSE decomposition: var + bias² == mse (naive, parametrized) ───────────────


class TestMSEDecompositionNaive:
    @pytest.mark.parametrize("f", [
        q,
        q**2,
        q**3,
        q**2 + q,
        3*q**2 - q + 2,
    ])
    def test_laplace(self, sys_laplace, f):
        report = sys_laplace.compare(f)
        mse = report["naive"]["mse"]
        var = report["naive"]["variance"]
        bias_sq = report["bias_naive_squared"]
        assert sp.simplify(sp.expand(mse - var - bias_sq)) == 0

    @pytest.mark.parametrize("f", [
        q,
        q**2,
        q**3,
    ])
    def test_gaussian(self, sys_gaussian, f):
        report = sys_gaussian.compare(f)
        mse = report["naive"]["mse"]
        var = report["naive"]["variance"]
        bias_sq = report["bias_naive_squared"]
        assert sp.simplify(sp.expand(mse - var - bias_sq)) == 0
