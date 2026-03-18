"""
Tests for qchem/integrals/boys.py

Run with: pytest test_boys.py -v
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, gamma, gammainc

from qchem.integrals.boys import boys, boys_array


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def boys_scipy(m, x):
    """Gold-standard reference via scipy's incomplete gamma function."""
    if x < 1e-10:
        return 1.0 / (2 * m + 1)
    a = m + 0.5
    return gamma(a) * gammainc(a, x) / (2.0 * x**a)


def boys_quadrature(m, x):
    """Brute-force numerical integration of the definition."""
    integrand = lambda t: t**(2 * m) * np.exp(-x * t**2)
    result, _ = quad(integrand, 0, 1, limit=200)
    return result


# ---------------------------------------------------------------------------
# Analytical values
# ---------------------------------------------------------------------------

class TestAnalytical:
    def test_at_zero(self):
        """F_m(0) = 1/(2m+1) exactly."""
        for m in range(10):
            assert abs(boys(m, 0.0) - 1.0 / (2 * m + 1)) < 1e-14, \
                f"Failed at m={m}"

    def test_f0_erf_identity(self):
        """F_0(x) = sqrt(pi) / (2*sqrt(x)) * erf(sqrt(x))."""
        for x in [0.1, 1.0, 5.0, 20.0, 100.0]:
            expected = np.sqrt(np.pi) / (2 * np.sqrt(x)) * erf(np.sqrt(x))
            assert abs(boys(0, x) - expected) < 1e-10, \
                f"Failed at x={x}"


# ---------------------------------------------------------------------------
# Cross-validation against scipy
# ---------------------------------------------------------------------------

class TestAgainstScipy:
    @pytest.mark.parametrize("m", range(0, 12))
    @pytest.mark.parametrize("x", [0.0, 1e-8, 1e-4, 0.1, 0.5, 1.0,
                                    5.0, 10.0, 20.0, 50.0, 100.0])
    def test_matches_scipy(self, m, x):
        ref = boys_scipy(m, x)
        got = boys(m, x)
        assert abs(got - ref) < 1e-10, \
            f"Mismatch at m={m}, x={x}: got {got:.15e}, expected {ref:.15e}"


# ---------------------------------------------------------------------------
# Numerical integration spot-checks
# ---------------------------------------------------------------------------

class TestAgainstQuadrature:
    @pytest.mark.parametrize("m,x", [
        (0, 0.0), (0, 1.0), (1, 0.5), (2, 0.01),
        (3, 2.0), (5, 10.0), (8, 0.1), (10, 5.0),
    ])
    def test_matches_quadrature(self, m, x):
        ref = boys_quadrature(m, x)
        got = boys(m, x)
        assert abs(got - ref) < 1e-9, \
            f"Quadrature mismatch at m={m}, x={x}"


# ---------------------------------------------------------------------------
# boys_array recurrence consistency
# ---------------------------------------------------------------------------

class TestBoysArray:
    @pytest.mark.parametrize("x", [0.01, 0.5, 2.0, 10.0, 50.0])
    def test_recurrence_self_consistent(self, x):
        """Every entry must satisfy the downward recurrence relation."""
        F = boys_array(m_max=10, x=x)
        exp_x = np.exp(-x)
        for m in range(9):
            reconstructed = (2 * x * F[m + 1] + exp_x) / (2 * m + 1)
            assert abs(F[m] - reconstructed) < 1e-12, \
                f"Recurrence broken at m={m}, x={x}"

    @pytest.mark.parametrize("x", [0.01, 1.0, 10.0, 50.0])
    def test_array_matches_scalar(self, x):
        """boys_array entries must agree with scalar boys() calls."""
        m_max = 8
        F = boys_array(m_max=m_max, x=x)
        for m in range(m_max + 1):
            assert abs(F[m] - boys(m, x)) < 1e-12, \
                f"Array/scalar mismatch at m={m}, x={x}"

    def test_length(self):
        """boys_array must return exactly m_max+1 elements."""
        for m_max in [0, 1, 5, 10]:
            assert len(boys_array(m_max=m_max, x=1.0)) == m_max + 1


# ---------------------------------------------------------------------------
# Crossover region (where small-x / large-x branches switch)
# ---------------------------------------------------------------------------

class TestCrossover:
    def test_no_discontinuity(self):
        """
        Dense sweep across crossover region. F_m(x) is smooth, so
        second differences (curvature) must stay tiny. A real discontinuity
        would produce a spike in the second differences.
        """
        xs = np.linspace(14.0, 26.0, 200)
        for m in range(6):
            vals = np.array([boys(m, x) for x in xs])
            second_diffs = np.abs(np.diff(vals, n=2))
            assert np.all(second_diffs < 1e-5), \
                f"Discontinuity near crossover for m={m}, " \
                f"max second diff={second_diffs.max():.2e}"

    def test_crossover_matches_scipy(self):
        """Both branches must agree with scipy reference near the boundary."""
        xs = np.linspace(14.0, 26.0, 50)
        for m in range(6):
            for x in xs:
                ref = boys_scipy(m, x)
                got = boys(m, x)
                assert abs(got - ref) / (abs(ref) + 1e-30) < 1e-10, \
                    f"Crossover failure at m={m}, x={x:.3f}"


# ---------------------------------------------------------------------------
# Qualitative / monotonicity properties
# ---------------------------------------------------------------------------

class TestQualitative:
    def test_positivity(self):
        """F_m(x) > 0 for all m >= 0, x >= 0."""
        for m in range(10):
            for x in np.linspace(0, 100, 30):
                assert boys(m, x) > 0, f"Non-positive at m={m}, x={x}"

    def test_decreasing_in_x(self):
        """F_m(x) is strictly decreasing in x for fixed m."""
        xs = np.linspace(0.01, 50.0, 100)
        for m in range(6):
            vals = [boys(m, x) for x in xs]
            assert np.all(np.diff(vals) < 0), \
                f"Not monotone decreasing in x for m={m}"

    def test_decreasing_in_m(self):
        """F_m(x) < F_{m-1}(x) for fixed x > 0."""
        for x in [0.5, 2.0, 10.0]:
            vals = [boys(m, x) for m in range(10)]
            assert np.all(np.diff(vals) < 0), \
                f"Not decreasing in m at x={x}"