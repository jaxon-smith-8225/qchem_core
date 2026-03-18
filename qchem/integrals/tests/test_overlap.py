"""
Tests for qchem/integrals/overlap.py

Run with: pytest test_overlap.py -v
"""

import numpy as np
import pytest
from itertools import product as iproduct
from scipy.special import factorial2

from qchem.integrals.overlap import (
    overlap_1d,
    overlap_primitive,
    overlap_contracted,
    build_overlap_matrix,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_shell(center, angular, exponents, coefficients):
    return {
        'center': np.array(center, dtype=float),
        'angular': angular,
        'exponents': list(exponents),
        'coefficients': list(coefficients),
    }


def s_shell(center, alpha, coeff=1.0):
    return make_shell(center, (0, 0, 0), [alpha], [coeff])


def p_shell(center, axis, alpha, coeff=1.0):
    """axis: 0=px, 1=py, 2=pz"""
    angular = [0, 0, 0]
    angular[axis] = 1
    return make_shell(center, tuple(angular), [alpha], [coeff])


def _double_factorial(n):
    """(-1)!! = 1 by convention."""
    if n <= 0:
        return 1.0
    return float(factorial2(n))


def norm_primitive(alpha, angular):
    """Normalization constant — mirrors overlap.py so tests stay consistent."""
    lx, ly, lz = angular
    L = lx + ly + lz
    prefactor = (2 * alpha / np.pi)**0.75 * (4 * alpha)**(L / 2)
    denom = np.sqrt(
        _double_factorial(2*lx - 1) *
        _double_factorial(2*ly - 1) *
        _double_factorial(2*lz - 1)
    )
    return prefactor / denom


def overlap_numerical(shell_a, shell_b, limit=5.0, n=40):
    """
    Brute-force 3D numerical integration including normalization constants,
    so it matches what overlap_contracted returns.
    """
    A = shell_a['center']
    B = shell_b['center']
    a_ang = shell_a['angular']
    b_ang = shell_b['angular']

    xs = np.linspace(-limit, limit, n)
    dx = xs[1] - xs[0]
    result = 0.0

    for alpha, ca in zip(shell_a['exponents'], shell_a['coefficients']):
        for beta, cb in zip(shell_b['exponents'], shell_b['coefficients']):
            Na = norm_primitive(alpha, a_ang)
            Nb = norm_primitive(beta, b_ang)
            val = 0.0
            for x in xs:
                for y in xs:
                    for z in xs:
                        r = np.array([x, y, z])
                        ra = r - A
                        rb = r - B
                        fa = (ra[0]**a_ang[0] * ra[1]**a_ang[1] * ra[2]**a_ang[2]
                              * np.exp(-alpha * np.dot(ra, ra)))
                        fb = (rb[0]**b_ang[0] * rb[1]**b_ang[1] * rb[2]**b_ang[2]
                              * np.exp(-beta * np.dot(rb, rb)))
                        val += fa * fb * dx**3
            result += Na * Nb * ca * cb * val
    return result


# ---------------------------------------------------------------------------
# 1D overlap tests
# ---------------------------------------------------------------------------

class TestOverlap1D:
    def test_s_s_same_center(self):
        """S_{0,0}(A=B) = sqrt(pi/p)."""
        alpha, beta = 1.0, 0.5
        p = alpha + beta
        expected = np.sqrt(np.pi / p)
        assert abs(overlap_1d(0, 0, 0.0, 0.0, alpha, beta) - expected) < 1e-14

    def test_s_s_offset(self):
        """S_{0,0} with offset: sqrt(pi/p) * exp(-alpha*beta/p * dX^2)."""
        alpha, beta = 1.0, 0.5
        Ax, Bx = 0.0, 1.0
        p = alpha + beta
        expected = np.sqrt(np.pi / p) * np.exp(-alpha * beta / p * (Ax - Bx)**2)
        assert abs(overlap_1d(0, 0, Ax, Bx, alpha, beta) - expected) < 1e-14

    def test_odd_angular_same_center_is_zero(self):
        """S_{1,0} with A=B=P is zero (odd integrand)."""
        val = overlap_1d(1, 0, 0.0, 0.0, 1.0, 1.0)
        assert abs(val) < 1e-15

    def test_symmetry(self):
        """S_{i,j}(A,B,alpha,beta) = S_{j,i}(B,A,beta,alpha)."""
        for i, j in [(0,1), (1,0), (1,1), (2,1), (0,2)]:
            a = overlap_1d(i, j, 0.0, 1.2, 0.8, 1.3)
            b = overlap_1d(j, i, 1.2, 0.0, 1.3, 0.8)
            assert abs(a - b) < 1e-14, f"Symmetry failed for i={i}, j={j}"

    def test_decays_with_distance(self):
        """1D overlap should decrease as centers move apart."""
        alpha, beta = 1.0, 1.0
        vals = [overlap_1d(0, 0, 0.0, R, alpha, beta) for R in [0, 0.5, 1.0, 2.0, 4.0]]
        assert all(vals[i] > vals[i+1] for i in range(len(vals)-1))


# ---------------------------------------------------------------------------
# Primitive overlap tests (no normalization)
# ---------------------------------------------------------------------------

class TestOverlapPrimitive:
    def test_ss_same_center(self):
        """Two s-functions at same center (unnormalized): (pi/p)^(3/2)."""
        alpha, beta = 1.0, 0.5
        A = np.zeros(3)
        expected = (np.pi / (alpha + beta))**1.5
        got = overlap_primitive((0,0,0), (0,0,0), alpha, beta, A, A)
        assert abs(got - expected) < 1e-13

    def test_symmetry(self):
        """<a|b> == <b|a> for primitives."""
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([1.0, 0.5, 0.2])
        pairs = [
            ((0,0,0), (0,0,0), 1.0, 0.8),
            ((1,0,0), (0,0,0), 1.2, 0.7),
            ((1,0,0), (1,0,0), 0.9, 1.1),
            ((1,1,0), (0,1,0), 1.0, 1.0),
        ]
        for a_ang, b_ang, alpha, beta in pairs:
            ab = overlap_primitive(a_ang, b_ang, alpha, beta, A, B)
            ba = overlap_primitive(b_ang, a_ang, beta, alpha, B, A)
            assert abs(ab - ba) < 1e-14, f"Symmetry failed for {a_ang},{b_ang}"

    def test_p_functions_odd_parity_zero(self):
        """px and py at same center have zero overlap (orthogonal)."""
        A = np.zeros(3)
        val = overlap_primitive((1,0,0), (0,1,0), 1.0, 1.0, A, A)
        assert abs(val) < 1e-15

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_s_p_same_center_zero(self, axis):
        """s and any p at the same center: zero overlap."""
        A = np.zeros(3)
        b_ang = [0, 0, 0]
        b_ang[axis] = 1
        val = overlap_primitive((0,0,0), tuple(b_ang), 1.0, 1.0, A, A)
        assert abs(val) < 1e-15


# ---------------------------------------------------------------------------
# Contracted overlap tests (includes normalization)
# ---------------------------------------------------------------------------

class TestOverlapContracted:
    def test_ss_same_center_analytical(self):
        """
        Single-primitive s-s at same center with normalization.
        Expected = N_alpha * N_beta * (pi/(a+b))^(3/2)
        where N = (2*alpha/pi)^(3/4) for s-functions.
        """
        alpha, beta = 1.3, 0.7
        Na = (2 * alpha / np.pi)**0.75
        Nb = (2 * beta  / np.pi)**0.75
        expected = Na * Nb * (np.pi / (alpha + beta))**1.5
        a = s_shell([0, 0, 0], alpha)
        b = s_shell([0, 0, 0], beta)
        assert abs(overlap_contracted(a, b) - expected) < 1e-13

    def test_normalized_self_overlap(self):
        """
        A normalized s-function <phi|phi> should equal 1.
        For a single primitive with coefficient 1, this holds only if the
        normalization is correctly applied: N^2 * (pi/2alpha)^(3/2) = 1.
        """
        for alpha in [0.3, 1.0, 2.5]:
            a = s_shell([0, 0, 0], alpha)
            val = overlap_contracted(a, a)
            assert abs(val - 1.0) < 1e-12, \
                f"Self-overlap != 1 for alpha={alpha}: got {val}"

    def test_symmetry(self):
        """overlap_contracted(a, b) == overlap_contracted(b, a)."""
        a = s_shell([0, 0, 0], 1.0)
        b = p_shell([1, 0, 0], 0, 0.7)
        assert abs(overlap_contracted(a, b) - overlap_contracted(b, a)) < 1e-14

    def test_decays_with_distance(self):
        """S(a,b) decreases as centers move apart."""
        alpha = 1.0
        vals = []
        for R in [0.0, 0.5, 1.0, 2.0, 4.0]:
            a = s_shell([0, 0, 0], alpha)
            b = s_shell([R,  0, 0], alpha)
            vals.append(overlap_contracted(a, b))
        assert all(vals[i] > vals[i+1] for i in range(len(vals)-1))

    def test_multi_primitive_contraction(self):
        """STO-3G-like contraction: result should be positive."""
        exps   = [3.4252509, 0.6239137, 0.1688554]
        coeffs = [0.1543290, 0.5353281, 0.4446345]
        a = make_shell([0,0,0], (0,0,0), exps, coeffs)
        b = make_shell([0,0,0], (0,0,0), exps, coeffs)
        val = overlap_contracted(a, b)
        assert val > 0

    def test_numerical_spot_check(self):
        """Compare normalized analytic vs normalized numerical integration."""
        alpha, beta = 1.5, 1.0
        a = s_shell([0, 0, 0], alpha)
        b = s_shell([0.8, 0, 0], beta)
        analytic  = overlap_contracted(a, b)
        numerical = overlap_numerical(a, b, limit=6.0, n=50)
        assert abs(analytic - numerical) / abs(analytic) < 1e-3


# ---------------------------------------------------------------------------
# Overlap matrix tests
# ---------------------------------------------------------------------------

class TestOverlapMatrix:
    @pytest.fixture
    def h2_sto3g_basis(self):
        """Minimal STO-3G basis for H2 at bond length 1.4 bohr."""
        exps   = [3.4252509, 0.6239137, 0.1688554]
        coeffs = [0.1543290, 0.5353281, 0.4446345]
        return [
            make_shell([0.0, 0.0,  0.0], (0,0,0), exps, coeffs),
            make_shell([0.0, 0.0,  1.4], (0,0,0), exps, coeffs),
        ]

    def test_diagonal_positive(self, h2_sto3g_basis):
        """Diagonal elements <phi|phi> must be positive."""
        S = build_overlap_matrix(h2_sto3g_basis)
        assert np.all(np.diag(S) > 0)

    def test_symmetric(self, h2_sto3g_basis):
        """S matrix must be symmetric."""
        S = build_overlap_matrix(h2_sto3g_basis)
        assert np.allclose(S, S.T, atol=1e-14)

    def test_positive_definite(self, h2_sto3g_basis):
        """All eigenvalues of S must be positive."""
        S = build_overlap_matrix(h2_sto3g_basis)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals.min()}"

    def test_off_diagonal_bounded(self, h2_sto3g_basis):
        """|S_ij| <= sqrt(S_ii * S_jj) by Cauchy-Schwarz."""
        S = build_overlap_matrix(h2_sto3g_basis)
        n = S.shape[0]
        for i, j in iproduct(range(n), repeat=2):
            if i != j:
                bound = np.sqrt(S[i, i] * S[j, j])
                assert abs(S[i, j]) <= bound + 1e-14, \
                    f"Cauchy-Schwarz violated at ({i},{j})"

    def test_h2_sto3g_known_value(self, h2_sto3g_basis):
        """
        STO-3G H2 at R=1.4 bohr: S_01 ~ 0.6593 (Szabo & Ostlund Table 3.1).
        """
        S = build_overlap_matrix(h2_sto3g_basis)
        assert abs(S[0, 1] - 0.6593) < 1e-3