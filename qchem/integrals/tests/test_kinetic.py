"""
Tests for qchem/integrals/kinetic.py

Run with: pytest test_kinetic.py -v

Test structure mirrors test_overlap.py:
  TestKinetic1D         — unit tests for kinetic_1d
  TestKineticPrimitive  — tests for kinetic_primitive (unnormalized)
  TestKineticContracted — tests for kinetic_contracted (with normalisation)
  TestKineticMatrix     — tests for build_kinetic_matrix

Reference implementations
--------------------------
All numerical references use scipy.integrate.quad on the 1D integrals,
combined via the same 3D separability formula the implementation uses.
This is independent of the Obara–Saika recursion and exercises the
low-shift guard (j < 2), the j+2 upper shift, and the j-2 lower shift
all in isolation before they appear together in the 3D integrals.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import factorial2
from itertools import product as iproduct

from qchem.integrals.kinetic import (
    kinetic_1d,
    kinetic_primitive,
    kinetic_contracted,
    build_kinetic_matrix,
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
    ang = [0, 0, 0]
    ang[axis] = 1
    return make_shell(center, tuple(ang), [alpha], [coeff])


def _double_factorial(n):
    """(-1)!! = 1 by convention."""
    if n <= 0:
        return 1.0
    return float(factorial2(n))


def norm_primitive(alpha, angular):
    """Normalization constant — mirrors kinetic.py so reference stays consistent."""
    lx, ly, lz = angular
    L = lx + ly + lz
    prefactor = (2 * alpha / np.pi)**0.75 * (4 * alpha)**(L / 2)
    denom = np.sqrt(
        _double_factorial(2 * lx - 1) *
        _double_factorial(2 * ly - 1) *
        _double_factorial(2 * lz - 1)
    )
    return prefactor / denom


# ---------------------------------------------------------------------------
# Reference implementations (scipy.integrate.quad)
# ---------------------------------------------------------------------------

def _overlap_1d_quad(i, j, Ax, Bx, alpha, beta):
    """
    1D overlap integral via scipy.integrate.quad.
    Used as part of the 3D kinetic reference.
    """
    def integrand(x):
        return ((x - Ax)**i * np.exp(-alpha * (x - Ax)**2) *
                (x - Bx)**j * np.exp(-beta  * (x - Bx)**2))
    result, _ = quad(integrand, -np.inf, np.inf, limit=200)
    return result


def _kinetic_1d_quad(i, j, Ax, Bx, alpha, beta):
    """
    1D kinetic integral via scipy.integrate.quad.

    Computes  -1/2 * integral g_i(x) * d²/dx² g_j(x) dx

    where  d²/dx² g_j = [j(j-1)(x-Bx)^{j-2} - 2β(2j+1)(x-Bx)^j
                         + 4β²(x-Bx)^{j+2}] * exp(-β(x-Bx)²)

    This is the same identity used by the Obara–Saika formula but
    evaluated by direct quadrature, providing an independent reference.
    """
    def integrand(x):
        gi  = (x - Ax)**i * np.exp(-alpha * (x - Ax)**2)
        rb  = x - Bx
        exp = np.exp(-beta * rb**2)
        d2gj  = (j * (j - 1) * rb**(j - 2) * exp) if j >= 2 else 0.0
        d2gj -= 2 * beta * (2 * j + 1) * rb**j * exp
        d2gj += 4 * beta**2 * rb**(j + 2) * exp
        return gi * d2gj
    result, _ = quad(integrand, -np.inf, np.inf, limit=200)
    return -0.5 * result


def kinetic_contracted_numerical(shell_a, shell_b):
    """
    Reference contracted kinetic integral using separable 1D quadrature.

    Uses the 3D identity  T = Tx·Sy·Sz + Sx·Ty·Sz + Sx·Sy·Tz
    with each 1D integral evaluated by scipy, providing an independent
    reference that does not rely on the Obara–Saika recursion at all.
    """
    result = 0.0
    A, a = shell_a['center'], shell_a['angular']
    B, b = shell_b['center'], shell_b['angular']

    for alpha, ca in zip(shell_a['exponents'], shell_a['coefficients']):
        for beta, cb in zip(shell_b['exponents'], shell_b['coefficients']):
            Na = norm_primitive(alpha, a)
            Nb = norm_primitive(beta,  b)

            Sx = _overlap_1d_quad(a[0], b[0], A[0], B[0], alpha, beta)
            Sy = _overlap_1d_quad(a[1], b[1], A[1], B[1], alpha, beta)
            Sz = _overlap_1d_quad(a[2], b[2], A[2], B[2], alpha, beta)
            Tx = _kinetic_1d_quad(a[0], b[0], A[0], B[0], alpha, beta)
            Ty = _kinetic_1d_quad(a[1], b[1], A[1], B[1], alpha, beta)
            Tz = _kinetic_1d_quad(a[2], b[2], A[2], B[2], alpha, beta)

            T_prim = Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz
            result += Na * Nb * ca * cb * T_prim

    return result


# ---------------------------------------------------------------------------
# 1D kinetic tests
# ---------------------------------------------------------------------------

class TestKinetic1D:
    def test_ss_same_center_analytical(self):
        """
        T_1d(0, 0) with A=B has the closed form  alpha*beta/p * sqrt(pi/p).

        Derivation: apply the Obara–Saika formula directly.
          T(0,0) = beta*(2*0+1)*S(0,0) - 2*beta²*S(0,2)
        with S(0,0)=sqrt(pi/p) and S(0,2)=sqrt(pi/p)/(2p), giving
          T(0,0) = beta*sqrt(pi/p) - beta²/p * sqrt(pi/p)
                 = sqrt(pi/p) * beta*(1 - beta/p)
                 = sqrt(pi/p) * alpha*beta/p
        """
        for alpha, beta in [(1.0, 0.5), (0.3, 2.0), (1.5, 1.5)]:
            p = alpha + beta
            expected = alpha * beta / p * np.sqrt(np.pi / p)
            got = kinetic_1d(0, 0, 0.0, 0.0, alpha, beta)
            assert abs(got - expected) < 1e-13, \
                f"T_1d(0,0) failed for alpha={alpha}, beta={beta}: " \
                f"got {got:.15e}, expected {expected:.15e}"

    @pytest.mark.parametrize("i,j", [
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (2, 2),
    ])
    def test_symmetry(self, i, j):
        """
        T_1d(i, j, Ax, Bx, alpha, beta) = T_1d(j, i, Bx, Ax, beta, alpha).

        The kinetic energy matrix is Hermitian (real symmetric), so swapping
        bra and ket — which also swaps the centres and exponents — must give
        the same value.
        """
        Ax, Bx, alpha, beta = 0.0, 1.2, 0.8, 1.3
        fwd = kinetic_1d(i, j, Ax, Bx, alpha, beta)
        rev = kinetic_1d(j, i, Bx, Ax, beta, alpha)
        assert abs(fwd - rev) < 1e-13, \
            f"Symmetry failed for i={i}, j={j}: fwd={fwd:.15e}, rev={rev:.15e}"

    @pytest.mark.parametrize("i,j,Ax,Bx", [
        (0, 0, 0.0,  0.0),
        (0, 0, 0.0,  1.0),
        (1, 0, 0.0,  0.0),
        (1, 0, 0.0,  0.8),
        (0, 1, 0.0,  0.5),
        (1, 1, 0.0,  0.0),
        (1, 1, 0.3,  1.1),
        (0, 2, 0.0,  0.5),  # exercises j=2 lower-shift term
        (2, 1, 0.0,  1.0),
        (2, 2, 0.0,  0.6),  # both i and j reach the j>=2 branch
    ])
    def test_against_quadrature(self, i, j, Ax, Bx):
        """
        kinetic_1d must agree with the scipy.integrate.quad reference to 1e-10.
        Covers both same-centre and offset cases, and exercises the j=0/1/2+
        branches of the lower-shift guard.
        """
        alpha, beta = 1.0, 0.7
        ref = _kinetic_1d_quad(i, j, Ax, Bx, alpha, beta)
        got = kinetic_1d(i, j, Ax, Bx, alpha, beta)
        assert abs(got - ref) < 1e-10, \
            f"Quadrature mismatch i={i}, j={j}, Ax={Ax}, Bx={Bx}: " \
            f"got {got:.15e}, ref {ref:.15e}"

    def test_low_shift_guard_j0(self):
        """
        j=0: the j*(j-1) lower-shift term must be absent.
        No j=-2 angular momentum is requested; result must be finite and positive.
        """
        val = kinetic_1d(0, 0, 0.0, 0.0, 1.0, 1.0)
        assert np.isfinite(val) and val > 0, \
            f"j=0 guard failed: got {val}"

    def test_low_shift_guard_j1(self):
        """
        j=1: j*(j-1)=0 so the lower-shift coefficient is zero.
        No j=-1 angular momentum is requested; result must be finite.
        """
        val = kinetic_1d(0, 1, 0.0, 0.0, 1.0, 1.0)
        assert np.isfinite(val), f"j=1 guard failed: got {val}"

    def test_sp_same_center_zero(self):
        """
        T_1d(0, 1) with A=B is zero: the integrand is an odd function
        of (x-A) so the integral over all space vanishes.
        """
        val = kinetic_1d(0, 1, 0.0, 0.0, 1.0, 1.0)
        assert abs(val) < 1e-15, f"T_1d(0,1) same centre not zero: {val}"

    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_diagonal_positive(self, j):
        """
        T_1d(j, j) at the same centre must be strictly positive: it
        represents the 1D contribution to kinetic energy from an electron
        in a pure Gaussian and must always be positive.
        """
        val = kinetic_1d(j, j, 0.0, 0.0, 1.0, 1.0)
        assert val > 0, f"T_1d({j},{j}) not positive: {val}"


# ---------------------------------------------------------------------------
# Primitive kinetic tests (unnormalised)
# ---------------------------------------------------------------------------

class TestKineticPrimitive:
    def test_ss_same_center_analytical(self):
        """
        For two s-functions at the same centre (unnormalised):

          T_prim = Tx·Sy·Sz + Sx·Ty·Sz + Sx·Sy·Tz
                 = 3 * (alpha*beta/p) * (pi/p)^(3/2)

        Each dimension contributes identically: T_1d = alpha*beta/p * sqrt(pi/p),
        S_1d = sqrt(pi/p).  Three equal terms sum to the formula above.
        """
        for alpha, beta in [(1.0, 0.5), (0.3, 2.0), (1.5, 1.5)]:
            p = alpha + beta
            expected = 3.0 * alpha * beta / p * (np.pi / p)**1.5
            A = np.zeros(3)
            got = kinetic_primitive((0,0,0), (0,0,0), alpha, beta, A, A)
            assert abs(got - expected) < 1e-13, \
                f"T_prim(ss) failed for alpha={alpha}, beta={beta}: " \
                f"got {got:.15e}, expected {expected:.15e}"

    def test_symmetry(self):
        """<a|T|b> == <b|T|a> for primitives."""
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([1.0, 0.5, 0.2])
        cases = [
            ((0,0,0), (0,0,0), 1.0, 0.8),
            ((1,0,0), (0,0,0), 1.2, 0.7),
            ((1,0,0), (1,0,0), 0.9, 1.1),
            ((1,1,0), (0,1,0), 1.0, 1.0),
            ((2,0,0), (0,0,0), 1.0, 1.0),   # d-type: exercises j=2 branch
        ]
        for a_ang, b_ang, alpha, beta in cases:
            ab = kinetic_primitive(a_ang, b_ang, alpha, beta, A, B)
            ba = kinetic_primitive(b_ang, a_ang, beta, alpha, B, A)
            assert abs(ab - ba) < 1e-13, \
                f"Symmetry failed for {a_ang},{b_ang}: ab={ab:.15e}, ba={ba:.15e}"

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_s_p_same_center_zero(self, axis):
        """
        <s|T|p_k> at the same centre is zero for all three Cartesian p-functions.
        Each 1D factor that contains the p angular momentum is an odd integrand;
        its S or T integral vanishes and takes the whole product with it.
        """
        A = np.zeros(3)
        b_ang = [0, 0, 0]
        b_ang[axis] = 1
        val = kinetic_primitive((0,0,0), tuple(b_ang), 1.0, 1.0, A, A)
        assert abs(val) < 1e-15, \
            f"T_prim(s, p_{axis}) same centre not zero: {val}"

    def test_diagonal_positive(self):
        """
        <phi|T|phi> for a single primitive at the same centre must be positive.
        Tests s, px, py, pz to confirm positivity for L=0 and L=1.
        """
        A = np.zeros(3)
        for angular in [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]:
            val = kinetic_primitive(angular, angular, 1.0, 1.0, A, A)
            assert val > 0, f"T_prim diagonal not positive for {angular}: {val}"


# ---------------------------------------------------------------------------
# Contracted kinetic tests (includes normalisation)
# ---------------------------------------------------------------------------

class TestKineticContracted:
    def test_ss_same_center_analytical(self):
        """
        Single-primitive normalised s-s at the same centre has the closed form:

          T = Na * Nb * 3 * alpha*beta/p * (pi/p)^(3/2)

        where Na = (2*alpha/pi)^(3/4), Nb = (2*beta/pi)^(3/4).
        """
        for alpha, beta in [(1.3, 0.7), (0.5, 2.0), (1.0, 1.0)]:
            p = alpha + beta
            Na = (2 * alpha / np.pi)**0.75
            Nb = (2 * beta  / np.pi)**0.75
            expected = Na * Nb * 3.0 * alpha * beta / p * (np.pi / p)**1.5
            a = s_shell([0, 0, 0], alpha)
            b = s_shell([0, 0, 0], beta)
            got = kinetic_contracted(a, b)
            assert abs(got - expected) < 1e-12, \
                f"T_contracted(ss) failed alpha={alpha}, beta={beta}: " \
                f"got {got:.15e}, expected {expected:.15e}"

    def test_diagonal_positive(self):
        """
        <phi|T|phi> for a normalised contracted basis function must be positive,
        regardless of the angular momentum or exponent.
        """
        for alpha in [0.3, 1.0, 2.5]:
            for angular in [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]:
                center = [0.0, 0.0, 0.0]
                shell = make_shell(center, angular, [alpha], [1.0])
                val = kinetic_contracted(shell, shell)
                assert val > 0, \
                    f"Diagonal not positive for alpha={alpha}, ang={angular}: {val}"

    def test_symmetry(self):
        """kinetic_contracted(a, b) == kinetic_contracted(b, a)."""
        a = s_shell([0.0, 0.0, 0.0], 1.0)
        b = p_shell([1.0, 0.0, 0.0], 0, 0.7)
        fwd = kinetic_contracted(a, b)
        rev = kinetic_contracted(b, a)
        assert abs(fwd - rev) < 1e-14, \
            f"T_contracted not symmetric: fwd={fwd:.15e}, rev={rev:.15e}"

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_s_p_same_center_zero(self, axis):
        """<s|T|p_k> at the same centre is zero after normalisation."""
        a = s_shell([0, 0, 0], 1.0)
        b = p_shell([0, 0, 0], axis, 1.0)
        val = kinetic_contracted(a, b)
        assert abs(val) < 1e-14, \
            f"T_contracted(s, p_{axis}) same centre not zero: {val}"

    def test_numerical_spot_check_ss(self):
        """
        Normalised s-s with offset: analytic vs quad reference.
        Uses kinetic_contracted_numerical, which integrates each 1D factor
        independently via scipy — completely bypassing Obara–Saika.
        """
        alpha, beta = 1.5, 1.0
        a = s_shell([0.0, 0.0, 0.0], alpha)
        b = s_shell([0.8, 0.0, 0.0], beta)
        analytic  = kinetic_contracted(a, b)
        numerical = kinetic_contracted_numerical(a, b)
        assert abs(analytic - numerical) / (abs(analytic) + 1e-30) < 1e-8, \
            f"ss spot-check failed: analytic={analytic:.12e}, numerical={numerical:.12e}"

    def test_numerical_spot_check_sp(self):
        """
        Normalised s-px with offset: analytic vs quad reference.
        Exercises the mixed angular-momentum path through the 3D decomposition.
        """
        alpha, beta = 1.2, 0.9
        a = s_shell([0.0, 0.0, 0.0], alpha)
        b = p_shell([0.7, 0.0, 0.0], 0, beta)   # px on centre B
        analytic  = kinetic_contracted(a, b)
        numerical = kinetic_contracted_numerical(a, b)
        assert abs(analytic - numerical) / (abs(numerical) + 1e-30) < 1e-8, \
            f"sp spot-check failed: analytic={analytic:.12e}, numerical={numerical:.12e}"

    def test_numerical_spot_check_pp(self):
        """
        Normalised px-px with offset: analytic vs quad reference.
        Exercises the j=2 branch of the 1D kinetic recursion via the
        upper-shift term S(i, j+2) = S(1, 3).
        """
        alpha, beta = 1.0, 0.8
        a = p_shell([0.0, 0.0, 0.0], 0, alpha)
        b = p_shell([0.6, 0.0, 0.0], 0, beta)
        analytic  = kinetic_contracted(a, b)
        numerical = kinetic_contracted_numerical(a, b)
        assert abs(analytic - numerical) / (abs(numerical) + 1e-30) < 1e-8, \
            f"pp spot-check failed: analytic={analytic:.12e}, numerical={numerical:.12e}"

    def test_sto3g_hydrogen_diagonal(self):
        """
        STO-3G hydrogen kinetic energy: T_00 = 0.7600 hartree.
        Reference: Szabo & Ostlund, Table 3.21.
        """
        exps   = [3.4252509, 0.6239137, 0.1688554]
        coeffs = [0.1543290, 0.5353281, 0.4446345]
        h = make_shell([0.0, 0.0, 0.0], (0,0,0), exps, coeffs)
        val = kinetic_contracted(h, h)
        assert abs(val - 0.7600) < 1e-3, \
            f"STO-3G H diagonal T: got {val:.6f}, expected 0.7600"

    def test_multi_primitive_contraction_positive(self):
        """STO-3G-like multi-primitive contraction: self-integral must be positive."""
        exps   = [3.4252509, 0.6239137, 0.1688554]
        coeffs = [0.1543290, 0.5353281, 0.4446345]
        a = make_shell([0,0,0], (0,0,0), exps, coeffs)
        assert kinetic_contracted(a, a) > 0


# ---------------------------------------------------------------------------
# Kinetic matrix tests
# ---------------------------------------------------------------------------

class TestKineticMatrix:
    @pytest.fixture
    def h2_sto3g_basis(self):
        """Minimal STO-3G basis for H2 at bond length 1.4 bohr."""
        exps   = [3.4252509, 0.6239137, 0.1688554]
        coeffs = [0.1543290, 0.5353281, 0.4446345]
        return [
            make_shell([0.0, 0.0,  0.0], (0,0,0), exps, coeffs),
            make_shell([0.0, 0.0,  1.4], (0,0,0), exps, coeffs),
        ]

    def test_symmetric(self, h2_sto3g_basis):
        """T matrix must be symmetric."""
        T = build_kinetic_matrix(h2_sto3g_basis)
        assert np.allclose(T, T.T, atol=1e-14), \
            f"T matrix not symmetric, max deviation: {np.max(np.abs(T - T.T)):.2e}"

    def test_positive_definite(self, h2_sto3g_basis):
        """
        All eigenvalues of T must be strictly positive.
        The kinetic energy operator is positive definite on any finite Gaussian
        basis — every non-zero coefficient vector gives a positive expectation
        value <psi|T|psi> > 0.
        """
        T = build_kinetic_matrix(h2_sto3g_basis)
        eigvals = np.linalg.eigvalsh(T)
        assert np.all(eigvals > 0), \
            f"T matrix not positive definite, min eigenvalue: {eigvals.min():.6e}"

    def test_diagonal_positive(self, h2_sto3g_basis):
        """Diagonal elements T_ii = <phi_i|T|phi_i> must be positive."""
        T = build_kinetic_matrix(h2_sto3g_basis)
        assert np.all(np.diag(T) > 0), \
            f"Non-positive diagonal: {np.diag(T)}"

    def test_h2_sto3g_diagonal(self, h2_sto3g_basis):
        """
        STO-3G H2 at R=1.4 bohr: T_00 = T_11 = 0.7600 hartree.
        Reference: Szabo & Ostlund, Table 3.21.
        """
        T = build_kinetic_matrix(h2_sto3g_basis)
        assert abs(T[0, 0] - 0.7600) < 1e-3, \
            f"T_00 = {T[0, 0]:.6f}, expected 0.7600"
        assert abs(T[1, 1] - 0.7600) < 1e-3, \
            f"T_11 = {T[1, 1]:.6f}, expected 0.7600"

    def test_h2_sto3g_off_diagonal(self, h2_sto3g_basis):
        """
        STO-3G H2 at R=1.4 bohr: T_01 = T_10 = 0.2365 hartree.
        Reference: Szabo & Ostlund, Table 3.21.
        """
        T = build_kinetic_matrix(h2_sto3g_basis)
        assert abs(T[0, 1] - 0.2365) < 1e-3, \
            f"T_01 = {T[0, 1]:.6f}, expected 0.2365"

    def test_diagonal_equal_by_symmetry(self, h2_sto3g_basis):
        """
        Because both H atoms carry the same STO-3G basis, the two diagonal
        elements must be identical to machine precision.
        """
        T = build_kinetic_matrix(h2_sto3g_basis)
        assert abs(T[0, 0] - T[1, 1]) < 1e-14, \
            f"Diagonal elements differ: T_00={T[0,0]:.15e}, T_11={T[1,1]:.15e}"

    def test_off_diagonal_bounded_by_diagonal(self, h2_sto3g_basis):
        """
        |T_ij| < sqrt(T_ii * T_jj) by the Cauchy–Schwarz inequality applied
        to the positive-definite kinetic energy inner product.
        """
        T = build_kinetic_matrix(h2_sto3g_basis)
        n = T.shape[0]
        for i, j in iproduct(range(n), repeat=2):
            if i != j:
                bound = np.sqrt(T[i, i] * T[j, j])
                assert abs(T[i, j]) <= bound + 1e-14, \
                    f"Cauchy–Schwarz violated at ({i},{j}): " \
                    f"|T_ij|={abs(T[i,j]):.6e} > bound={bound:.6e}"
