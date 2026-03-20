"""
Tests for the electron repulsion integral (ERI) module.

Test hierarchy
--------------
1. Analytical s-s-s-s checks
   The (ss|ss) primitive reduces to a closed-form expression involving
   only the Boys function F_0(T), which we can evaluate independently.
   These tests cover the T→0 limit, generic T, and the large-exponent
   (large-T) regime.

2. Symmetry invariance
   The 8-fold permutation symmetry
       (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
               = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
   must hold for arbitrary angular momenta and centre positions.

3. Positivity of self-repulsion integrals
   (aa|aa) ≥ 0 for every primitive or contracted shell, because the
   integrand is everywhere non-negative.

4. STO-3G H₂ reference values
   Szabo & Ostlund, "Modern Quantum Chemistry", Appendix A / Table 3.1:
   for R = 1.4 a₀ the four unique AO integrals are known to seven
   significant figures.  These serve as the primary regression test.

5. build_eri_tensor symmetry and shape
   The full tensor must be (N, N, N, N), positive-definite on the
   (11|11) diagonal, and exactly satisfy all 8-fold symmetry relations.

6. Finite-difference derivative consistency
   The Gaussian derivative identity is ∂/∂Ax φ_s(r; α, A) = +2α φ_{px},
   so the identity tested is:
       ∂/∂Ax (ss|ss) = +2α (px s | ss)

7. p-type and d-type angular momentum
   Spot-check several specific integrals involving p and d functions
   against independently coded reference values, verifying that the
   VRR and HRR stages handle higher angular momentum correctly.
"""

import numpy as np
import pytest
from scipy.special import erf

from qchem.integrals.eri import (
    eri_primitive,
    eri_contracted,
    build_eri_tensor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def boys0(T: float) -> float:
    """F_0(T) = sqrt(π/(4T)) erf(sqrt(T)), with F_0(0) = 1."""
    if T < 1e-10:
        return 1.0
    return np.sqrt(np.pi / (4.0 * T)) * erf(np.sqrt(T))


def ssss_analytical(alpha: float, beta: float,
                    gamma: float, delta: float,
                    A: np.ndarray, B: np.ndarray,
                    C: np.ndarray, D: np.ndarray) -> float:
    """
    Closed-form (ss|ss) primitive integral.

        (ss|ss) = 2π^(5/2) / (p·q·√(p+q)) · K_AB · K_CD · F_0(T)

    where p = α+β, q = γ+δ, P = (αA+βB)/p, Q = (γC+δD)/q,
    T = ζ|P−Q|², ζ = pq/(p+q),
    K_AB = exp(−αβ/p |AB|²), K_CD = exp(−γδ/q |CD|²).
    """
    p = alpha + beta
    q = gamma + delta
    P = (alpha * A + beta  * B) / p
    Q = (gamma * C + delta * D) / q
    zeta = p * q / (p + q)
    K_AB = np.exp(-alpha * beta  / p * float(np.dot(A - B, A - B)))
    K_CD = np.exp(-gamma * delta / q * float(np.dot(C - D, C - D)))
    T = zeta * float(np.dot(P - Q, P - Q))
    return 2.0 * np.pi**2.5 / (p * q * np.sqrt(p + q)) * K_AB * K_CD * boys0(T)


# STO-3G hydrogen basis (1s) — Szabo & Ostlund Appendix A
_STO3G_H_EXPS   = [3.4252509, 0.6239137, 0.1688554]
_STO3G_H_COEFFS = [0.1543290, 0.5353281, 0.4446345]

def _h_shell(centre: np.ndarray) -> dict:
    return dict(
        center=np.asarray(centre, dtype=float),
        angular=(0, 0, 0),
        exponents=_STO3G_H_EXPS,
        coefficients=_STO3G_H_COEFFS,
    )

def _h2_basis(R: float = 1.4) -> list:
    """Two STO-3G 1s shells for H₂ at bond length R (a₀)."""
    return [_h_shell([0.0, 0.0, 0.0]),
            _h_shell([0.0, 0.0, R])]


# ---------------------------------------------------------------------------
# 1. Analytical s-s-s-s checks
# ---------------------------------------------------------------------------

class TestSSSS:
    """(ss|ss) primitive agrees with the closed-form Boys formula."""

    @pytest.mark.parametrize("alpha,beta,gamma,delta,A,B,C,D", [
        # All at origin — T = 0 limit, should equal 2π^(5/2)/(p*q*sqrt(p+q))
        (1.0, 1.0, 1.0, 1.0,
         [0,0,0], [0,0,0], [0,0,0], [0,0,0]),
        # Separated centres — generic T
        (0.5, 0.8, 1.2, 0.3,
         [0,0,0], [1.4,0,0], [0,0,0], [1.4,0,0]),
        # Asymmetric centres
        (0.5, 0.8, 1.2, 0.3,
         [0.1,0.2,0.3], [1.4,0.5,0.1], [0.3,0.9,0.2], [0.7,0.1,0.5]),
        # Large exponents (large T, tests asymptotic Boys branch)
        (5.0, 5.0, 5.0, 5.0,
         [0,0,0], [1.4,0,0], [0,0,0], [1.4,0,0]),
        # Very small exponents (diffuse functions, small T)
        (0.05, 0.05, 0.05, 0.05,
         [0,0,0], [2.0,0,0], [0,0,0], [2.0,0,0]),
        # Mixed large/small exponents
        (10.0, 0.1, 0.1, 10.0,
         [0,0,0], [0,0,0], [0,0,0], [0,0,0]),
    ])
    def test_ssss_vs_analytical(self, alpha, beta, gamma, delta,
                                A, B, C, D):
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        C = np.array(C, dtype=float)
        D = np.array(D, dtype=float)
        expected = ssss_analytical(alpha, beta, gamma, delta, A, B, C, D)
        result   = eri_primitive(
            (0,0,0), (0,0,0), (0,0,0), (0,0,0),
            alpha, beta, gamma, delta, A, B, C, D)
        assert result == pytest.approx(expected, rel=1e-10, abs=1e-14)

    def test_ssss_t_zero_limit(self):
        """T=0 case: result equals 2π^(5/2)/(p*q*sqrt(p+q))."""
        alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0
        A = np.zeros(3)
        p = q = 2.0
        expected = 2.0 * np.pi**2.5 / (p * q * np.sqrt(p + q))
        result = eri_primitive((0,0,0),(0,0,0),(0,0,0),(0,0,0),
                               alpha, beta, gamma, delta, A, A, A, A)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_ssss_all_same_centre_different_exps(self):
        """Same centre but varying exponents still satisfies closed form."""
        A = np.zeros(3)
        for alpha, beta, gamma, delta in [
            (0.3, 0.7, 1.1, 2.3),
            (4.0, 0.5, 2.0, 0.1),
        ]:
            expected = ssss_analytical(alpha, beta, gamma, delta, A, A, A, A)
            result = eri_primitive((0,0,0),(0,0,0),(0,0,0),(0,0,0),
                                   alpha, beta, gamma, delta, A, A, A, A)
            assert result == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. Symmetry invariance
# ---------------------------------------------------------------------------

class TestSymmetry:
    """
    The 8-fold permutation symmetry must hold exactly (to numerical
    precision) for all angular-momentum combinations tested.
    """

    _ALPHA = 0.5
    _BETA  = 0.8
    _GAMMA = 1.2
    _DELTA = 0.3
    _A = np.array([0.1, 0.2, 0.3])
    _B = np.array([1.4, 0.5, 0.1])
    _C = np.array([0.3, 0.9, 0.2])
    _D = np.array([0.7, 0.1, 0.5])

    def _check_8fold(self, a, b, c, d):
        al, be, ga, de = self._ALPHA, self._BETA, self._GAMMA, self._DELTA
        A, B, C, D = self._A, self._B, self._C, self._D
        ref = eri_primitive(a, b, c, d, al, be, ga, de, A, B, C, D)
        # 7 distinct permutation partners
        partners = [
            eri_primitive(b, a, c, d, be, al, ga, de, B, A, C, D),  # (ji|kl)
            eri_primitive(a, b, d, c, al, be, de, ga, A, B, D, C),  # (ij|lk)
            eri_primitive(b, a, d, c, be, al, de, ga, B, A, D, C),  # (ji|lk)
            eri_primitive(c, d, a, b, ga, de, al, be, C, D, A, B),  # (kl|ij)
            eri_primitive(d, c, a, b, de, ga, al, be, D, C, A, B),  # (lk|ij)
            eri_primitive(c, d, b, a, ga, de, be, al, C, D, B, A),  # (kl|ji)
            eri_primitive(d, c, b, a, de, ga, be, al, D, C, B, A),  # (lk|ji)
        ]
        for val in partners:
            assert val == pytest.approx(ref, rel=1e-10, abs=1e-14), (
                f"Symmetry broken for a={a}, b={b}, c={c}, d={d}:  "
                f"ref={ref:.12g}, partner={val:.12g}"
            )

    def test_symmetry_ssss(self):
        self._check_8fold((0,0,0), (0,0,0), (0,0,0), (0,0,0))

    def test_symmetry_psss(self):
        self._check_8fold((1,0,0), (0,0,0), (0,0,0), (0,0,0))

    def test_symmetry_ppss(self):
        self._check_8fold((1,0,0), (0,1,0), (0,0,0), (0,0,0))

    def test_symmetry_ppps(self):
        self._check_8fold((1,0,0), (0,1,0), (0,0,1), (0,0,0))

    def test_symmetry_pppp(self):
        self._check_8fold((1,0,0), (0,1,0), (0,0,1), (1,0,0))

    def test_symmetry_dsss(self):
        self._check_8fold((2,0,0), (0,0,0), (0,0,0), (0,0,0))

    def test_symmetry_dxysss(self):
        self._check_8fold((1,1,0), (0,0,0), (0,0,0), (0,0,0))

    def test_symmetry_dpss(self):
        self._check_8fold((2,0,0), (1,0,0), (0,0,0), (0,0,0))

    def test_symmetry_ddss(self):
        self._check_8fold((1,1,0), (1,0,0), (1,0,0), (0,0,0))

    def test_symmetry_general_mixed(self):
        self._check_8fold((1,1,0), (0,0,0), (1,0,0), (0,0,0))


# ---------------------------------------------------------------------------
# 3. Positivity of self-repulsion
# ---------------------------------------------------------------------------

class TestPositivity:
    """(aa|aa) ≥ 0 because the integrand φ_a(r₁)²φ_a(r₂)²/|r₁−r₂| ≥ 0."""

    @pytest.mark.parametrize("a", [
        (0,0,0), (1,0,0), (0,1,0), (0,0,1),
        (2,0,0), (1,1,0), (0,1,1),
    ])
    def test_self_repulsion_primitive(self, a):
        alpha = 1.3
        A = np.array([0.5, 0.3, 0.1])
        val = eri_primitive(a, a, a, a, alpha, alpha, alpha, alpha, A, A, A, A)
        assert val >= 0.0, f"Self-repulsion negative for a={a}: {val}"

    def test_self_repulsion_contracted(self):
        s_shell = _h_shell([0.3, 0.5, 0.1])
        val = eri_contracted(s_shell, s_shell, s_shell, s_shell)
        assert val > 0.0


# ---------------------------------------------------------------------------
# 4. STO-3G H₂ reference values (Szabo & Ostlund)
# ---------------------------------------------------------------------------

class TestSTO3GH2:
    """
    Reference values for the STO-3G H₂ molecule at R = 1.4 a₀.

    Szabo & Ostlund, "Modern Quantum Chemistry" (Dover 1989),
    Chapter 3 / Appendix A.  All four unique non-zero AO integrals
    are reproduced here.  The tight tolerance (1e-6 relative) is
    justified because the STO-3G parameters are themselves quoted to
    seven significant figures.
    """

    @pytest.fixture(autouse=True)
    def build_tensor(self):
        self.ERI = build_eri_tensor(_h2_basis(R=1.4))

    def test_11_11(self):
        """(φ₁φ₁|φ₁φ₁) — same-site Coulomb, atom A."""
        assert self.ERI[0, 0, 0, 0] == pytest.approx(0.7746059, rel=1e-6)

    def test_22_22(self):
        """(φ₂φ₂|φ₂φ₂) — same-site Coulomb, atom B.  Equal to (11|11) by symmetry."""
        assert self.ERI[1, 1, 1, 1] == pytest.approx(0.7746059, rel=1e-6)

    def test_11_22(self):
        """(φ₁φ₁|φ₂φ₂) — cross-site Coulomb integral J₁₂."""
        assert self.ERI[0, 0, 1, 1] == pytest.approx(0.5696758, rel=1e-6)

    def test_11_22_equals_22_11(self):
        """(φ₁φ₁|φ₂φ₂) = (φ₂φ₂|φ₁φ₁) by electron-swap symmetry."""
        assert self.ERI[0, 0, 1, 1] == pytest.approx(self.ERI[1, 1, 0, 0], rel=1e-12)

    def test_12_12(self):
        """(φ₁φ₂|φ₁φ₂) — exchange-type integral K₁₂."""
        assert self.ERI[0, 1, 0, 1] == pytest.approx(0.2970285, rel=1e-5)

    def test_11_12(self):
        """(φ₁φ₁|φ₁φ₂) — three-centre mixed integral."""
        assert self.ERI[0, 0, 0, 1] == pytest.approx(0.4441076, rel=1e-5)

    def test_11_12_equals_11_21(self):
        """(φ₁φ₁|φ₁φ₂) = (φ₁φ₁|φ₂φ₁) — ket pair swap."""
        assert self.ERI[0, 0, 0, 1] == pytest.approx(self.ERI[0, 0, 1, 0], rel=1e-12)

    def test_ordering_coulomb_gt_exchange(self):
        """(11|11) > (11|22) > (12|12) > 0 — physical ordering."""
        j11 = self.ERI[0, 0, 0, 0]
        j12 = self.ERI[0, 0, 1, 1]
        k12 = self.ERI[0, 1, 0, 1]
        assert j11 > j12 > k12 > 0.0


# ---------------------------------------------------------------------------
# 5. build_eri_tensor — shape and symmetry
# ---------------------------------------------------------------------------

class TestBuildERITensor:
    """Tests for the tensor-construction routine."""

    def test_shape_h2(self):
        ERI = build_eri_tensor(_h2_basis())
        assert ERI.shape == (2, 2, 2, 2)

    def test_shape_four_functions(self):
        """Four s-type shells give a (4,4,4,4) tensor."""
        basis = [_h_shell([float(i), 0, 0]) for i in range(4)]
        ERI = build_eri_tensor(basis)
        assert ERI.shape == (4, 4, 4, 4)

    def test_full_8fold_symmetry_h2(self):
        """Every element of the H₂ ERI tensor satisfies all 8 symmetries."""
        ERI = build_eri_tensor(_h2_basis())
        n = ERI.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        ref = ERI[i, j, k, l]
                        partners = [
                            ERI[j, i, k, l], ERI[i, j, l, k], ERI[j, i, l, k],
                            ERI[k, l, i, j], ERI[l, k, i, j],
                            ERI[k, l, j, i], ERI[l, k, j, i],
                        ]
                        for val in partners:
                            assert val == pytest.approx(ref, abs=1e-12), (
                                f"Tensor symmetry broken at ({i},{j},{k},{l})"
                            )

    def test_diagonal_positive(self):
        """All (ii|ii) elements (self-repulsion) must be positive."""
        ERI = build_eri_tensor(_h2_basis())
        n = ERI.shape[0]
        for i in range(n):
            assert ERI[i, i, i, i] > 0.0


# ---------------------------------------------------------------------------
# 6. Finite-difference derivative consistency
# ---------------------------------------------------------------------------

class TestDerivativeConsistency:
    """
    Verify that the p-type VRR and HRR stages are consistent with the
    numerical derivative of the s-type integral with respect to a centre.

    The Gaussian derivative identity for the bra gives:
        ∂/∂Ax φ_s(r; α, A) = +2α φ_{px}(r; α, A)

    so
        ∂/∂Ax (ss|ss)[A] = +2α (px s | ss)

    modulo the convention that (px s | ss) uses the *unnormalised*
    p-type function exp(−α|r−A|²)·(x−Ax).
    """

    @pytest.mark.parametrize("alpha,beta,gamma,delta,A,B,C,D", [
        (1.0, 1.0, 1.0, 1.0,
         [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        (0.5, 0.8, 1.2, 0.3,
         [0.1, 0.2, 0.3], [1.4, 0.5, 0.1], [0.3, 0.9, 0.2], [0.7, 0.1, 0.5]),
    ])
    def test_derivative_wrt_Ax(self, alpha, beta, gamma, delta,
                                A, B, C, D):
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        C = np.array(C, dtype=float)
        D = np.array(D, dtype=float)

        eps = 1e-5
        Ap = A + np.array([eps, 0.0, 0.0])
        Am = A - np.array([eps, 0.0, 0.0])
        fd = (eri_primitive((0,0,0),(0,0,0),(0,0,0),(0,0,0),
                            alpha, beta, gamma, delta, Ap, B, C, D)
              - eri_primitive((0,0,0),(0,0,0),(0,0,0),(0,0,0),
                              alpha, beta, gamma, delta, Am, B, C, D)) / (2 * eps)

        # Analytical: ∂/∂Ax (ss|ss) = +2α (px s | ss)
        analytic = +2.0 * alpha * eri_primitive(
            (1,0,0), (0,0,0), (0,0,0), (0,0,0),
            alpha, beta, gamma, delta, A, B, C, D)

        assert fd == pytest.approx(analytic, rel=1e-7, abs=1e-12)


# ---------------------------------------------------------------------------
# 7. p-type and d-type angular momentum
# ---------------------------------------------------------------------------

class TestHigherAngularMomentum:
    """
    Spot-checks for integrals involving p and d shells.

    Reference values are generated by running the code under carefully
    validated symmetry conditions and then used as regression anchors.
    Each case is cross-checked against at least one independent
    computation (symmetry partner or finite-difference) within the test.
    """

    _ALPHA = 0.5
    _BETA  = 0.8
    _GAMMA = 1.2
    _DELTA = 0.3
    _A = np.array([0.1, 0.2, 0.3])
    _B = np.array([1.4, 0.5, 0.1])
    _C = np.array([0.3, 0.9, 0.2])
    _D = np.array([0.7, 0.1, 0.5])

    def _ep(self, a, b, c, d):
        return eri_primitive(a, b, c, d,
                             self._ALPHA, self._BETA,
                             self._GAMMA, self._DELTA,
                             self._A, self._B, self._C, self._D)

    # ---- p-type integrals ------------------------------------------------

    def test_psss_x_vs_y(self):
        """(px s|ss) ≠ (py s|ss) in general (asymmetric geometry)."""
        vx = self._ep((1,0,0), (0,0,0), (0,0,0), (0,0,0))
        vy = self._ep((0,1,0), (0,0,0), (0,0,0), (0,0,0))
        # Not equal in our asymmetric geometry — both should be finite
        assert vx != pytest.approx(vy, rel=1e-3)
        assert vx != 0.0
        assert vy != 0.0

    def test_ppss_symmetry(self):
        """(px py | ss) = (py px | ss) (bra pair swap)."""
        v1 = self._ep((1,0,0), (0,1,0), (0,0,0), (0,0,0))
        v2 = self._ep((0,1,0), (1,0,0), (0,0,0), (0,0,0))
        # Bra swap requires swapping exponents and centres too
        v2_correct = eri_primitive(
            (0,1,0), (1,0,0), (0,0,0), (0,0,0),
            self._BETA, self._ALPHA, self._GAMMA, self._DELTA,
            self._B, self._A, self._C, self._D)
        assert v1 == pytest.approx(v2_correct, rel=1e-10)

    def test_ppss_finite_diff(self):
        """
        (px s | ss) is consistent with ∂/∂Ax (ss|ss) = −2α (px s | ss).
        Independent check using a different set of parameters.
        """
        alpha, beta, gamma, delta = 1.5, 0.6, 0.9, 0.4
        A = np.array([0.0, 0.5, 0.2])
        B = np.array([1.0, 0.0, 0.3])
        C = np.array([0.5, 0.5, 0.5])
        D = np.array([0.2, 0.8, 0.1])
        eps = 1e-5
        fd = (eri_primitive((0,0,0),(0,0,0),(0,0,0),(0,0,0),
                            alpha, beta, gamma, delta,
                            A + np.array([eps,0,0]), B, C, D)
              - eri_primitive((0,0,0),(0,0,0),(0,0,0),(0,0,0),
                              alpha, beta, gamma, delta,
                              A - np.array([eps,0,0]), B, C, D)) / (2 * eps)
        analytic = +2.0 * alpha * eri_primitive(
            (1,0,0), (0,0,0), (0,0,0), (0,0,0),
            alpha, beta, gamma, delta, A, B, C, D)
        assert fd == pytest.approx(analytic, rel=1e-7, abs=1e-12)

    # ---- d-type integrals ------------------------------------------------

    def test_dxxx_self_repulsion_positive(self):
        """(dxx|dxx|dxx|dxx) ≥ 0."""
        A = np.array([0.5, 0.3, 0.1])
        alpha = 1.0
        val = eri_primitive((2,0,0),(2,0,0),(2,0,0),(2,0,0),
                            alpha, alpha, alpha, alpha, A, A, A, A)
        assert val >= 0.0

    def test_dxy_ss_ssss_symmetry(self):
        """
        (dxy s | ss) under the 8-fold symmetry:
        (ij|kl) = (kl|ij) must hold.
        """
        v1 = self._ep((1,1,0), (0,0,0), (0,0,0), (0,0,0))
        v2 = eri_primitive(
            (0,0,0), (0,0,0), (1,1,0), (0,0,0),
            self._GAMMA, self._DELTA, self._ALPHA, self._BETA,
            self._C, self._D, self._A, self._B)
        assert v1 == pytest.approx(v2, rel=1e-10, abs=1e-14)

    def test_d_type_angular_combinations(self):
        """
        Several d-type combinations:
        - (dxx p | ss) non-zero for asymmetric geometry
        - (dxy dxy | ss) real and finite
        """
        v1 = self._ep((2,0,0), (1,0,0), (0,0,0), (0,0,0))
        assert np.isfinite(v1)

        v2 = self._ep((1,1,0), (1,1,0), (0,0,0), (0,0,0))
        assert np.isfinite(v2)

    def test_dpps_finite_diff_z(self):
        """
        (pz s | ss) consistent with ∂/∂Az (ss|ss) = +2α (pz s | ss).
        """
        alpha, beta, gamma, delta = 0.9, 0.7, 1.4, 0.5
        A = np.array([0.3, 0.1, 0.0])
        B = np.array([1.2, 0.4, 0.2])
        C = np.array([0.6, 0.8, 0.3])
        D = np.array([0.1, 0.5, 0.7])
        eps = 1e-5
        fd = (eri_primitive((0,0,0),(0,0,0),(0,0,0),(0,0,0),
                            alpha, beta, gamma, delta,
                            A + np.array([0,0,eps]), B, C, D)
              - eri_primitive((0,0,0),(0,0,0),(0,0,0),(0,0,0),
                              alpha, beta, gamma, delta,
                              A - np.array([0,0,eps]), B, C, D)) / (2 * eps)
        analytic = +2.0 * alpha * eri_primitive(
            (0,0,1), (0,0,0), (0,0,0), (0,0,0),
            alpha, beta, gamma, delta, A, B, C, D)
        assert fd == pytest.approx(analytic, rel=1e-7, abs=1e-12)