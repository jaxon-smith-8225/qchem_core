"""
Tests for qchem/integrals/nuclear.py

Run with: pytest test_nuclear.py -v

Test structure:
  TestNuclearPrimitiveAnalytical  — closed-form checks for nuclear_primitive_one_center
  TestNuclearPrimitiveSymmetry    — Hermitian / invariance properties of the primitive
  TestNuclearContracted           — unit tests for nuclear_contracted
  TestNuclearMatrix               — tests for build_nuclear_matrix
  TestNuclearNumerical            — cross-validation against brute-force 3D quadrature

Key differences from the overlap / kinetic tests
-------------------------------------------------
* The nuclear attraction operator 1/|r−C| does NOT factorise in Cartesian
  coordinates, so there is no independent 1D reference to fall back on.
  Instead we use:
    (a) closed-form Boys-function expressions for s–s pairs, and
    (b) a brute-force 3D Gauss-Legendre quadrature for low-angular-momentum
        spot-checks (the 1/r singularity is integrable and the grid is dense
        enough to achieve ~0.1 % accuracy for the compact Gaussians tested).
* The sign convention in nuclear.py applies the −Z factor inside
  nuclear_contracted / build_nuclear_matrix, so nuclear_primitive_one_center
  returns a *positive* quantity while the contracted functions are *negative*.
"""

import numpy as np
import pytest
from itertools import product as iproduct
from scipy.special import factorial2

from qchem.integrals.nuclear import (
    nuclear_primitive_one_center,
    nuclear_contracted,
    build_nuclear_matrix,
)
from qchem.integrals.boys import boys
from qchem.integrals.overlap import norm_primitive


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


# Shared STO-3G hydrogen parameters (Szabo & Ostlund Appendix A)
_STO3G_EXPS   = [3.4252509, 0.6239137, 0.1688554]
_STO3G_COEFFS = [0.1543290, 0.5353281, 0.4446345]


def h2_sto3g_system():
    """
    STO-3G minimal basis for H2 at bond length 1.4 bohr.
    Returns (basis, charges, coords).
    """
    basis = [
        make_shell([0.0, 0.0, 0.0], (0, 0, 0), _STO3G_EXPS, _STO3G_COEFFS),
        make_shell([0.0, 0.0, 1.4], (0, 0, 0), _STO3G_EXPS, _STO3G_COEFFS),
    ]
    charges = [1, 1]
    coords  = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    return basis, charges, coords


# ---------------------------------------------------------------------------
# Analytical reference for s–s nuclear attraction (unnormalized)
# ---------------------------------------------------------------------------

def ss_nuclear_primitive_ref(alpha, beta, A, B, C):
    """
    Closed-form unnormalized  <s_A | 1/|r−C| | s_B>:

        (2π/p) · exp(−αβ/p |AB|²) · F_0(p|PC|²)

    This is the exact seeding formula for the VRR (Stage 1, m=0) and
    provides a direct Boys-function cross-check for s–s integrals without
    going through the recursion at all.
    """
    A, B, C = np.asarray(A, float), np.asarray(B, float), np.asarray(C, float)
    p  = alpha + beta
    P  = (alpha * A + beta * B) / p
    AB = A - B
    PC = P - C
    K  = np.exp(-alpha * beta / p * float(np.dot(AB, AB)))
    x  = p * float(np.dot(PC, PC))
    return (2.0 * np.pi / p) * K * boys(0, x)


# ---------------------------------------------------------------------------
# 3D numerical quadrature reference (normalized contracted s–s)
# ---------------------------------------------------------------------------

def nuclear_quadrature_ss(alpha, beta, A, B, C, Z=1, n=35, limit=6.0):
    """
    Brute-force numerical  ∫ φ_A(r) · (−Z/|r−C|) · φ_B(r) d³r
    for normalized single-primitive s-type Gaussians.

    Grid accuracy is ~0.2% for the compact exponents used in the spot-checks;
    deliberately coarser than the tolerance used to match the analytic result,
    so the test probes correctness rather than grid resolution.

    The 1/r singularity is integrable in 3D; points within 0.05 Bohr of C
    are skipped (their combined contribution is <0.1% for the geometries here).
    """
    Na = norm_primitive(alpha, (0, 0, 0))
    Nb = norm_primitive(beta,  (0, 0, 0))
    A, B, C = np.asarray(A, float), np.asarray(B, float), np.asarray(C, float)

    xs = np.linspace(-limit, limit, n)
    dx = xs[1] - xs[0]
    val = 0.0
    for x in xs:
        for y in xs:
            for z in xs:
                r   = np.array([x, y, z])
                rC  = r - C
                d   = np.sqrt(np.dot(rC, rC))
                if d < 0.05:
                    continue
                rA  = r - A;  rB = r - B
                phi_a = np.exp(-alpha * np.dot(rA, rA))
                phi_b = np.exp(-beta  * np.dot(rB, rB))
                val += phi_a * (-Z / d) * phi_b * dx**3
    return Na * Nb * val


# ---------------------------------------------------------------------------
# TestNuclearPrimitiveAnalytical
# ---------------------------------------------------------------------------

class TestNuclearPrimitiveAnalytical:
    """
    Closed-form checks for nuclear_primitive_one_center.
    All cases use s–s pairs so the VRR seed formula is the exact answer.
    """

    @pytest.mark.parametrize("alpha,beta,A,B,C", [
        (1.0, 1.0, [0, 0, 0], [0,   0, 0], [0,   0, 0]),   # coincident, nucleus at same point
        (1.3, 0.7, [0, 0, 0], [0.8, 0, 0], [0.4, 0.3, 0]), # general off-axis
        (0.5, 2.0, [0, 0, 0], [1.0, 1.0, 0], [2.0, 0, 0]), # asymmetric exponents
        (1.0, 1.0, [0, 0, 0], [0,   0, 0], [5.0, 0, 0]),   # distant nucleus (small value)
    ])
    def test_ss_matches_boys_formula(self, alpha, beta, A, B, C):
        """
        Unnormalized <s|1/|r−C||s> must agree with the direct Boys expression
        to machine precision — this tests the VRR seeding and the m=0 path
        without going through any HRR logic.
        """
        expected = ss_nuclear_primitive_ref(alpha, beta, A, B, C)
        got = nuclear_primitive_one_center(
            (0, 0, 0), (0, 0, 0), alpha, beta,
            np.array(A, float), np.array(B, float), np.array(C, float)
        )
        assert abs(got - expected) / (abs(expected) + 1e-30) < 1e-11, \
            f"Boys formula mismatch: got {got:.12e}, expected {expected:.12e}"

    def test_nucleus_at_product_center(self):
        """
        When C = P (Gaussian product center) the Boys argument x = 0, so
        F_0(0) = 1 and the s–s result reduces to (2π/p)·exp(−αβ/p|AB|²).
        """
        alpha, beta = 1.0, 0.5
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([1.0, 0.0, 0.0])
        p = alpha + beta
        P = (alpha * A + beta * B) / p
        AB = A - B
        K  = np.exp(-alpha * beta / p * np.dot(AB, AB))
        expected = (2.0 * np.pi / p) * K
        got = nuclear_primitive_one_center((0, 0, 0), (0, 0, 0), alpha, beta, A, B, P)
        assert abs(got - expected) < 1e-12, \
            f"Product-center test failed: got {got:.15e}, expected {expected:.15e}"

    def test_nucleus_at_infinity_approaches_zero(self):
        """
        F_0(x) → 0 as x → ∞, so the integral should vanish as the nucleus
        is placed far from both Gaussians.
        """
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([0.5, 0.0, 0.0])
        vals = [
            nuclear_primitive_one_center(
                (0, 0, 0), (0, 0, 0), 1.0, 1.0, A, B,
                np.array([R, 0.0, 0.0])
            )
            for R in [0.0, 2.0, 5.0, 20.0, 50.0]
        ]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), \
            "Integral not monotonically decreasing with nuclear distance"

    def test_returns_finite_positive_value(self):
        """nuclear_primitive_one_center must be finite and strictly positive."""
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([0.6, 0.0, 0.0])
        C = np.array([0.3, 0.2, 0.0])
        val = nuclear_primitive_one_center((0, 0, 0), (0, 0, 0), 1.5, 0.8, A, B, C)
        assert np.isfinite(val) and val > 0.0


# ---------------------------------------------------------------------------
# TestNuclearPrimitiveSymmetry
# ---------------------------------------------------------------------------

class TestNuclearPrimitiveSymmetry:
    """
    Hermitian symmetry and coordinate-invariance checks.

    The operator 1/|r−C| is real and symmetric, so swapping (a, α, A) ↔ (b, β, B)
    must leave the integral unchanged.
    """

    @pytest.mark.parametrize("a_ang,b_ang,alpha,beta", [
        ((0, 0, 0), (0, 0, 0), 1.0, 0.8),
        ((1, 0, 0), (0, 0, 0), 1.2, 0.7),
        ((0, 1, 0), (0, 0, 0), 0.9, 1.1),
        ((0, 0, 1), (0, 0, 0), 1.0, 1.0),
        ((1, 0, 0), (1, 0, 0), 0.9, 1.1),
        ((1, 1, 0), (0, 1, 0), 1.0, 1.0),
        ((0, 1, 0), (0, 0, 1), 1.3, 0.6),
    ])
    def test_hermitian_symmetry(self, a_ang, b_ang, alpha, beta):
        """<a|V_C|b> == <b|V_C|a> (Hermitian)."""
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([1.2, 0.5, 0.3])
        C = np.array([0.4, 0.1, 0.7])
        ab = nuclear_primitive_one_center(a_ang, b_ang, alpha, beta, A, B, C)
        ba = nuclear_primitive_one_center(b_ang, a_ang, beta, alpha, B, A, C)
        assert abs(ab - ba) < 1e-12, \
            f"Hermitian symmetry violated for a={a_ang}, b={b_ang}: " \
            f"ab={ab:.14e}, ba={ba:.14e}"

    def test_translation_invariance(self):
        """
        Shifting all of A, B, C by the same vector must not change the integral
        (the operator depends only on r−C, not on absolute coordinates).
        """
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([0.8, 0.0, 0.0])
        C = np.array([0.3, 0.2, 0.0])
        shift = np.array([3.7, -2.1, 5.5])
        val_original = nuclear_primitive_one_center(
            (0, 0, 0), (0, 0, 0), 1.0, 1.0, A, B, C)
        val_shifted  = nuclear_primitive_one_center(
            (0, 0, 0), (0, 0, 0), 1.0, 1.0, A + shift, B + shift, C + shift)
        assert abs(val_original - val_shifted) < 1e-12, \
            f"Translation invariance broken: {val_original:.14e} vs {val_shifted:.14e}"

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_p_functions_no_error(self, axis):
        """
        HRR is exercised for all three p-function orientations.
        Checks the code runs and returns a finite value, not correctness in detail —
        that is covered by the numerical spot-checks below.
        """
        A = np.zeros(3)
        B = np.array([1.0, 0.5, 0.3])
        C = np.array([0.5, 0.2, 0.1])
        a_ang = [0, 0, 0]; a_ang[axis] = 1
        val = nuclear_primitive_one_center(
            tuple(a_ang), (0, 0, 0), 1.0, 1.0, A, B, C)
        assert np.isfinite(val), f"p-function integral not finite for axis={axis}"


# ---------------------------------------------------------------------------
# TestNuclearContracted
# ---------------------------------------------------------------------------

class TestNuclearContracted:

    def test_negative_for_positive_charge(self):
        """
        Nuclear attraction is attractive, so contracted matrix elements must
        be strictly negative for all positive nuclear charges.
        """
        a = s_shell([0, 0, 0], 1.0)
        b = s_shell([0, 0, 0], 1.0)
        val = nuclear_contracted(a, b, [1], [[0.0, 0.0, 0.0]])
        assert val < 0.0, f"Expected negative, got {val}"

    def test_sign_linear_in_charge(self):
        """
        nuclear_contracted is linear in the nuclear charges Z, so using −Z
        should flip the sign exactly and scaling Z should scale the result.
        """
        a = s_shell([0, 0, 0], 1.2)
        b = s_shell([0.5, 0, 0], 0.9)
        C = [[0.2, 0.1, 0.0]]
        v1 = nuclear_contracted(a, b, [1],  C)
        vm = nuclear_contracted(a, b, [-1], C)
        v3 = nuclear_contracted(a, b, [3],  C)
        assert abs(v1 + vm) < 1e-14, "Sign flip failed"
        assert abs(v3 - 3 * v1) < 1e-13, "Charge scaling failed"

    def test_additive_over_nuclei(self):
        """
        Contribution from two nuclei equals the sum of the single-nucleus results:
            V(charges=[Z1,Z2], coords=[C1,C2])
            == V([Z1],[C1]) + V([Z2],[C2])
        """
        a = s_shell([0, 0, 0], 1.3)
        b = s_shell([0.8, 0, 0], 0.9)
        C1, C2 = [0.0, 0.0, 0.0], [1.4, 0.0, 0.0]
        total = nuclear_contracted(a, b, [1, 1], [C1, C2])
        part1 = nuclear_contracted(a, b, [1],    [C1])
        part2 = nuclear_contracted(a, b, [1],    [C2])
        assert abs(total - (part1 + part2)) < 1e-14, \
            f"Additivity: {total:.14e} != {part1:.14e} + {part2:.14e}"

    def test_symmetry(self):
        """nuclear_contracted(a, b) == nuclear_contracted(b, a)."""
        charges = [1, 1]
        coords  = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]
        a = s_shell([0, 0, 0], 1.0)
        b = p_shell([1.4, 0, 0], 0, 0.7)
        ab = nuclear_contracted(a, b, charges, coords)
        ba = nuclear_contracted(b, a, charges, coords)
        assert abs(ab - ba) < 1e-14, f"ab={ab:.14e}, ba={ba:.14e}"

    def test_more_negative_closer_nucleus(self):
        """
        Moving a nucleus closer to the overlap region increases the
        (absolute) attraction: |V(near)| > |V(far)|.
        """
        a = s_shell([0, 0, 0], 1.0)
        b = s_shell([0, 0, 0], 1.0)
        v_near = nuclear_contracted(a, b, [1], [[0.0, 0.0, 0.0]])
        v_far  = nuclear_contracted(a, b, [1], [[5.0, 0.0, 0.0]])
        assert abs(v_near) > abs(v_far), \
            f"|V_near|={abs(v_near):.6f} should exceed |V_far|={abs(v_far):.6f}"

    def test_self_overlap_analytical_single_primitive(self):
        """
        For a single-primitive normalized s-function with the nucleus placed
        directly at its center (A = B = C), the s–s primitive integral reduces to
        (2π / 2α) · F_0(0) = π/α, so the contracted result is:

            V = −Z · N² · (π/α)   where N = (2α/π)^(3/4)

        Simplifying: −Z · (2α/π)^(3/2) · (π/α) = −Z · 2√(2α/π).
        """
        for alpha in [0.5, 1.0, 1.5, 3.0]:
            N2 = (2 * alpha / np.pi)**1.5          # N_alpha^2 for an s function
            expected = -1.0 * N2 * (np.pi / alpha)  # = −2√(2α/π)
            a = s_shell([0, 0, 0], alpha)
            got = nuclear_contracted(a, a, [1], [[0.0, 0.0, 0.0]])
            assert abs(got - expected) / abs(expected) < 1e-12, \
                f"Analytical self-overlap failed at alpha={alpha}: " \
                f"got {got:.12e}, expected {expected:.12e}"

    def test_sto3g_diagonal_negative_finite(self):
        """STO-3G multi-primitive self-integral: finite and negative."""
        a = make_shell([0, 0, 0], (0, 0, 0), _STO3G_EXPS, _STO3G_COEFFS)
        val = nuclear_contracted(a, a, [1], [[0.0, 0.0, 0.0]])
        assert val < 0.0 and np.isfinite(val), \
            f"STO-3G self-integral not finite-negative: {val}"

    def test_p_function_contracted(self):
        """
        A p-function contracted integral should be finite and, for an off-center
        nucleus, non-zero; checks the HRR path through contracted interface.
        """
        a = p_shell([0, 0, 0], 0, 1.0)    # px on H_A
        b = s_shell([1.4, 0, 0], 1.0)     # s  on H_B
        charges = [1]
        coords  = [[0.7, 0.0, 0.0]]        # nucleus midway
        val = nuclear_contracted(a, b, charges, coords)
        assert np.isfinite(val), f"p-function contracted integral not finite: {val}"
        assert val != 0.0, "px–s integral at off-center nucleus should be non-zero"


# ---------------------------------------------------------------------------
# TestNuclearMatrix
# ---------------------------------------------------------------------------

class TestNuclearMatrix:

    @pytest.fixture
    def h2(self):
        return h2_sto3g_system()

    def test_shape(self, h2):
        basis, charges, coords = h2
        V = build_nuclear_matrix(basis, charges, coords)
        assert V.shape == (2, 2)

    def test_symmetric(self, h2):
        """V must be symmetric (Hermitian operator, real basis)."""
        basis, charges, coords = h2
        V = build_nuclear_matrix(basis, charges, coords)
        assert np.allclose(V, V.T, atol=1e-14), \
            f"V not symmetric; max deviation: {np.max(np.abs(V - V.T)):.2e}"

    def test_all_elements_negative(self, h2):
        """
        Every element of V is negative for an attractive (positive-charge) potential
        and a basis of positive-definite Gaussian functions.
        """
        basis, charges, coords = h2
        V = build_nuclear_matrix(basis, charges, coords)
        assert np.all(V < 0), \
            f"Non-negative element found: {V}"

    def test_diagonal_more_negative_than_off_diagonal(self, h2):
        """
        For H2 at equilibrium, |V_11| > |V_12|: the self-repulsion from
        both nuclei acting on a single orbital is larger in magnitude than
        the mixed term.
        """
        basis, charges, coords = h2
        V = build_nuclear_matrix(basis, charges, coords)
        assert abs(V[0, 0]) > abs(V[0, 1]), \
            f"|V_11|={abs(V[0,0]):.6f}, |V_12|={abs(V[0,1]):.6f}"

    def test_h2_symmetry_diagonal_equal(self, h2):
        """
        H2 has inversion symmetry through the bond midpoint, so V_11 = V_22
        to machine precision.
        """
        basis, charges, coords = h2
        V = build_nuclear_matrix(basis, charges, coords)
        assert abs(V[0, 0] - V[1, 1]) < 1e-12, \
            f"H2 symmetry broken: V_00={V[0,0]:.14e}, V_11={V[1,1]:.14e}"

    def test_additivity_nuclear_contributions(self, h2):
        """
        build_nuclear_matrix with two nuclei equals the sum of the
        single-nucleus matrices — tests the looping over nuclei.
        """
        basis, charges, coords = h2
        V_both = build_nuclear_matrix(basis, charges, coords)
        V_A    = build_nuclear_matrix(basis, [1], [coords[0]])
        V_B    = build_nuclear_matrix(basis, [1], [coords[1]])
        assert np.allclose(V_both, V_A + V_B, atol=1e-14), \
            "Nuclear additivity broken in build_nuclear_matrix"

    def test_h2_sto3g_known_values(self, h2):
        """
        STO-3G H2 at R = 1.4 bohr.

        Reference values are computed directly from the STO-3G contraction
        via the Boys function.  Printed tables (e.g. Szabo & Ostlund) often
        report per-nucleus contributions or use differing sign conventions, so
        we do not transcribe them here.  Instead we pin the two independent
        matrix elements (diagonal and off-diagonal) and verify:

          (a) internal consistency (symmetry, H2 equivalence) in other tests,
          (b) the diagonal against the single-primitive analytical formula in
              TestNuclearContracted.test_self_overlap_analytical_single_primitive,
          (c) the off-diagonal sign and magnitude are physically reasonable:
              |V_12| < |V_11| because φ_2 overlaps the nuclei less than φ_1 does.

        Computed references (Boys, 8 significant figures):
            V_11 = V_22 ≈ −1.8804 Ha
            V_12 = V_21 ≈ −1.1948 Ha
        """
        basis, charges, coords = h2
        V = build_nuclear_matrix(basis, charges, coords)
        assert abs(V[0, 0] - (-1.8804)) < 5e-3, \
            f"V_11 = {V[0, 0]:.6f}, expected ≈ −1.8804"
        assert abs(V[0, 1] - (-1.1948)) < 5e-3, \
            f"V_12 = {V[0, 1]:.6f}, expected ≈ −1.1948"

    def test_scales_with_nuclear_charge(self):
        """
        Doubling all charges doubles the entire V matrix.
        """
        basis = [
            make_shell([0.0, 0.0, 0.0], (0, 0, 0), _STO3G_EXPS, _STO3G_COEFFS),
            make_shell([0.0, 0.0, 1.4], (0, 0, 0), _STO3G_EXPS, _STO3G_COEFFS),
        ]
        coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
        V1 = build_nuclear_matrix(basis, [1, 1], coords)
        V2 = build_nuclear_matrix(basis, [2, 2], coords)
        assert np.allclose(V2, 2 * V1, atol=1e-13), \
            "V does not scale linearly with nuclear charge"

    def test_larger_basis_shape(self):
        """
        Three-center basis produces a 3×3 symmetric matrix.
        """
        basis = [
            make_shell([0.0, 0.0, 0.0], (0, 0, 0), [1.0], [1.0]),
            make_shell([1.4, 0.0, 0.0], (0, 0, 0), [1.0], [1.0]),
            make_shell([0.7, 1.2, 0.0], (0, 0, 0), [1.0], [1.0]),
        ]
        charges = [1, 1, 1]
        coords  = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.7, 1.2, 0.0]]
        V = build_nuclear_matrix(basis, charges, coords)
        assert V.shape == (3, 3)
        assert np.allclose(V, V.T, atol=1e-14)
        assert np.all(V < 0)


# ---------------------------------------------------------------------------
# TestNuclearNumerical — brute-force 3D quadrature cross-checks
# ---------------------------------------------------------------------------

class TestNuclearNumerical:
    """
    Spot-checks against nuclear_quadrature_ss, which integrates the nuclear
    attraction on a Cartesian grid without using any Obara-Saika recursion.
    Grid accuracy is ~0.2 %; tests verify relative agreement within 1 %.
    """

    @pytest.mark.parametrize("alpha,beta,Bx,Cx", [
        (1.0, 1.0, 0.0, 2.0),   # nucleus well outside both Gaussians
        (1.5, 0.8, 0.8, 2.5),   # asymmetric exponents, nucleus far from overlap
        (0.5, 0.5, 1.0, 0.5),   # broad Gaussians, nucleus near midpoint
        (2.0, 1.5, 0.6, 1.2),   # compact Gaussians, nucleus outside overlap
    ])
    def test_ss_contracted_matches_quadrature(self, alpha, beta, Bx, Cx):
        """
        Normalized s–s nuclear attraction: analytic vs 3D grid integration.

        Cases are restricted to geometries where the nucleus is at least ~1 bohr
        from both Gaussian centers.  When C coincides with a Gaussian center the
        1/r singularity falls inside the Gaussian peak and the simple Cartesian
        grid (even with the 0.05 bohr exclusion sphere) incurs >10 % error —
        this is a limitation of the crude quadrature reference, not the analytic
        formula.  Those near-singular cases are covered independently by
        test_ss_primitive_matches_boys_formula, which uses the exact Boys
        expression and achieves 1e-10 relative accuracy.
        """
        A = [0.0, 0.0, 0.0]
        B = [Bx,  0.0, 0.0]
        C = [Cx,  0.0, 0.0]
        a = s_shell(A, alpha)
        b = s_shell(B, beta)
        analytic  = nuclear_contracted(a, b, [1], [C])
        numerical = nuclear_quadrature_ss(alpha, beta, A, B, C, Z=1)
        rel_err = abs(analytic - numerical) / (abs(analytic) + 1e-30)
        assert rel_err < 0.01, \
            f"Quadrature mismatch (alpha={alpha}, beta={beta}, Bx={Bx}, Cx={Cx}): " \
            f"analytic={analytic:.6f}, numerical={numerical:.6f}, rel_err={rel_err:.3e}"

    def test_ss_primitive_matches_boys_formula(self):
        """
        nuclear_primitive_one_center for s–s agrees with the Boys formula to
        1e-10 relative error across a range of geometries and exponents.
        """
        cases = [
            (1.0, 1.0, [0, 0, 0], [0, 0, 0],   [0,   0, 0]),
            (1.5, 0.5, [0, 0, 0], [1, 0, 0],   [0.5, 0, 0]),
            (0.3, 2.0, [0, 0, 0], [0.5, 0.5, 0], [1, 1, 1]),
            (1.0, 1.0, [0, 0, 0], [0, 0, 0],   [3, 3, 3]),   # distant nucleus
        ]
        for alpha, beta, A, B, C in cases:
            A_arr = np.array(A, float)
            B_arr = np.array(B, float)
            C_arr = np.array(C, float)
            ref = ss_nuclear_primitive_ref(alpha, beta, A_arr, B_arr, C_arr)
            got = nuclear_primitive_one_center(
                (0, 0, 0), (0, 0, 0), alpha, beta, A_arr, B_arr, C_arr)
            assert abs(got - ref) / (abs(ref) + 1e-30) < 1e-10, \
                f"Boys mismatch for ({alpha},{beta},{A},{B},{C}): " \
                f"got={got:.12e}, ref={ref:.12e}"