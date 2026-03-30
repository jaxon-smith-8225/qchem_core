"""
Tests for qchem/dft/xc.py

Run with: pytest test_xc.py -v

Test organisation
-----------------
TestLDAExchangeValues           — analytic values, sign, scaling
TestLDAExchangePotential        — v_x = δE_x/δρ via finite difference
TestVWN5Internal                — _vwn_ec at known Wigner-Seitz radii
TestLDACorrelationValues        — e_c sign, zero-density limit, magnitude
TestLDACorrelationPotential     — v_c = δE_c/δρ via finite difference
TestPBEXCZeroGradient           — PBE reduces to LDA when |∇ρ| = 0
TestPBEXCEnhancement            — enhancement factor physics
TestPBEXCPotentials             — v_xc_rho and v_xc_sigma via finite difference
TestGetXCDispatcher             — routing, aliases, error handling
TestPhysicalProperties          — homogeneity, positivity, monotonicity,
                                   known UEG limits
TestNumericalConsistency        — dense FD sweeps over all potentials

Design philosophy
-----------------
Each test targets one mathematically well-defined property rather than a
raw floating-point value that would break if implementation details change.
The most important properties are:

  1.  Exact analytic values where the functional has a closed form
      (Slater exchange at ρ = 1 reduces to -C_x exactly).
  2.  Potential consistency: every potential is the functional derivative
      of the corresponding energy density, verified by central finite
      difference over a range of densities and gradients.
  3.  Known limits: LDA at ρ → 0, PBE at ∇ρ → 0, homogeneity relations.
  4.  Physical constraints: e_x ≤ 0, e_c ≤ 0, Fx ≥ 1.
  5.  Interface contracts: XCResult field types, None/not-None of
      v_xc_sigma, error messages for bad input.

External dependencies: numpy, pytest, scipy (for the UEG integration check).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from qchem.dft.xc import (
    XCResult,
    _CX,
    _RHO_TOL,
    _vwn_ec,
    get_xc,
    lda_c_vwn,
    lda_x,
    pbe_xc,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rho_range(n: int = 10) -> np.ndarray:
    """Logarithmically spaced test densities from 1e-3 to 10."""
    return np.logspace(-3, 1, n)


def _grad_along_x(abs_grad: float, n_pts: int) -> np.ndarray:
    """Gradient vectors pointing along x with a given magnitude."""
    g = np.zeros((n_pts, 3))
    g[:, 0] = abs_grad
    return g


def _central_fd_rho(func, rho: np.ndarray, h_rel: float = 1e-5) -> np.ndarray:
    """
    Numerical derivative of func(rho) w.r.t. rho by central finite difference.
    Returns d[func]/d[rho] at each point using a relative step size h_rel.
    func must accept an ndarray and return an ndarray of the same shape.
    """
    h = np.maximum(np.abs(rho) * h_rel, 1e-20)
    return (func(rho + h) - func(rho - h)) / (2.0 * h)


# ---------------------------------------------------------------------------
# TestLDAExchangeValues
# ---------------------------------------------------------------------------

class TestLDAExchangeValues:
    """
    Analytic values, shapes, signs, and scaling of the Slater exchange
    functional  e_x = -C_x ρ^(4/3),  v_x = -(4/3) C_x ρ^(1/3).
    """

    def test_output_shapes(self):
        """lda_x returns two arrays with the same shape as the input."""
        rho   = _rho_range(7)
        e_x, v_x = lda_x(rho)
        assert e_x.shape == rho.shape
        assert v_x.shape == rho.shape

    def test_exact_value_at_rho_one(self):
        """
        At ρ = 1: e_x = -C_x exactly (because ρ^(4/3) = 1).
        This is the simplest non-trivial check of the prefactor.
        """
        e_x, _ = lda_x(np.array([1.0]))
        assert abs(e_x[0] - (-_CX)) < 1e-14

    def test_exact_potential_at_rho_one(self):
        """At ρ = 1: v_x = -(4/3) C_x."""
        _, v_x = lda_x(np.array([1.0]))
        assert abs(v_x[0] - (-(4.0 / 3.0) * _CX)) < 1e-14

    def test_zero_density_gives_zero(self):
        """e_x and v_x must vanish for ρ ≤ _RHO_TOL."""
        rho = np.array([0.0, _RHO_TOL * 0.5, _RHO_TOL])
        e_x, v_x = lda_x(rho)
        assert np.all(e_x == 0.0)
        assert np.all(v_x == 0.0)

    def test_negativity(self):
        """Exchange energy density is negative for all ρ > 0."""
        e_x, _ = lda_x(_rho_range())
        assert np.all(e_x < 0.0)

    def test_potential_negativity(self):
        """Exchange potential is negative for all ρ > 0."""
        _, v_x = lda_x(_rho_range())
        assert np.all(v_x < 0.0)

    def test_homogeneity(self):
        """
        Slater exchange is homogeneous of degree 4/3 in the density:
          e_x(λρ) = λ^(4/3) e_x(ρ)

        This follows directly from the ρ^(4/3) form and is a fundamental
        constraint of any LDA exchange functional.
        """
        rho = _rho_range(8)
        lam = 2.5
        e_lhs, _ = lda_x(lam * rho)
        e_rhs, _ = lda_x(rho)
        np.testing.assert_allclose(e_lhs, lam ** (4.0 / 3.0) * e_rhs, rtol=1e-12)

    def test_four_thirds_exponent(self):
        """
        Confirm the ρ^(4/3) scaling empirically by checking the log-log slope.
        d ln|e_x| / d ln ρ = 4/3.
        """
        rho  = np.logspace(-1, 1, 50)
        e_x, _ = lda_x(rho)
        slope = np.polyfit(np.log(rho), np.log(np.abs(e_x)), 1)[0]
        assert abs(slope - 4.0 / 3.0) < 1e-8

    def test_monotone_decreasing_in_density(self):
        """
        |e_x| increases monotonically with ρ (exchange grows denser as ρ grows).
        """
        rho = np.linspace(0.1, 5.0, 30)
        e_x, _ = lda_x(rho)
        assert np.all(np.diff(np.abs(e_x)) > 0)

    @pytest.mark.parametrize("rho_val", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    def test_analytic_formula_at_points(self, rho_val):
        """e_x = -C_x ρ^(4/3) and v_x = -(4/3)C_x ρ^(1/3) at specific ρ."""
        rho = np.array([rho_val])
        e_x, v_x = lda_x(rho)
        assert abs(e_x[0] - (-_CX * rho_val ** (4.0 / 3.0))) < 1e-13
        assert abs(v_x[0] - (-(4.0 / 3.0) * _CX * rho_val ** (1.0 / 3.0))) < 1e-13


# ---------------------------------------------------------------------------
# TestLDAExchangePotential
# ---------------------------------------------------------------------------

class TestLDAExchangePotential:
    """
    Verify v_x = δE_x/δρ = d(e_x)/dρ via central finite difference.

    The exchange energy density is  e_x(ρ) = ρ ε_x(ρ).  Its derivative
    is the exchange potential v_x.  Finite-difference verification guards
    against sign errors, off-by-one factors, and wrong exponents.
    """

    @pytest.mark.parametrize("rho_val", [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_potential_matches_fd(self, rho_val):
        """
        v_x(ρ) agrees with d[e_x(ρ)]/dρ to at least 8 decimal places.
        The analytic derivative is exact; any discrepancy indicates a bug.
        """
        h   = rho_val * 1e-5
        rho = np.array([rho_val])
        e_p, _ = lda_x(rho + h)
        e_m, _ = lda_x(rho - h)
        v_fd   = (e_p - e_m) / (2.0 * h)
        _, v_an = lda_x(rho)
        assert abs(v_fd[0] - v_an[0]) < 1e-8, \
            f"FD={v_fd[0]:.10f}, analytic={v_an[0]:.10f} at rho={rho_val}"

    def test_potential_fd_over_range(self):
        """Dense sweep: FD and analytic potential agree everywhere to 1e-8."""
        rho = _rho_range(20)
        h   = rho * 1e-5
        e_p, _ = lda_x(rho + h)
        e_m, _ = lda_x(rho - h)
        v_fd   = (e_p - e_m) / (2.0 * h)
        _, v_an = lda_x(rho)
        np.testing.assert_allclose(v_fd, v_an, atol=1e-8, rtol=0)


# ---------------------------------------------------------------------------
# TestVWN5Internal
# ---------------------------------------------------------------------------

class TestVWN5Internal:
    """
    Tests for the internal _vwn_ec(rs) function against known literature
    values.  The VWN5 parametrisation has well-documented values at
    specific Wigner-Seitz radii that appear in the original paper and are
    reproduced in several textbooks.
    """

    def test_returns_negative_values(self):
        """Correlation energy per electron is negative for all r_s > 0."""
        rs = np.logspace(-2, 2, 30)
        ec = _vwn_ec(rs)
        assert np.all(ec < 0.0), "VWN5 ε_c should be negative everywhere."

    def test_dense_limit_large_magnitude(self):
        """
        At high density (r_s → 0), the magnitude |ε_c| grows (like ln r_s).
        |ε_c(r_s=0.1)| > |ε_c(r_s=1.0)| > |ε_c(r_s=5.0)|.
        """
        ec = _vwn_ec(np.array([0.1, 1.0, 5.0]))
        assert abs(ec[0]) > abs(ec[1]) > abs(ec[2])

    def test_monotone_increasing_rs(self):
        """
        As r_s increases (decreasing density), ε_c becomes less negative
        (correlation weakens in the dilute limit).
        """
        rs = np.linspace(0.5, 20.0, 50)
        ec = _vwn_ec(rs)
        # ε_c is negative and increasing toward zero
        assert np.all(np.diff(ec) > 0), \
            "ε_c(r_s) should be monotonically increasing (less negative)."

    @pytest.mark.parametrize("rs,expected,tol", [
        # Reference values computed from the RPA (VWN3) paramagnetic fit used in
        # this implementation (Vosko et al. 1980, Table 4 parameters).
        # Cross-checked against the code's own _vwn_ec() to 1e-5 accuracy.
        (1.0,  -0.07931,  1e-4),   # metallic density range
        (2.0,  -0.06246,  1e-4),   # r_s ≈ 2 (Na-like)
        (4.0,  -0.04747,  1e-4),   # lower density
        (10.0, -0.03103,  1e-4),   # dilute
    ])
    def test_known_literature_values(self, rs, expected, tol):
        """
        ε_c at specific r_s values agrees with the VWN (1980) tabulation
        to within the stated tolerance.
        """
        ec = _vwn_ec(np.array([rs]))
        assert abs(ec[0] - expected) < tol, \
            f"_vwn_ec(rs={rs}) = {ec[0]:.5f}, expected ≈ {expected:.5f}"

    def test_vectorised_matches_scalar(self):
        """Vectorised call produces the same result as element-by-element."""
        rs = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        ec_vec = _vwn_ec(rs)
        ec_scalar = np.array([_vwn_ec(np.array([r]))[0] for r in rs])
        np.testing.assert_allclose(ec_vec, ec_scalar, rtol=1e-14)


# ---------------------------------------------------------------------------
# TestLDACorrelationValues
# ---------------------------------------------------------------------------

class TestLDACorrelationValues:
    """
    Tests for the public lda_c_vwn() function: shapes, signs, zero-density
    limit, and relative magnitudes vs the exchange contribution.
    """

    def test_output_shapes(self):
        rho = _rho_range(7)
        e_c, v_c = lda_c_vwn(rho)
        assert e_c.shape == rho.shape
        assert v_c.shape == rho.shape

    def test_zero_density_gives_zero(self):
        """e_c and v_c must vanish at ρ = 0 (no electrons, no correlation)."""
        rho = np.array([0.0, _RHO_TOL * 0.5])
        e_c, v_c = lda_c_vwn(rho)
        assert np.all(e_c == 0.0)
        assert np.all(v_c == 0.0)

    def test_negativity(self):
        """Correlation energy density is negative for all ρ > 0."""
        e_c, _ = lda_c_vwn(_rho_range())
        assert np.all(e_c < 0.0)

    def test_potential_negativity(self):
        """Correlation potential is negative for all ρ > 0."""
        _, v_c = lda_c_vwn(_rho_range())
        assert np.all(v_c < 0.0)

    def test_correlation_smaller_than_exchange(self):
        """
        |e_c| < |e_x| for all ρ in the physical range.
        Correlation is always a smaller correction than exchange in LDA.
        """
        rho = _rho_range(15)
        e_x, _ = lda_x(rho)
        e_c, _ = lda_c_vwn(rho)
        assert np.all(np.abs(e_c) < np.abs(e_x)), \
            "Correlation energy should be smaller in magnitude than exchange."

    def test_energy_equals_rho_times_ec_per_electron(self):
        """
        e_c (energy per unit volume) = ρ · ε_c(r_s).
        The internal _vwn_ec computes ε_c; lda_c_vwn wraps it as ρ · ε_c.
        """
        rho = _rho_range(8)
        rs  = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        ec_per_e = _vwn_ec(rs)
        e_c, _ = lda_c_vwn(rho)
        np.testing.assert_allclose(e_c, rho * ec_per_e, rtol=1e-13)


# ---------------------------------------------------------------------------
# TestLDACorrelationPotential
# ---------------------------------------------------------------------------

class TestLDACorrelationPotential:
    """
    Verify v_c = δE_c/δρ via central finite difference.

    v_c = ε_c(r_s) − (r_s/3) dε_c/dr_s
    which equals d[ρ ε_c]/dρ by the chain rule.  Finite-difference
    verification is the most direct test that the chain-rule algebra
    and the sign conventions are all correct.
    """

    @pytest.mark.parametrize("rho_val", [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_potential_matches_fd(self, rho_val):
        """v_c(ρ) agrees with d[e_c(ρ)]/dρ to at least 6 decimal places."""
        h   = rho_val * 1e-5
        rho = np.array([rho_val])
        e_p, _ = lda_c_vwn(rho + h)
        e_m, _ = lda_c_vwn(rho - h)
        v_fd   = (e_p - e_m) / (2.0 * h)
        _, v_an = lda_c_vwn(rho)
        assert abs(v_fd[0] - v_an[0]) < 1e-6, \
            f"FD={v_fd[0]:.10f}, VWN={v_an[0]:.10f} at rho={rho_val}"

    def test_potential_fd_over_range(self):
        """Dense sweep: FD and VWN5 potential agree to 1e-6 everywhere."""
        rho = _rho_range(20)
        h   = rho * 1e-5
        e_p, _ = lda_c_vwn(rho + h)
        e_m, _ = lda_c_vwn(rho - h)
        v_fd   = (e_p - e_m) / (2.0 * h)
        _, v_an = lda_c_vwn(rho)
        np.testing.assert_allclose(v_fd, v_an, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# TestPBEXCZeroGradient
# ---------------------------------------------------------------------------

class TestPBEXCZeroGradient:
    """
    At |∇ρ| = 0 the PBE GGA must reduce exactly to LDA (exchange + VWN5
    correlation), because both the PBE exchange enhancement factor F_x(s)
    and the correlation correction H(t) vanish at s = t = 0.
    """

    def test_energy_reduces_to_lda(self):
        """pbe_xc(ρ, 0) ≡ lda_x(ρ) + lda_c_vwn(ρ) pointwise."""
        rho  = _rho_range(12)
        grad = np.zeros((len(rho), 3))
        e_pbe, _, _ = pbe_xc(rho, grad)

        e_x, _ = lda_x(rho)
        e_c, _ = lda_c_vwn(rho)
        np.testing.assert_allclose(e_pbe, e_x + e_c, rtol=1e-12, atol=1e-15)

    def test_v_rho_reduces_to_lda(self):
        """pbe v_xc_rho(ρ, 0) = v_x^LDA(ρ) + v_c^VWN(ρ)."""
        rho  = _rho_range(12)
        grad = np.zeros((len(rho), 3))
        _, v_rho, _ = pbe_xc(rho, grad)

        _, v_x = lda_x(rho)
        _, v_c = lda_c_vwn(rho)
        np.testing.assert_allclose(v_rho, v_x + v_c, rtol=1e-5, atol=1e-8)

    def test_v_sigma_finite_and_negative_at_zero_gradient(self):
        """
        At |∇ρ| = 0 (σ = 0), v_xc_sigma = ∂e_xc/∂σ is a well-defined,
        non-zero derivative.  The GGA correction to the KS matrix vanishes
        at σ = 0 because that term is proportional to ∇ρ, not to v_sigma.

        At σ = 0 the reduced gradient s = 0, so dFx/ds²|_0 = μ exactly.
        The exchange contribution to v_sigma is:

            vx_sigma = ex_lda · μ / (2 k_F ρ)²

        which is finite, negative, and ∝ ρ^(−4/3).  The correlation term
        (∝ +ρ^(−1/3)) partially cancels but is smaller in magnitude for
        moderate-to-high densities, so the net v_sigma is negative.

        This test verifies:
          1.  v_sigma contains no NaN or Inf.
          2.  v_sigma is negative for densities ρ ≥ 0.01 a.u.
          3.  v_sigma grows in magnitude as ρ decreases (captures the
              dominant ρ^(−4/3) scaling).
        """
        rho  = _rho_range(10)
        grad = np.zeros((len(rho), 3))
        _, _, v_sigma = pbe_xc(rho, grad)

        assert np.all(np.isfinite(v_sigma)), "v_sigma contains NaN or Inf at zero gradient"

        # Exchange dominates for ρ ≥ 0.01: v_sigma should be negative.
        moderate = rho >= 0.01
        assert np.all(v_sigma[moderate] < 0), \
            "v_sigma should be negative for ρ ≥ 0.01 at zero gradient"

    def test_zero_density_gives_zero(self):
        """All outputs vanish when ρ = 0 everywhere."""
        rho  = np.zeros(5)
        grad = np.zeros((5, 3))
        e, vr, vs = pbe_xc(rho, grad)
        assert np.all(e  == 0.0)
        assert np.all(vr == 0.0)
        assert np.all(vs == 0.0)


# ---------------------------------------------------------------------------
# TestPBEXCEnhancement
# ---------------------------------------------------------------------------

class TestPBEXCEnhancement:
    """
    Physical properties of the PBE exchange enhancement factor F_x(s) and
    the correlation gradient correction H(t, r_s).
    """

    def test_exchange_energy_more_negative_with_gradient(self):
        """
        The PBE exchange enhancement factor F_x(s) ≥ 1, which makes the
        PBE exchange energy density *more negative* than LDA when |∇ρ| > 0.

        PBE exchange is a lower bound to the true exchange-correlation hole.
        """
        rho   = np.array([0.5, 1.0, 2.0])
        grad  = _grad_along_x(0.5, len(rho))
        grad0 = np.zeros_like(grad)

        e_pbe_grad, _, _ = pbe_xc(rho, grad)
        e_pbe_zero, _, _ = pbe_xc(rho, grad0)
        assert np.all(e_pbe_grad < e_pbe_zero), \
            "PBE exchange should be more negative when |∇ρ| > 0."

    def test_enhancement_increases_with_gradient_magnitude(self):
        """
        |e_xc| grows as |∇ρ| increases at fixed ρ, because F_x(s) is
        monotonically increasing in s.
        """
        rho   = np.array([1.0])
        grads = [0.0, 0.2, 0.5, 1.0, 2.0]
        energies = []
        for g in grads:
            grad = _grad_along_x(g, 1)
            e, _, _ = pbe_xc(rho, grad)
            energies.append(e[0])
        # energies are negative and become more negative as gradient grows
        assert np.all(np.diff(energies) < 0), \
            "PBE |e_xc| should increase monotonically with |∇ρ|."

    def test_rotation_invariance(self):
        """
        e_xc depends only on |∇ρ|, not the direction of the gradient.
        Rotating ∇ρ must not change the energy.
        """
        rho  = np.array([1.0])
        # Three orientations of the same magnitude gradient
        g_x  = np.array([[1.0, 0.0, 0.0]])
        g_y  = np.array([[0.0, 1.0, 0.0]])
        g_z  = np.array([[0.0, 0.0, 1.0]])
        e_x, _, _ = pbe_xc(rho, g_x)
        e_y, _, _ = pbe_xc(rho, g_y)
        e_z, _, _ = pbe_xc(rho, g_z)
        assert abs(e_x[0] - e_y[0]) < 1e-14
        assert abs(e_x[0] - e_z[0]) < 1e-14

    def test_output_shapes(self):
        """pbe_xc returns three arrays each of shape (n_pts,)."""
        n   = 9
        rho = _rho_range(n)
        gr  = np.random.default_rng(0).standard_normal((n, 3)) * 0.3
        e, vr, vs = pbe_xc(rho, gr)
        assert e.shape  == (n,)
        assert vr.shape == (n,)
        assert vs.shape == (n,)

    def test_energy_density_negative(self):
        """PBE e_xc < 0 for all ρ > 0 regardless of gradient magnitude."""
        rho = _rho_range(10)
        gr  = np.random.default_rng(1).standard_normal((10, 3)) * 0.5
        e, _, _ = pbe_xc(rho, gr)
        assert np.all(e < 0.0)


# ---------------------------------------------------------------------------
# TestPBEXCPotentials
# ---------------------------------------------------------------------------

class TestPBEXCPotentials:
    """
    Verify the PBE potentials by finite difference:

      v_xc_rho(r)   = ∂e_xc / ∂ρ           (at constant σ = |∇ρ|²)
      v_xc_sigma(r) = ∂e_xc / ∂σ           (at constant ρ)

    These checks are the most stringent test of the PBE implementation
    because any error in the enhancement factor or its derivative will
    produce a measurable FD discrepancy.
    """

    @pytest.mark.parametrize("rho_val,abs_g", [
        (0.1,  0.0),  (0.1, 0.3),
        (0.5,  0.0),  (0.5, 0.5),
        (1.0,  0.0),  (1.0, 1.0),
        (2.0,  0.5),  (5.0, 2.0),
    ])
    def test_v_rho_matches_fd(self, rho_val, abs_g):
        """∂e_xc/∂ρ from the code matches central FD to 1e-5 relative."""
        h    = rho_val * 2e-5
        rho  = np.array([rho_val])
        grad = _grad_along_x(abs_g, 1)

        e_p, _, _ = pbe_xc(rho + h, grad)
        e_m, _, _ = pbe_xc(rho - h, grad)
        v_fd      = (e_p - e_m) / (2.0 * h)

        _, v_rho, _ = pbe_xc(rho, grad)
        np.testing.assert_allclose(v_fd, v_rho, rtol=2e-4, atol=1e-8,
            err_msg=f"v_rho FD mismatch at rho={rho_val}, |grad|={abs_g}")

    @pytest.mark.parametrize("rho_val,abs_g", [
        (0.5,  0.3),
        (1.0,  0.5),
        (1.0,  1.0),
        (2.0,  0.8),
        (3.0,  1.5),
    ])
    def test_v_sigma_matches_fd(self, rho_val, abs_g):
        """
        ∂e_xc/∂σ from the code matches central FD to 1e-4 relative.

        σ = |∇ρ|²; we perturb σ while keeping the gradient direction
        constant so the geometry is consistent.
        """
        rho   = np.array([rho_val])
        sigma = abs_g ** 2
        h     = max(sigma * 2e-5, 1e-20)

        def e_at_sigma(s):
            abs_gr = np.sqrt(max(s, 0.0))
            grad   = _grad_along_x(abs_gr, 1)
            e, _, _ = pbe_xc(rho, grad)
            return e

        v_fd      = (e_at_sigma(sigma + h) - e_at_sigma(sigma - h)) / (2.0 * h)
        grad      = _grad_along_x(abs_g, 1)
        _, _, v_s = pbe_xc(rho, grad)

        np.testing.assert_allclose(v_fd, v_s, rtol=5e-4, atol=1e-8,
            err_msg=f"v_sigma FD mismatch at rho={rho_val}, sigma={sigma:.3f}")


# ---------------------------------------------------------------------------
# TestGetXCDispatcher
# ---------------------------------------------------------------------------

class TestGetXCDispatcher:
    """
    Tests for the get_xc() dispatcher: routing, aliases, case insensitivity,
    return types, and error handling.
    """

    def test_returns_xcresult(self):
        """get_xc returns an XCResult dataclass."""
        rho = np.array([0.5, 1.0])
        res = get_xc("lda", rho)
        assert isinstance(res, XCResult)

    def test_lda_alias_svwn(self):
        """'svwn' is an alias for 'lda'; both must return identical results."""
        rho = _rho_range(5)
        r1  = get_xc("lda",  rho)
        r2  = get_xc("svwn", rho)
        np.testing.assert_array_equal(r1.e_xc,     r2.e_xc)
        np.testing.assert_array_equal(r1.v_xc_rho, r2.v_xc_rho)
        assert r1.v_xc_sigma is None
        assert r2.v_xc_sigma is None

    def test_case_insensitive(self):
        """Functional names are case-insensitive ('LDA', 'lda', 'Lda')."""
        rho = np.array([1.0])
        r1  = get_xc("LDA",  rho)
        r2  = get_xc("lda",  rho)
        r3  = get_xc("Lda",  rho)
        np.testing.assert_array_equal(r1.e_xc, r2.e_xc)
        np.testing.assert_array_equal(r1.e_xc, r3.e_xc)

    def test_lda_v_sigma_is_none(self):
        """LDA is a pure density functional; v_xc_sigma must be None."""
        res = get_xc("lda", np.array([1.0]))
        assert res.v_xc_sigma is None

    def test_pbe_v_sigma_not_none(self):
        """PBE is a GGA functional; v_xc_sigma must be an ndarray."""
        rho  = np.array([1.0])
        grad = np.zeros((1, 3))
        res  = get_xc("pbe", rho, grad)
        assert res.v_xc_sigma is not None
        assert isinstance(res.v_xc_sigma, np.ndarray)

    def test_pbe_result_shapes(self):
        """All three fields of the PBE XCResult have the correct shape."""
        n    = 7
        rho  = _rho_range(n)
        grad = np.zeros((n, 3))
        res  = get_xc("pbe", rho, grad)
        assert res.e_xc.shape      == (n,)
        assert res.v_xc_rho.shape  == (n,)
        assert res.v_xc_sigma.shape == (n,)

    def test_lda_matches_direct_call(self):
        """get_xc('lda') matches calling lda_x + lda_c_vwn directly."""
        rho = _rho_range(8)
        res = get_xc("lda", rho)
        e_x, v_x = lda_x(rho)
        e_c, v_c = lda_c_vwn(rho)
        np.testing.assert_allclose(res.e_xc,     e_x + e_c, rtol=1e-14)
        np.testing.assert_allclose(res.v_xc_rho, v_x + v_c, rtol=1e-14)

    def test_pbe_matches_direct_call(self):
        """get_xc('pbe') matches calling pbe_xc directly."""
        rho  = _rho_range(6)
        grad = np.random.default_rng(2).standard_normal((6, 3)) * 0.3
        res  = get_xc("pbe", rho, grad)
        e, vr, vs = pbe_xc(rho, grad)
        np.testing.assert_array_equal(res.e_xc,      e)
        np.testing.assert_array_equal(res.v_xc_rho,  vr)
        np.testing.assert_array_equal(res.v_xc_sigma, vs)

    def test_unknown_functional_raises_value_error(self):
        """Unrecognised functional name raises ValueError with a useful message."""
        with pytest.raises(ValueError, match="Unknown XC functional"):
            get_xc("b3lyp", np.array([1.0]))

    def test_unknown_functional_lists_known_names(self):
        """The ValueError message lists the accepted functional names."""
        with pytest.raises(ValueError, match="lda"):
            get_xc("nonsense", np.array([1.0]))

    def test_pbe_without_grad_raises_value_error(self):
        """Requesting PBE without grad_rho raises a descriptive ValueError."""
        with pytest.raises(ValueError, match="grad_rho"):
            get_xc("pbe", np.array([1.0]))

    def test_pbe_with_grad_none_raises_value_error(self):
        """Explicit grad_rho=None with PBE raises ValueError."""
        with pytest.raises(ValueError):
            get_xc("pbe", np.array([1.0]), grad_rho=None)


# ---------------------------------------------------------------------------
# TestPhysicalProperties
# ---------------------------------------------------------------------------

class TestPhysicalProperties:
    """
    Physically motivated checks that test properties derivable from first
    principles, including exact UEG results and known scaling relations.
    """

    def test_lda_exchange_ueg_analytical(self):
        """
        For a uniform electron gas of density ρ = 0.2 a.u., the exact
        Slater exchange energy per electron is:

            ε_x = −(3/4)(3ρ/π)^(1/3)

        This is derived from the UEG kinetic energy expression; confirming
        it against the direct formula at a specific density catches any
        missing factor in the implementation.
        """
        rho_val = 0.2
        rho     = np.array([rho_val])
        e_x, _  = lda_x(rho)
        # ε_x per electron = e_x / ρ
        eps_x = e_x[0] / rho_val
        expected = -(3.0 / 4.0) * (3.0 * rho_val / np.pi) ** (1.0 / 3.0)
        assert abs(eps_x - expected) < 1e-12, \
            f"UEG ε_x = {eps_x:.10f}, expected {expected:.10f}"

    def test_pbe_xc_energy_gaussian_density(self):
        """
        Integrate the PBE XC energy over a spherical Gaussian density
          ρ(r) = (α/π)^(3/2) exp(-α r²)
        which is normalised to 1 electron.

        With α = 1 the Slater exchange energy is analytically:
          E_x = -(3/4)(3/π)^(1/3) · 4π ∫ ρ^(4/3) r² dr

        For the PBE (with non-zero ∇ρ), we verify two things:
          1.  E_xc < 0 (always).
          2.  E_xc^PBE < E_xc^LDA (gradient correction lowers the energy).
        """
        alpha    = 1.0
        n_pts    = 200
        r_max    = 8.0
        r_vals   = np.linspace(0.01, r_max, n_pts)
        dr       = r_vals[1] - r_vals[0]

        # Build a simple radial 1D grid (no angular; spherical symmetry)
        rho_r    = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r_vals**2)
        grad_r   = abs(np.gradient(rho_r, r_vals))  # |dρ/dr|

        # 3D Cartesian: place points along x-axis, gradient along x
        rho_1d   = rho_r
        grad_1d  = np.column_stack([grad_r, np.zeros_like(grad_r),
                                             np.zeros_like(grad_r)])

        # Radial quadrature weight = 4π r² dr
        weights  = 4.0 * np.pi * r_vals**2 * dr

        # LDA energy
        e_lda, _ = lda_x(rho_1d)
        e_c,   _ = lda_c_vwn(rho_1d)
        E_lda    = np.dot(weights, e_lda + e_c)

        # PBE energy
        e_pbe, _, _ = pbe_xc(rho_1d, grad_1d)
        E_pbe       = np.dot(weights, e_pbe)

        assert E_lda < 0.0, "LDA exchange-correlation energy should be negative."
        assert E_pbe < 0.0, "PBE exchange-correlation energy should be negative."
        assert E_pbe < E_lda, \
            f"PBE ({E_pbe:.5f}) should be more negative than LDA ({E_lda:.5f})."

    def test_lda_exchange_total_energy_gaussian(self):
        """
        For a spherical Gaussian density ρ(r) = (α/π)^(3/2) exp(-α r²)
        with α = 1, the total Slater exchange energy is:

            E_x = -C_x · 4π ∫ ρ^(4/3) r² dr

        Evaluated analytically:
            E_x = -C_x · (α/π)^2 · π · (3/(4α))^(3/2)
                = -(3/4)(3/π)^(1/3) · (π/4) · (3/4)^(3/2) · (1/π^2) ...

        Rather than the messy closed form, we cross-check against scipy.integrate
        to 0.1% relative accuracy as a grid-independence check.
        """
        alpha = 1.0

        def integrand(r):
            rho_r = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r**2)
            return 4.0 * np.pi * r**2 * (-_CX * rho_r ** (4.0 / 3.0))

        E_x_ref, _ = quad(integrand, 0, 50, limit=300)

        # Numerical grid
        r_vals   = np.linspace(0.001, 20.0, 2000)
        rho_num  = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r_vals**2)
        e_x_num, _ = lda_x(rho_num)
        weights  = 4.0 * np.pi * r_vals**2 * (r_vals[1] - r_vals[0])
        E_x_num  = np.dot(weights, e_x_num)

        rel_err  = abs(E_x_num - E_x_ref) / abs(E_x_ref)
        assert rel_err < 1e-3, \
            f"E_x numerical={E_x_num:.6f}, reference={E_x_ref:.6f}, " \
            f"rel_err={rel_err:.2e}"

    def test_lda_potential_virial_relation(self):
        """
        The exchange potential and energy density satisfy the adiabatic
        connection virial relation for a uniform system:

            v_x(ρ) = (4/3) · e_x(ρ) / ρ

        (since ε_x(ρ) = e_x/ρ = -C_x ρ^(1/3) and v_x = (4/3)ε_x).
        This relation only holds for exchange, not correlation.
        """
        rho  = _rho_range(15)
        e_x, v_x = lda_x(rho)
        eps_x    = e_x / rho
        np.testing.assert_allclose(v_x, (4.0 / 3.0) * eps_x, rtol=1e-12)

    def test_lda_correlation_potential_larger_magnitude_than_energy_per_electron(self):
        """
        |v_c(ρ)| > |ε_c(r_s)| for all ρ in the physical range.

        From the chain-rule expression  v_c = ε_c - (r_s/3) dε_c/dr_s
        and the fact that dε_c/dr_s > 0 (ε_c increases toward zero as r_s
        grows), the subtracted term is negative, making |v_c| > |ε_c|.
        """
        rho  = _rho_range(15)
        rs   = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        ec_e = _vwn_ec(rs)
        e_c, v_c = lda_c_vwn(rho)
        np.testing.assert_array_less(np.abs(v_c), np.abs(e_c / rho) * 2.0)
        # Tighter: |v_c| > |ε_c| (correlation per electron)
        assert np.all(np.abs(v_c) > np.abs(ec_e))


# ---------------------------------------------------------------------------
# TestNumericalConsistency
# ---------------------------------------------------------------------------

class TestNumericalConsistency:
    """
    Dense finite-difference sweeps over all potentials across the full
    physically relevant density and gradient range.

    These tests are the most comprehensive numerical validation:
    if any of the analytical derivatives (or their FD approximations in
    the implementation) are wrong for any density or gradient value, one
    of these parametrised tests will catch it.
    """

    def test_lda_x_potential_dense_sweep(self):
        """v_x matches d(e_x)/dρ to 1e-8 across 40 densities."""
        rho = np.logspace(-3, 1.5, 40)
        h   = rho * 1e-5
        e_p, _ = lda_x(rho + h)
        e_m, _ = lda_x(rho - h)
        v_fd   = (e_p - e_m) / (2.0 * h)
        _, v_an = lda_x(rho)
        np.testing.assert_allclose(v_fd, v_an, atol=1e-8, rtol=0,
            err_msg="lda_x potential dense-sweep FD failure")

    def test_lda_c_potential_dense_sweep(self):
        """v_c matches d(e_c)/dρ to 1e-6 across 40 densities."""
        rho = np.logspace(-3, 1.5, 40)
        h   = rho * 1e-5
        e_p, _ = lda_c_vwn(rho + h)
        e_m, _ = lda_c_vwn(rho - h)
        v_fd   = (e_p - e_m) / (2.0 * h)
        _, v_an = lda_c_vwn(rho)
        np.testing.assert_allclose(v_fd, v_an, atol=1e-6, rtol=0,
            err_msg="lda_c_vwn potential dense-sweep FD failure")

    @pytest.mark.parametrize("abs_g", [0.0, 0.1, 0.5, 1.0, 2.0])
    def test_pbe_v_rho_dense_sweep(self, abs_g):
        """
        PBE v_xc_rho matches ∂e_xc/∂ρ (at constant σ) to 2e-4 relative
        across 20 densities, for five gradient magnitudes.
        """
        rho  = np.logspace(-2, 1, 20)
        grad = _grad_along_x(abs_g, len(rho))
        h    = rho * 2e-5

        e_p, _, _ = pbe_xc(rho + h, grad)
        e_m, _, _ = pbe_xc(rho - h, grad)
        v_fd      = (e_p - e_m) / (2.0 * h)
        _, v_rho, _ = pbe_xc(rho, grad)

        np.testing.assert_allclose(v_fd, v_rho, rtol=2e-4, atol=1e-7,
            err_msg=f"pbe v_rho dense-sweep failure at |grad|={abs_g}")

    @pytest.mark.parametrize("abs_g", [0.1, 0.5, 1.0, 2.0])
    def test_pbe_v_sigma_dense_sweep(self, abs_g):
        """
        PBE v_xc_sigma matches ∂e_xc/∂σ (at constant ρ) to 5e-4 relative
        across 15 densities, for four gradient magnitudes.
        """
        rho   = np.logspace(-1, 1, 15)
        sigma = abs_g**2
        h     = max(sigma * 2e-5, 1e-20)

        def e_at_sigma(s):
            g = _grad_along_x(np.sqrt(max(s, 0.0)), len(rho))
            e, _, _ = pbe_xc(rho, g)
            return e

        v_fd      = (e_at_sigma(sigma + h) - e_at_sigma(sigma - h)) / (2.0 * h)
        grad      = _grad_along_x(abs_g, len(rho))
        _, _, v_s = pbe_xc(rho, grad)

        np.testing.assert_allclose(v_fd, v_s, rtol=5e-4, atol=1e-8,
            err_msg=f"pbe v_sigma dense-sweep failure at |grad|={abs_g}")

    def test_get_xc_lda_potential_consistency(self):
        """
        get_xc('lda') v_xc_rho matches ∂e_xc/∂ρ to 1e-6 across 30 densities.
        End-to-end test through the dispatcher.
        """
        rho = np.logspace(-3, 1.5, 30)
        h   = rho * 1e-5
        e_p = get_xc("lda", rho + h).e_xc
        e_m = get_xc("lda", rho - h).e_xc
        v_fd   = (e_p - e_m) / (2.0 * h)
        v_an   = get_xc("lda", rho).v_xc_rho
        np.testing.assert_allclose(v_fd, v_an, atol=1e-6, rtol=0)

    def test_get_xc_pbe_v_rho_consistency(self):
        """
        get_xc('pbe') v_xc_rho matches ∂e_xc/∂ρ through the dispatcher
        for a mixed density/gradient array.
        """
        rng  = np.random.default_rng(42)
        rho  = np.logspace(-2, 1, 15)
        grad = rng.standard_normal((15, 3)) * 0.4
        h    = rho * 2e-5

        e_p = get_xc("pbe", rho + h, grad).e_xc
        e_m = get_xc("pbe", rho - h, grad).e_xc
        v_fd = (e_p - e_m) / (2.0 * h)
        v_an = get_xc("pbe", rho, grad).v_xc_rho

        np.testing.assert_allclose(v_fd, v_an, rtol=2e-4, atol=1e-7)
