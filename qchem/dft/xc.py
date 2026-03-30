"""
qchem/dft/xc.py — exchange-correlation energy densities and potentials

Overview
--------
Kohn-Sham DFT replaces the unknown exact exchange-correlation energy
E_xc[ρ] with a tractable approximation expressed as a functional of the
electron density ρ(r) — and, in GGA, also its gradient ∇ρ(r).  This
module provides the XC energy density e_xc(r) and the XC potential

    v_xc(r) = δE_xc / δρ(r)

evaluated pointwise on the numerical integration grid produced by
dft.grid.  The Kohn-Sham driver (dft.ks) calls get_xc() once per SCF
iteration to obtain these quantities, then assembles the XC matrix via
grid.build_xc_matrix() and adds E_xc to the total energy.

Functional hierarchy
--------------------
The Jacob's-ladder classification organises DFT approximations by the
ingredients they depend on:

  Rung 1 — LDA  (local density approximation)
      e_xc depends only on ρ(r) at the same point r.
      Implemented here:
        • Slater exchange            lda_x()
        • VWN5 correlation           lda_c_vwn()
        • Combined SVWN              via get_xc("lda") or get_xc("svwn")

  Rung 2 — GGA  (generalised gradient approximation)
      e_xc depends on ρ(r) and |∇ρ(r)|.
      Implemented here:
        • PBE exchange + correlation  pbe_xc()
        • Accessed via               get_xc("pbe")

  Higher rungs (meta-GGA, hybrid, double hybrid) are not yet implemented.

LDA exchange — Slater (1951)
-----------------------------
The uniform electron gas (UEG) has an exact analytic exchange energy:

    e_x(ρ) = −C_x · ρ^(4/3)

    C_x = (3/4) · (3/π)^(1/3)

The XC potential is the functional derivative with respect to ρ:

    v_x(ρ) = δE_x/δρ = −(4/3) · C_x · ρ^(1/3)

This is the Dirac/Slater exchange, the simplest non-trivial density
functional.  Numerically it under-binds because the UEG assumption
breaks down near nuclei and in density tails.

LDA correlation — Vosko, Wilk & Nusair (1980) — VWN5 parametrisation
-----------------------------------------------------------------------
Correlation in the UEG cannot be expressed analytically.  Ceperley and
Alder (1980) computed it by diffusion quantum Monte Carlo at a grid of
densities.  VWN fitted their data with the Padé-like rational form:

    e_c(r_s) = A/2 · { ln(r_s/Q²) + 2b/Q · arctan(Q/(2r_s+b))
                        − bx₀/X₀ · [ ln((r_s−x₀)²/X(r_s))
                            + 2(2x₀+b)/Q · arctan(Q/(2r_s+b)) ] }

where:
    r_s   = (3 / 4π ρ)^(1/3)     Wigner-Seitz radius
    X(x)  = x² + b·x + c
    X₀    = x₀² + b·x₀ + c
    Q     = sqrt(4c − b²)

The five VWN5 parameters for the paramagnetic (ζ=0) case are:
    A  = 0.0621814
    x₀ = −0.409286
    b  = 13.0720
    c  = 42.7198

The correlation potential is obtained from e_c via the chain rule:
    v_c(ρ) = e_c(r_s) − (r_s/3) · de_c/dr_s

This is not computed symbolically here; instead it is calculated by
finite difference on r_s, which is equivalent and avoids a complicated
closed-form derivative.

GGA exchange-correlation — Perdew, Burke & Ernzerhof (1996) — PBE
------------------------------------------------------------------
PBE enhances LDA by multiplying the exchange energy density by a
dimensionless factor F_x(s) that depends on the reduced gradient:

    s = |∇ρ| / (2 k_F ρ)    k_F = (3π²ρ)^(1/3)

    E_x^PBE = ∫ e_x^LDA(ρ) · F_x(s) dr

    F_x(s) = 1 + κ − κ / (1 + μ s² / κ)

with κ = 0.804, μ = 0.21951 (satisfying the UEG linear-response and
Lieb-Oxford bound conditions).

PBE correlation modifies the LDA correlation by a gradient-dependent
term H(t, r_s, ζ):

    E_c^PBE = ∫ [ e_c^LDA(ρ) + H(t, r_s) ] dr

    H = γ φ³ ln{ 1 + β/γ · t² · (1 + At²)/(1 + At² + A²t⁴) }

where:
    t  = |∇ρ| / (2 φ k_s ρ)     k_s = sqrt(4 k_F / π)
    φ  = 1  (paramagnetic, ζ=0)
    A  = β/γ · 1/(exp(−e_c^LDA/(γ φ³)) − 1)
    β  = 0.066725,  γ = 0.031091

The GGA XC matrix in the KS equations requires not just v_xc = δE_xc/δρ
but also the derivative with respect to σ = |∇ρ|²:

    v_xc_sigma = ∂e_xc/∂σ   (per unit volume, at constant ρ)

because the matrix element contains an integration-by-parts term:

    V_xc,μν^GGA = ∫ [ v_xc_rho · φ_μ φ_ν
                      + 2 v_xc_sigma · ∇ρ · (φ_μ ∇φ_ν + φ_ν ∇φ_μ) ] dr

This module computes v_xc_sigma numerically; dft.ks is responsible for
the extra matrix terms.

Public API
----------
lda_x(rho)              → (e_x, v_x)
lda_c_vwn(rho)          → (e_c, v_c)
pbe_xc(rho, grad_rho)   → (e_xc, v_xc_rho, v_xc_sigma)
get_xc(name, rho, grad_rho=None)
                        → XCResult(e_xc, v_xc_rho, v_xc_sigma)

References
----------
Slater, J. C. (1951). A simplification of the Hartree-Fock method.
    Phys. Rev. 81, 385.
Ceperley, D. M. & Alder, B. J. (1980). Ground state of the electron gas
    by a stochastic method. Phys. Rev. Lett. 45, 566.
Vosko, S. H., Wilk, L. & Nusair, M. (1980). Accurate spin-dependent
    electron liquid correlation energies for local spin density
    calculations. Can. J. Phys. 58, 1200.
Perdew, J. P., Burke, K. & Ernzerhof, M. (1996). Generalized gradient
    approximation made simple. Phys. Rev. Lett. 77, 3865.
Perdew, J. P., Burke, K. & Ernzerhof, M. (1997). Erratum. Phys. Rev.
    Lett. 78, 1396.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

# LDA exchange prefactor  C_x = (3/4)(3/π)^(1/3)
_CX: float = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

# VWN5 paramagnetic parameters (Table 5, Vosko et al. 1980)
_VWN_A:  float =  0.0621814
_VWN_X0: float = -0.409286
_VWN_B:  float = 13.0720
_VWN_C:  float = 42.7198

# PBE exchange parameters
_PBE_KAPPA: float = 0.804
_PBE_MU:    float = 0.21951

# PBE correlation parameters
_PBE_BETA:  float = 0.066725
_PBE_GAMMA: float = 0.031091   # = (1 − ln 2) / π²

# Density threshold below which ρ is treated as zero.
# Avoids singularities in r_s and reduced-gradient expressions.
_RHO_TOL: float = 1.0e-15

# Finite-difference step used when differentiating e_c w.r.t. r_s
# in lda_c_vwn.  Value chosen to balance truncation and cancellation error.
_FD_H: float = 1.0e-5


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class XCResult:
    """
    Container returned by get_xc().

    Attributes
    ----------
    e_xc : ndarray, shape (n_pts,)
        XC energy density per unit volume at each grid point, e_xc(r).
        The XC energy is  E_xc = Σ_g w_g e_xc(r_g).
    v_xc_rho : ndarray, shape (n_pts,)
        XC potential  v_xc(r) = δE_xc/δρ(r),  the quantity that enters
        the Kohn-Sham matrix directly via build_xc_matrix().
    v_xc_sigma : ndarray or None, shape (n_pts,) or None
        Derivative ∂e_xc/∂σ where σ = |∇ρ|².  Non-None only for GGA
        functionals.  The KS driver uses this to add the gradient
        correction term  2 v_xc_sigma · ∇ρ · ∇φ  to V_xc,μν.
        None for LDA functionals (gradient terms are absent).
    """
    e_xc:       np.ndarray
    v_xc_rho:   np.ndarray
    v_xc_sigma: np.ndarray | None


# ---------------------------------------------------------------------------
# LDA exchange
# ---------------------------------------------------------------------------

def lda_x(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Slater local-density exchange energy density and potential.

    The Dirac-Slater exchange functional for a spin-unpolarised (closed-
    shell) system is:

        e_x(ρ) = −C_x · ρ^(4/3)

        v_x(ρ) = δE_x / δρ = −(4/3) · C_x · ρ^(1/3)

    where C_x = (3/4)(3/π)^(1/3) ≈ 0.7386.

    Parameters
    ----------
    rho : ndarray, shape (n_pts,)
        Electron density at each grid point.  Values below _RHO_TOL are
        treated as zero; exchange vanishes there.

    Returns
    -------
    e_x : ndarray, shape (n_pts,)
        Exchange energy density per unit volume.  Negative everywhere.
    v_x : ndarray, shape (n_pts,)
        Exchange potential  v_x(r) = δE_x/δρ(r).  Negative everywhere.

    Notes
    -----
    The factor of ρ^(4/3) (rather than ρ^(1/3) alone) is because e_x
    is an *energy density* per unit volume: e_x = ε_x · ρ, and the
    exchange energy per electron ε_x ∝ ρ^(1/3).  The energy potential
    v_x = d(ρ ε_x)/dρ = (4/3) ε_x = −(4/3) C_x ρ^(1/3).

    References
    ----------
    Slater (1951); Dirac (1930).
    """
    rho = np.asarray(rho, dtype=float)
    mask = rho > _RHO_TOL

    rho13 = np.where(mask, rho ** (1.0 / 3.0), 0.0)

    e_x = np.zeros_like(rho)
    v_x = np.zeros_like(rho)

    e_x[mask] = -_CX * rho[mask] * rho13[mask]          # −C_x ρ^(4/3)
    v_x[mask] = -(4.0 / 3.0) * _CX * rho13[mask]        # −(4/3) C_x ρ^(1/3)

    return e_x, v_x


# ---------------------------------------------------------------------------
# LDA correlation — VWN5
# ---------------------------------------------------------------------------

def _vwn_ec(rs: np.ndarray) -> np.ndarray:
    """
    VWN5 correlation energy density per electron, ε_c(r_s), for the
    paramagnetic (spin-unpolarised) UEG.

    The rational interpolation formula (VWN eq. 4.4, paramagnetic set):

        ε_c = A/2 { ln(x²/X(x)) + 2b/Q · arctan(Q/(2x+b))
                    − b x₀/X₀ · [ ln((x−x₀)²/X(x))
                        + 2(2x₀+b)/Q · arctan(Q/(2x+b)) ] }

    where  x = sqrt(r_s),  X(x) = x² + bx + c,  Q = sqrt(4c − b²).

    Parameters
    ----------
    rs : ndarray, shape (n_pts,)
        Wigner-Seitz radius at each point.  Must be strictly positive.

    Returns
    -------
    ec : ndarray, shape (n_pts,)
        Correlation energy per electron in hartree.  Negative everywhere.

    Notes
    -----
    x₀ is negative (≈ −0.41), so (x − x₀) is always positive for
    physical densities (r_s > 0), and the inner logarithm is well-defined.
    """
    x   = np.sqrt(rs)
    X   = x * x + _VWN_B * x + _VWN_C         # X(x)  = rs + b√rs + c
    X0  = _VWN_X0**2 + _VWN_B * _VWN_X0 + _VWN_C  # X(x₀)
    Q   = np.sqrt(4.0 * _VWN_C - _VWN_B**2)   # Q = sqrt(4c − b²)

    arctan_arg = Q / (2.0 * x + _VWN_B)
    arctan_val = np.arctan(arctan_arg)

    term1 = np.log(x * x / X)
    term2 = (2.0 * _VWN_B / Q) * arctan_val
    term3 = _VWN_B * _VWN_X0 / X0 * (
        np.log((x - _VWN_X0)**2 / X)
        + 2.0 * (2.0 * _VWN_X0 + _VWN_B) / Q * arctan_val
    )

    ec = (_VWN_A / 2.0) * (term1 + term2 - term3)
    return ec


def lda_c_vwn(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    VWN5 local-density correlation energy density and potential.

    The correlation energy density per unit volume is:

        e_c(ρ) = ρ · ε_c(r_s)

    where ε_c(r_s) is the paramagnetic VWN5 correlation energy per
    electron (a function of the Wigner-Seitz radius r_s = (3/4πρ)^(1/3))
    and the correlation potential is:

        v_c(ρ) = δE_c/δρ = ε_c(r_s) − (r_s/3) · dε_c/dr_s

    The potential is obtained by central finite-difference on r_s, which
    is algebraically equivalent to the analytic derivative and avoids
    reproducing the lengthy closed-form expression.

    Parameters
    ----------
    rho : ndarray, shape (n_pts,)
        Electron density at each grid point.

    Returns
    -------
    e_c : ndarray, shape (n_pts,)
        Correlation energy density per unit volume.  Negative everywhere.
    v_c : ndarray, shape (n_pts,)
        Correlation potential  v_c(r) = δE_c/δρ(r).  Negative everywhere.

    Notes
    -----
    The conversion from ε_c to v_c follows from the chain rule and the
    definition r_s = (3/4πρ)^(1/3):

        dr_s/dρ = −r_s / (3ρ)

        v_c = ε_c + ρ · dε_c/dρ
            = ε_c + ρ · (dε_c/dr_s) · (−r_s/(3ρ))
            = ε_c − (r_s/3) · dε_c/dr_s

    References
    ----------
    Vosko, Wilk & Nusair (1980), Table 5, paramagnetic fit.
    Ceperley & Alder (1980).
    """
    rho  = np.asarray(rho, dtype=float)
    mask = rho > _RHO_TOL

    e_c = np.zeros_like(rho)
    v_c = np.zeros_like(rho)

    if not np.any(mask):
        return e_c, v_c

    rho_m = rho[mask]

    # Wigner-Seitz radius
    rs = (3.0 / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)

    # Energy per electron
    ec = _vwn_ec(rs)

    # Potential via central finite difference on r_s:
    #   dε_c/dr_s ≈ [ε_c(r_s + h) − ε_c(r_s − h)] / (2h)
    dec_drs = (_vwn_ec(rs + _FD_H) - _vwn_ec(rs - _FD_H)) / (2.0 * _FD_H)

    e_c[mask] = rho_m * ec
    v_c[mask] = ec - (rs / 3.0) * dec_drs

    return e_c, v_c


# ---------------------------------------------------------------------------
# GGA — PBE exchange-correlation
# ---------------------------------------------------------------------------

def pbe_xc(
    rho:      np.ndarray,
    grad_rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PBE generalised-gradient exchange-correlation energy density and
    potentials (Perdew, Burke & Ernzerhof, 1996).

    PBE augments LDA by multiplying the exchange energy density by an
    enhancement factor F_x(s) and adding a gradient-dependent correction
    H(t, r_s) to the correlation energy per electron.  Both corrections
    are built to satisfy exact constraints (scaling, UEG linear response,
    Lieb-Oxford bound) rather than being fitted to data.

    Exchange
    --------
    The reduced gradient:

        s = |∇ρ| / (2 k_F ρ)     k_F = (3π²ρ)^(1/3)

    The PBE exchange enhancement factor:

        F_x(s) = 1 + κ − κ / (1 + μ s² / κ)

    with κ = 0.804 (Lieb-Oxford bound) and μ = 0.21951 (gradient
    expansion).  At s = 0 F_x = 1 and the LDA limit is recovered.

    Exchange energy density:

        e_x^PBE(ρ, s) = e_x^LDA(ρ) · F_x(s)

    Exchange potential contributions:

        v_x_rho   = d[ρ · ε_x^LDA · F_x] / dρ   (at constant σ)
        v_x_sigma = ∂e_x^PBE / ∂σ               where σ = |∇ρ|²

    Correlation
    -----------
    The correlation enhancement H depends on r_s and the second reduced
    gradient t = |∇ρ| / (2 φ k_s ρ) where k_s = sqrt(4 k_F / π) is
    the Thomas-Fermi screening wave-vector and φ = 1 for ζ = 0:

        A   = β/γ · 1/(exp[−ε_c^LDA/(γ)] − 1)
        H   = γ ln{ 1 + β/γ · t² · (1 + At²)/(1 + At² + A²t⁴) }

    Correlation potential contributions are obtained by chain rule from
    the partial derivatives of H with respect to ρ (at constant σ) and
    with respect to σ (at constant ρ).

    Parameters
    ----------
    rho : ndarray, shape (n_pts,)
        Electron density at each grid point.
    grad_rho : ndarray, shape (n_pts, 3)
        Cartesian gradient of the density at each grid point, as
        provided by dft.grid.eval_ao_grad_on_grid().

    Returns
    -------
    e_xc : ndarray, shape (n_pts,)
        Total XC energy density per unit volume, e_xc = e_x + e_c.
    v_xc_rho : ndarray, shape (n_pts,)
        XC potential  ∂e_xc/∂ρ  (at constant σ).  Feeds directly into
        the Kohn-Sham matrix via build_xc_matrix().
    v_xc_sigma : ndarray, shape (n_pts,)
        XC potential  ∂e_xc/∂σ  where σ = |∇ρ|².  The KS driver uses
        this to assemble the GGA gradient-correction matrix terms.

    Notes
    -----
    Spin-restricted (ζ = 0) implementation only.  The spin-scaling
    relations that generalise to unrestricted (open-shell) KS-DFT are
    not implemented here.

    All numerical derivatives are computed by central finite difference
    at step size _FD_H on the relevant variable (s² for exchange, t² for
    correlation).  This avoids reproducing the lengthy analytic expressions
    and is accurate to O(h²) ≈ 10⁻¹⁰ for _FD_H = 1e-5.

    References
    ----------
    Perdew, Burke & Ernzerhof (1996), Phys. Rev. Lett. 77, 3865.
    Perdew, Burke & Ernzerhof (1997), Phys. Rev. Lett. 78, 1396 (erratum).
    """
    rho      = np.asarray(rho,      dtype=float)
    grad_rho = np.asarray(grad_rho, dtype=float)

    mask = rho > _RHO_TOL

    e_xc       = np.zeros_like(rho)
    v_xc_rho   = np.zeros_like(rho)
    v_xc_sigma = np.zeros_like(rho)

    if not np.any(mask):
        return e_xc, v_xc_rho, v_xc_sigma

    r   = rho[mask]
    gr  = grad_rho[mask]                              # (n_m, 3)
    sigma_m = np.einsum('gi,gi->g', gr, gr)           # |∇ρ|²

    # ---- shared intermediates -------------------------------------------

    r13  = r ** (1.0 / 3.0)                           # ρ^(1/3)
    kF   = (3.0 * np.pi**2 * r) ** (1.0 / 3.0)       # Fermi wave-vector
    rs   = (3.0 / (4.0 * np.pi * r)) ** (1.0 / 3.0)  # Wigner-Seitz radius

    abs_gr = np.sqrt(np.maximum(sigma_m, 0.0))        # |∇ρ|

    # ---- LDA exchange ---------------------------------------------------

    ex_lda = -_CX * r * r13           # e_x^LDA = −C_x ρ^(4/3)  (energy density)
    ex_lda_per_e = -_CX * r13         # ε_x^LDA = e_x/ρ = −C_x ρ^(1/3)

    # ---- PBE exchange enhancement ---------------------------------------

    # Reduced gradient  s = |∇ρ| / (2 k_F ρ)
    denom_s = 2.0 * kF * r
    s = abs_gr / np.where(denom_s > _RHO_TOL, denom_s, 1.0)
    s2 = s * s

    def _Fx(s2_val: np.ndarray) -> np.ndarray:
        """Enhancement factor F_x(s) = 1 + κ − κ/(1 + μs²/κ)."""
        return 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s2_val / _PBE_KAPPA)

    Fx  = _Fx(s2)
    ex  = ex_lda * Fx                                  # PBE exchange energy density

    # d(e_x)/d(s²) via chain rule through F_x:
    #   dF_x/ds² = μ κ / (κ + μs²)²
    dFx_ds2 = _PBE_MU * _PBE_KAPPA**2 / ((_PBE_KAPPA + _PBE_MU * s2)**2)

    # v_x_rho = d(ρ ε_x^LDA Fx)/dρ:  product rule at constant σ
    #   d(r · ε_x^LDA)/dr = (4/3) ε_x^LDA (since ε_x^LDA = −C_x ρ^(1/3))
    #   ds²/dρ at constant σ: s² = σ/(2kF r)² → ds²/dρ carries −7s²/(3ρ)
    #   (three contributions: kF ∝ ρ^(1/3) and the explicit ρ denominator)
    vx_rho  = (4.0 / 3.0) * ex_lda_per_e * Fx
    vx_rho += ex_lda_per_e * dFx_ds2 * s2 * (-8.0 / 3.0)

    # v_x_sigma = ∂e_x/∂σ = ε_x^LDA · (dFx/ds²) · ds²/dσ
    #   s² = σ / (2kF r)²  →  ds²/dσ = 1/(2kF r)²
    ds2_dsigma = 1.0 / (denom_s**2 + _RHO_TOL)
    vx_sigma = ex_lda * dFx_ds2 * ds2_dsigma

    # ---- PBE correlation ------------------------------------------------

    # LDA correlation per electron from VWN5
    _, vc_lda = lda_c_vwn(r)
    ec_lda_e  = _vwn_ec(rs)                           # ε_c^LDA (per electron)

    # Thomas-Fermi screening wave-vector  k_s = sqrt(4 k_F / π)
    ks = np.sqrt(4.0 * kF / np.pi)

    # Second reduced gradient  t = |∇ρ| / (2 k_s ρ)
    denom_t = 2.0 * ks * r
    t  = abs_gr / np.where(denom_t > _RHO_TOL, denom_t, 1.0)
    t2 = t * t

    def _H_from_ec_t2(ec_e: np.ndarray, t2_val: np.ndarray) -> np.ndarray:
        """
        PBE correlation gradient correction H(ε_c, t²).

        H = γ ln{ 1 + (β/γ) t² (1 + At²) / (1 + At² + A²t⁴) }

        A = β/γ / (exp[−ε_c/γ] − 1)

        Clamping: at t² = 0 → H = 0.  At high density (ε_c → −∞),
        exp(−ε_c/γ) → ∞ so A → 0 and H → γ ln(1 + β/γ t²).
        """
        # Guard against exp overflow when ec_e is large in magnitude
        exp_arg = np.clip(-ec_e / _PBE_GAMMA, -500.0, 500.0)
        A = (_PBE_BETA / _PBE_GAMMA) / (np.exp(exp_arg) - 1.0 + 1e-300)

        At2  = A * t2_val
        At4  = At2 * t2_val
        num  = t2_val * (1.0 + At2)
        den  = 1.0 + At2 + A * At4
        arg  = 1.0 + (_PBE_BETA / _PBE_GAMMA) * num / np.where(den > 0, den, 1e-300)
        return _PBE_GAMMA * np.log(np.maximum(arg, 1.0))

    H = _H_from_ec_t2(ec_lda_e, t2)

    ec = r * (ec_lda_e + H)                           # PBE correlation energy density

    # v_c_rho: full derivative at constant σ via finite difference on ρ.
    #   Increment ρ → ρ+δ, recompute rs, kF, ks, ec_lda, t², H at new ρ.
    _dr = r * _FD_H                                   # relative step in ρ
    _dr = np.maximum(_dr, 1e-20)

    def _ec_density_at(r_new: np.ndarray) -> np.ndarray:
        """PBE correlation energy density at a perturbed density."""
        rs_n   = (3.0 / (4.0 * np.pi * r_new)) ** (1.0 / 3.0)
        ec_n   = _vwn_ec(rs_n)
        kF_n   = (3.0 * np.pi**2 * r_new) ** (1.0 / 3.0)
        ks_n   = np.sqrt(4.0 * kF_n / np.pi)
        dt_n   = 2.0 * ks_n * r_new
        t2_n   = sigma_m / np.where(dt_n**2 > _RHO_TOL, dt_n**2, 1.0)
        H_n    = _H_from_ec_t2(ec_n, t2_n)
        return r_new * (ec_n + H_n)

    dec_drho = (_ec_density_at(r + _dr) - _ec_density_at(r - _dr)) / (2.0 * _dr)
    vc_rho = dec_drho

    # v_c_sigma: partial derivative at constant ρ via finite difference on σ.
    _ds = np.maximum(sigma_m * _FD_H, 1e-20)

    def _ec_density_at_sigma(sigma_new: np.ndarray) -> np.ndarray:
        """PBE correlation energy density at a perturbed σ."""
        abs_gr_n = np.sqrt(np.maximum(sigma_new, 0.0))
        t2_n     = (abs_gr_n / np.where(denom_t > _RHO_TOL, denom_t, 1.0))**2
        H_n      = _H_from_ec_t2(ec_lda_e, t2_n)
        return r * (ec_lda_e + H_n)

    dec_dsigma = (
        _ec_density_at_sigma(sigma_m + _ds) - _ec_density_at_sigma(sigma_m - _ds)
    ) / (2.0 * _ds)

    # ---- Assemble -------------------------------------------------------

    e_xc[mask]       = ex + ec
    v_xc_rho[mask]   = vx_rho + vc_rho
    v_xc_sigma[mask] = vx_sigma + dec_dsigma

    return e_xc, v_xc_rho, v_xc_sigma


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

#: Map from canonical functional name → set of recognised aliases.
_FUNCTIONAL_ALIASES: dict[str, str] = {
    "lda":  "lda",
    "svwn": "lda",   # Slater exchange + VWN correlation
    "pbe":  "pbe",
}


def get_xc(
    name:      str,
    rho:       np.ndarray,
    grad_rho:  np.ndarray | None = None,
) -> XCResult:
    """
    Evaluate the named XC functional and return an XCResult.

    This is the only function dft.ks needs to import from this module.
    It dispatches to the appropriate LDA or GGA implementation, validates
    inputs, and returns a unified XCResult regardless of the functional's
    rung in the Jacob's ladder hierarchy.

    Parameters
    ----------
    name : str
        Name of the functional.  Case-insensitive.  Recognised values:
          'lda'  or 'svwn'  — Slater exchange + VWN5 correlation.
          'pbe'             — PBE GGA exchange-correlation.
    rho : ndarray, shape (n_pts,)
        Electron density at each grid point.
    grad_rho : ndarray, shape (n_pts, 3) or None
        Density gradient at each grid point.  Required for GGA
        functionals ('pbe'); ignored for LDA functionals.

    Returns
    -------
    XCResult
        Dataclass with fields:
          .e_xc        — XC energy density (n_pts,)
          .v_xc_rho    — ∂e_xc/∂ρ  (n_pts,)
          .v_xc_sigma  — ∂e_xc/∂σ  (n_pts,) for GGA; None for LDA.

    Raises
    ------
    ValueError
        If ``name`` is not a recognised functional, or if a GGA
        functional is requested but ``grad_rho`` is None.

    Examples
    --------
    LDA single-point:

        >>> import numpy as np
        >>> from qchem.dft.xc import get_xc
        >>> rho = np.array([0.1, 0.2, 0.3])
        >>> result = get_xc("lda", rho)
        >>> result.v_xc_sigma is None
        True

    PBE single-point:

        >>> grad = np.zeros((3, 3))   # ∇ρ = 0 → reduces to LDA values
        >>> result = get_xc("pbe", rho, grad)
        >>> result.v_xc_sigma.shape
        (3,)
    """
    canonical = _FUNCTIONAL_ALIASES.get(name.lower())
    if canonical is None:
        recognised = sorted(set(_FUNCTIONAL_ALIASES.keys()))
        raise ValueError(
            f"Unknown XC functional '{name}'.  "
            f"Recognised names: {recognised}."
        )

    rho = np.asarray(rho, dtype=float)

    if canonical == "lda":
        e_x, v_x = lda_x(rho)
        e_c, v_c = lda_c_vwn(rho)
        return XCResult(
            e_xc       = e_x + e_c,
            v_xc_rho   = v_x + v_c,
            v_xc_sigma = None,
        )

    if canonical == "pbe":
        if grad_rho is None:
            raise ValueError(
                "PBE is a GGA functional and requires 'grad_rho' (shape "
                "(n_pts, 3)).  Pass the density gradient array from "
                "dft.grid.eval_ao_grad_on_grid()."
            )
        grad_rho = np.asarray(grad_rho, dtype=float)
        e_xc, v_xc_rho, v_xc_sigma = pbe_xc(rho, grad_rho)
        return XCResult(
            e_xc       = e_xc,
            v_xc_rho   = v_xc_rho,
            v_xc_sigma = v_xc_sigma,
        )

    # Unreachable if _FUNCTIONAL_ALIASES is kept consistent.
    raise RuntimeError(f"Unhandled canonical functional '{canonical}'.")  # pragma: no cover
