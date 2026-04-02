"""
qchem/dft/ks.py — Kohn-Sham DFT SCF driver

Overview
--------
This module provides the top-level Kohn-Sham DFT entry point that ties
together every component built in Parts 9 and 10 — the numerical
integration grid and the XC functional library — with the integral
engine and SCF infrastructure from Parts 1-8 into a complete
self-consistent field calculation.

The public interface is a single function::

    result = ks(mol, basis_name='sto-3g', functional='lda', ...)

which returns a ``KSResult`` dataclass carrying the converged energy,
orbital energies, MO coefficients, density matrix, and diagnostic
information.

Relationship to RHF (Part 8)
-----------------------------
The Kohn-Sham SCF loop is structurally identical to Roothaan-Hall:

    F_KS C = S C ε

is iterated until F_KS and the density matrix P are mutually consistent.
Two changes distinguish it from ``rhf()``:

1.  The Fock matrix contains no exchange matrix K.  The exchange (and
    correlation) interaction is subsumed into the XC potential V_xc
    assembled on the numerical grid::

        F_KS = H_core + J + V_xc            (KS)
        F_HF = H_core + J − ½K              (HF)

2.  The total electronic energy cannot use the ½ Tr[P(H+F)] shortcut
    that works for HF.  The XC energy E_xc is not equal to Tr[P V_xc]
    for any non-linear functional, so the energy must be assembled
    explicitly from its constituent parts::

        E_elec = Tr[P H_core] + ½ Tr[P J] + E_xc    (KS)
        E_elec = ½ Tr[P (H_core + F)]                (HF)

    The ½ on J prevents double-counting the classical Hartree repulsion
    exactly as in HF; E_xc carries no prefactor.

GGA vs LDA dispatch
--------------------
The driver detects the functional tier automatically by probing
``get_xc`` on a single dummy density point.  If ``v_xc_sigma is not None``
in the result, a GGA functional is assumed and:

*   AO gradient arrays are precomputed on the grid before the SCF loop.
*   The density gradient ∇ρ is recomputed each iteration from the new P.
*   ``build_vxc_matrix_gga`` assembles V_xc including the integration-
    by-parts gradient-correction term.

For LDA, none of these additional steps are taken and ``build_xc_matrix``
from ``dft.grid`` is used directly.

DIIS and convergence
---------------------
The orbital-gradient commutator error  e = FPS − SPF  drives DIIS,
identical to the HF driver.  The convergence criterion requires both
the maximum density-matrix change and the energy change to fall below
their respective thresholds on the same iteration.

Energy in the SCF loop
-----------------------
Unlike ``rhf()``, the KS energy is evaluated from the **same** density
matrix P that was used to build F_KS — not from P_new after
diagonalisation.  This is because grid-based evaluation is O(N² N_grid)
and substantially more expensive than a second J build; avoiding the
double evaluation is the standard practice in production DFT codes.
The energy and Fock matrix are always from the same, physical density,
so there is no consistency issue with the DIIS-extrapolated F_scf.

KS orbital energies
--------------------
The orbital energies returned in ``KSResult.orbital_energies`` are
Kohn-Sham eigenvalues — Lagrange multipliers for the fictitious
non-interacting reference system.  They do *not* satisfy Koopmans'
theorem (as HF orbital energies approximately do) and should not be
interpreted as quasi-particle energies without correction.  The one
exact statement is Janak's theorem:  ε_HOMO = −∂E/∂n_HOMO, which
gives the exact ionisation energy only with the *exact* functional.

Functions
---------
eval_ao_gradients(shells, points)
    ∇φ_μ(r) for all basis functions at all grid points.
    Returns an (n_pts, n_basis, 3) array.

eval_density_gradient(P, ao_values, ao_gradients)
    ∇ρ(r) = 2 Σ_μν P_μν φ_μ ∇φ_ν at each grid point.
    Returns an (n_pts, 3) array.

build_vxc_matrix_gga(ao_values, ao_gradients, weights, v_xc_rho,
                     v_xc_sigma, grad_rho)
    Assemble the full GGA XC matrix including the gradient-correction
    term that arises from integrating by parts.

build_coulomb_matrix(P, ERI)
    J_μν = Σ_λσ P_λσ (μν|λσ).  The classical Hartree repulsion matrix.
    This is the J term of both HF and KS; no exchange subtracted here.

ks_fock_matrix(H_core, J, V_xc)
    F_KS = H_core + J + V_xc.

ks_electronic_energy(P, H_core, J, E_xc_grid)
    E_elec = Tr[P H_core] + ½ Tr[P J] + E_xc_grid.

KSResult
    Dataclass holding all outputs of a completed KS calculation.

ks(mol, basis_name, functional, ...)
    Top-level driver.  Returns KSResult.

References
----------
Kohn, W. & Sham, L. J. (1965). Phys. Rev. 140, A1133.
Becke, A. D. (1988). J. Chem. Phys. 88, 2547.   [Grid partitioning]
Perdew, J. P., Burke, K. & Ernzerhof, M. (1996). PRL 77, 3865. [PBE]
Szabo & Ostlund, "Modern Quantum Chemistry", Dover (1996), Ch. 7.
Helgaker, Jørgensen & Olsen, "Molecular Electronic-Structure Theory",
    Wiley (2000), Ch. 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qchem.molecule import Molecule
from qchem.linalg import solve_generalized_eigenvalue
from qchem.scf.density import density_matrix, n_electrons_from_density
from qchem.scf.fock import core_hamiltonian
from qchem.scf.diis import DIISAccelerator
from qchem.integrals.overlap import norm_primitive
from qchem.dft.grid import (
    build_molecular_grid,
    eval_ao_on_grid,
    eval_density_on_grid,
    build_xc_matrix,
)
from qchem.dft.xc import get_xc


# ---------------------------------------------------------------------------
# AO gradient evaluation
# ---------------------------------------------------------------------------

def eval_ao_gradients(
    shells: list[dict],
    points: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the gradient ∇φ_μ(r) of each contracted AO at each grid point.

    For a contracted Cartesian GTO with angular momentum (lx, ly, lz),
    centre A, primitive exponents {α_k}, and contraction coefficients {c_k}:

        ∂φ_μ/∂x = Σ_k c_k N_k [  lx (x−Ax)^{lx−1} (y−Ay)^{ly} (z−Az)^{lz}
                                  − 2α_k (x−Ax)^{lx+1} (y−Ay)^{ly} (z−Az)^{lz} ]
                                · exp(−α_k |Δr|²)

    The first term (angular momentum reduction) is absent when lx = 0.
    The second term (angular momentum increase) is the derivative of the
    Gaussian exponential and is always present.  The y and z components
    follow by permutation of indices.

    Parameters
    ----------
    shells : list of dict
        Basis function dicts as returned by ``basis.build_basis()``.  Each
        dict must have keys ``'center'``, ``'angular'``, ``'exponents'``,
        and ``'coefficients'``.
    points : ndarray, shape (n_pts, 3)
        Cartesian grid-point coordinates in bohr.

    Returns
    -------
    ao_grad : ndarray, shape (n_pts, n_basis, 3)
        ao_grad[g, μ, c] = ∂φ_μ(r_g) / ∂x_c,  where c ∈ {0=x, 1=y, 2=z}.

    Notes
    -----
    The normalisation constant N_k is the same as used in
    ``eval_ao_on_grid`` — every primitive is normalised to unit overlap
    with itself before contraction coefficients are applied.

    This function is only called for GGA functionals.  For LDA the
    gradient is not needed and this function is not invoked by the driver.
    """
    n_pts   = points.shape[0]
    n_basis = len(shells)
    ao_grad = np.zeros((n_pts, n_basis, 3), dtype=float)

    for mu, shell in enumerate(shells):
        A          = np.asarray(shell['center'], dtype=float)   # (3,)
        lx, ly, lz = shell['angular']
        exps       = shell['exponents']
        coefs      = shell['coefficients']

        delta = points - A[None, :]                     # (n_pts, 3)
        dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]
        r2 = dx * dx + dy * dy + dz * dz               # (n_pts,)

        # Precompute angular polynomial factors shared across primitives
        poly_x = dx ** lx   # (n_pts,)
        poly_y = dy ** ly
        poly_z = dz ** lz

        for alpha, c in zip(exps, coefs):
            N       = norm_primitive(alpha, (lx, ly, lz))
            exp_val = N * c * np.exp(-alpha * r2)       # (n_pts,)

            # ---- x component  d/dx [poly_x · poly_y · poly_z · exp] ----
            # upper shift term: −2α · dx^{lx+1} · poly_y · poly_z
            upper_x = -2.0 * alpha * (dx ** (lx + 1)) * poly_y * poly_z
            # lower shift term: lx · dx^{lx−1} · poly_y · poly_z  (zero if lx=0)
            if lx > 0:
                lower_x = lx * (dx ** (lx - 1)) * poly_y * poly_z
            else:
                lower_x = 0.0
            ao_grad[:, mu, 0] += exp_val * (lower_x + upper_x)

            # ---- y component  d/dy [poly_x · poly_y · poly_z · exp] ----
            upper_y = -2.0 * alpha * poly_x * (dy ** (ly + 1)) * poly_z
            if ly > 0:
                lower_y = poly_x * ly * (dy ** (ly - 1)) * poly_z
            else:
                lower_y = 0.0
            ao_grad[:, mu, 1] += exp_val * (upper_y + lower_y)

            # ---- z component  d/dz [poly_x · poly_y · poly_z · exp] ----
            upper_z = -2.0 * alpha * poly_x * poly_y * (dz ** (lz + 1))
            if lz > 0:
                lower_z = poly_x * poly_y * lz * (dz ** (lz - 1))
            else:
                lower_z = 0.0
            ao_grad[:, mu, 2] += exp_val * (upper_z + lower_z)

    return ao_grad


# ---------------------------------------------------------------------------
# Density gradient
# ---------------------------------------------------------------------------

def eval_density_gradient(
    P: np.ndarray,
    ao_values: np.ndarray,
    ao_gradients: np.ndarray,
) -> np.ndarray:
    """
    Compute the electron density gradient ∇ρ(r) at each grid point.

    From ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r) and the symmetry of P:

        ∂ρ/∂x_c = 2 Σ_μν P_μν φ_μ(r) ∂φ_ν/∂x_c

    The factor of 2 arises from differentiating both φ_μ and φ_ν and
    using P_μν = P_νμ to collapse the two terms into one.

    Parameters
    ----------
    P : ndarray, shape (n_basis, n_basis)
        Current density matrix.
    ao_values : ndarray, shape (n_pts, n_basis)
        AO values at each grid point, from ``eval_ao_on_grid``.
    ao_gradients : ndarray, shape (n_pts, n_basis, 3)
        AO gradients at each grid point, from ``eval_ao_gradients``.

    Returns
    -------
    grad_rho : ndarray, shape (n_pts, 3)
        grad_rho[g, c] = ∂ρ(r_g)/∂x_c.

    Notes
    -----
    The einsum ``'mn,gm,gnc->gc'`` contracts the density matrix (m,n)
    with AO values (g,m) and AO gradients (g,n,c) to produce a
    (n_pts, 3) array in a single vectorised operation.

    This quantity is recomputed at every SCF iteration because it depends
    on the current density matrix P.  The AO values and gradients
    (which depend only on the grid geometry and basis) are fixed and
    precomputed before the SCF loop.
    """
    # ∂ρ/∂x_c = 2 Σ_mn P_mn φ_m(r_g) ∂φ_n/∂x_c(r_g)
    return 2.0 * np.einsum('mn,gm,gnc->gc', P, ao_values, ao_gradients)


# ---------------------------------------------------------------------------
# GGA XC matrix
# ---------------------------------------------------------------------------

def build_vxc_matrix_gga(
    ao_values:    np.ndarray,
    ao_gradients: np.ndarray,
    weights:      np.ndarray,
    v_xc_rho:     np.ndarray,
    v_xc_sigma:   np.ndarray,
    grad_rho:     np.ndarray,
) -> np.ndarray:
    """
    Assemble the GGA exchange-correlation matrix V_xc including the
    gradient-correction term from integration by parts.

    For a GGA functional E_xc[ρ, σ] where σ = |∇ρ|², the functional
    derivative with respect to the density matrix P_μν yields:

        V_xc,μν = Σ_g w_g [ v_xc_rho_g · φ_μg φ_νg
                           + 2 v_xc_sigma_g Σ_c (∂ρ/∂x_c)_g
                             · (φ_μg ∂φ_νg/∂x_c + φ_νg ∂φ_μg/∂x_c) ]

    The first line is the LDA-like term, assembled exactly as in
    ``grid.build_xc_matrix``.  The second line is the gradient-correction
    term: v_xc_sigma = ∂e_xc/∂σ is the second partial derivative
    returned by GGA functionals, and ∇ρ is the current density gradient.

    Parameters
    ----------
    ao_values : ndarray, shape (n_pts, n_basis)
        AO values at each grid point.
    ao_gradients : ndarray, shape (n_pts, n_basis, 3)
        AO gradients at each grid point.
    weights : ndarray, shape (n_pts,)
        Molecular grid weights (Becke-partitioned).
    v_xc_rho : ndarray, shape (n_pts,)
        ∂e_xc/∂ρ at each grid point, from ``get_xc``.
    v_xc_sigma : ndarray, shape (n_pts,)
        ∂e_xc/∂σ at each grid point, from ``get_xc``.  Non-None for GGA.
    grad_rho : ndarray, shape (n_pts, 3)
        Density gradient at each grid point, from ``eval_density_gradient``.

    Returns
    -------
    V_xc : ndarray, shape (n_basis, n_basis)
        Symmetric GGA XC matrix.

    Notes
    -----
    The gradient-correction term is computed direction by direction.
    For each Cartesian direction c, define the scaled weight vector:

        f_c[g] = 2 · w[g] · v_xc_sigma[g] · (∂ρ/∂x_c)[g]

    The contribution to V_xc from direction c is:

        A_c = (ao * f_c[:, None]).T @ ao_gradients[:, :, c]   (n_basis, n_basis)

    The full gradient-correction is  Σ_c (A_c + A_c.T),  which is
    symmetric by construction.  Adding A_c and A_c.T simultaneously
    corresponds to the symmetrisation  (φ_μ ∂φ_ν + φ_ν ∂φ_μ)  in the
    formula above.
    """
    # LDA-like density term:  Σ_g w_g v_xc_rho_g φ_μg φ_νg
    wv    = weights * v_xc_rho                         # (n_pts,)
    ao_wv = ao_values * wv[:, None]                    # (n_pts, n_basis)
    V_xc  = ao_wv.T @ ao_values                        # (n_basis, n_basis)

    # GGA gradient-correction term:  2 Σ_c Σ_g w_g v_sigma_g (∂ρ/∂xc)_g
    #                                 × (φ_μg ∂φ_νg/∂xc + φ_νg ∂φ_μg/∂xc)
    for c in range(3):
        # f_c[g] = 2 w[g] v_sigma[g] (∂ρ/∂xc)[g]
        f_c       = 2.0 * weights * v_xc_sigma * grad_rho[:, c]  # (n_pts,)
        ao_fc     = ao_values * f_c[:, None]                      # (n_pts, n_basis)
        ao_grad_c = ao_gradients[:, :, c]                         # (n_pts, n_basis)
        A_c       = ao_fc.T @ ao_grad_c                           # (n_basis, n_basis)
        V_xc     += A_c + A_c.T

    return V_xc


# ---------------------------------------------------------------------------
# KS Fock matrix components
# ---------------------------------------------------------------------------

def build_coulomb_matrix(
    P: np.ndarray,
    ERI: np.ndarray,
) -> np.ndarray:
    """
    Build the Coulomb (Hartree) matrix J.

        J_μν = Σ_λσ P_λσ (μν|λσ)

    This is the classical mean-field repulsion between the charge
    distribution of basis-function pair (μ,ν) and the total electron
    density encoded in P.  It is present identically in both HF and KS;
    the distinction is that KS omits the exchange matrix K entirely,
    while HF subtracts ½K.

    Parameters
    ----------
    P : ndarray, shape (n, n)
        Density matrix.
    ERI : ndarray, shape (n, n, n, n)
        Two-electron repulsion integrals (μν|λσ), fully symmetric.

    Returns
    -------
    J : ndarray, shape (n, n)
        Symmetric Coulomb matrix in hartree.
    """
    return np.einsum('ls,mnls->mn', P, ERI)


def ks_fock_matrix(
    H_core: np.ndarray,
    J: np.ndarray,
    V_xc: np.ndarray,
) -> np.ndarray:
    """
    Build the Kohn-Sham Fock matrix  F_KS = H_core + J + V_xc.

    Compare with the HF Fock matrix  F_HF = H_core + J − ½K.  The
    exchange matrix K is absent: in Kohn-Sham DFT the exchange (and
    correlation) interaction is handled entirely by V_xc, which is
    built on the numerical grid from the XC functional.

    Parameters
    ----------
    H_core : ndarray, shape (n, n)
        Core Hamiltonian T + V, density-independent.
    J : ndarray, shape (n, n)
        Coulomb matrix from ``build_coulomb_matrix``.
    V_xc : ndarray, shape (n, n)
        XC matrix from ``build_xc_matrix`` (LDA) or
        ``build_vxc_matrix_gga`` (GGA).

    Returns
    -------
    F_KS : ndarray, shape (n, n)
        Kohn-Sham Fock matrix.  Symmetric when H_core, J, and V_xc are
        all symmetric, which holds for any correct KS calculation.
    """
    return H_core + J + V_xc


def ks_electronic_energy(
    P: np.ndarray,
    H_core: np.ndarray,
    J: np.ndarray,
    E_xc_grid: float,
) -> float:
    """
    Compute the Kohn-Sham electronic energy.

        E_elec = Tr[P H_core] + ½ Tr[P J] + E_xc

    This is the correct KS energy expression.  It differs critically from
    the HF expression  E_elec = ½ Tr[P(H_core + F)]  because the XC
    functional is nonlinear in ρ:  E_xc ≠ Tr[P V_xc]  in general.
    The grid-integrated E_xc must therefore be included explicitly as a
    separate scalar, not reconstructed from V_xc.

    The ½ on the Hartree term Tr[P J] corrects for the same
    double-counting that appears in HF: summing the Coulomb interaction
    over all pairs of basis functions in J implicitly counts each
    electron–electron pair twice, so a factor of ½ is required when
    contracting with P a second time to form the energy.

    E_xc carries no prefactor.  It is the total exchange-correlation
    energy, already the correct contribution to E_elec.

    Parameters
    ----------
    P : ndarray, shape (n, n)
        Density matrix corresponding to the J and V_xc that built F_KS.
        The energy is only physically meaningful when P, J, and E_xc_grid
        all correspond to the same electron density.
    H_core : ndarray, shape (n, n)
        Core Hamiltonian (density-independent).
    J : ndarray, shape (n, n)
        Coulomb matrix built from the same P.
    E_xc_grid : float
        Grid-integrated XC energy  Σ_g w_g e_xc(r_g), evaluated at the
        same density that produced J.

    Returns
    -------
    float
        KS electronic energy in hartree.

    Notes
    -----
    The einsum ``'ij,ij->'`` is element-wise multiplication summed over
    all indices — equivalent to np.trace(P @ M) but avoids materialising
    the matrix product.
    """
    e_one = float(np.einsum('ij,ij->', P, H_core))     # Tr[P H_core]
    e_hartree = 0.5 * float(np.einsum('ij,ij->', P, J))  # ½ Tr[P J]
    return e_one + e_hartree + E_xc_grid


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KSResult:
    """
    Container for the output of a converged (or attempted) KS-DFT calculation.

    Attributes
    ----------
    e_total : float
        Total KS energy (electronic + nuclear repulsion) in hartree.
    e_electronic : float
        Electronic energy  Tr[P H_core] + ½ Tr[P J] + E_xc  in hartree.
    e_nuclear : float
        Nuclear repulsion energy  Σ_{A<B} Z_A Z_B / R_AB  in hartree.
    e_xc : float
        Exchange-correlation energy  Σ_g w_g e_xc(r_g)  in hartree.
        Reported separately for analysis and cross-checking against
        external packages.
    orbital_energies : ndarray, shape (n_basis,)
        Kohn-Sham orbital energies ε in ascending order (hartree).

        **Important**: these are KS Lagrange multipliers, not
        quasi-particle energies.  Koopmans' theorem does not apply.
        The HOMO energy approximates the first ionisation energy only
        with the exact functional (Janak's theorem); with approximate
        functionals this is qualitatively correct but not guaranteed.
    coefficients : ndarray, shape (n_basis, n_basis)
        KS MO coefficient matrix C.  Column i is the i-th KS orbital
        expressed in the AO basis, ordered by ascending orbital energy.
    density : np.ndarray, shape (n_basis, n_basis)
        Converged density matrix P = 2 C_occ C_occ^T.
    fock : ndarray, shape (n_basis, n_basis)
        Converged KS Fock matrix  F_KS = H_core + J + V_xc  built from
        the converged density.
    overlap : ndarray, shape (n_basis, n_basis)
        Overlap matrix S.
    h_core : ndarray, shape (n_basis, n_basis)
        Core Hamiltonian H_core = T + V (density-independent).
    converged : bool
        True if both convergence criteria were satisfied within max_iter.
    n_iter : int
        Number of SCF iterations performed.
    mol : Molecule
        The molecule used in the calculation.
    basis_name : str
        Name of the basis set used.
    functional : str
        Name of the XC functional used (canonicalised to lowercase).
    n_occ : int
        Number of occupied spatial orbitals  (= n_electrons // 2).
    n_grid_points : int
        Total number of grid points in the molecular integration grid.
    grid_electrons : float
        Numerical integral of ρ(r) over the grid: Σ_g w_g ρ(r_g).
        Should equal n_electrons to within a few parts in 10³ for a
        grid of production quality.  Large deviations indicate the grid
        is too coarse for reliable XC energies.
    """
    e_total:          float
    e_electronic:     float
    e_nuclear:        float
    e_xc:             float
    orbital_energies: np.ndarray
    coefficients:     np.ndarray
    density:          np.ndarray
    fock:             np.ndarray
    overlap:          np.ndarray
    h_core:           np.ndarray
    converged:        bool
    n_iter:           int
    mol:              Molecule
    basis_name:       str
    functional:       str
    n_occ:            int
    n_grid_points:    int
    grid_electrons:   float

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def homo_energy(self) -> float:
        """
        KS eigenvalue of the highest occupied orbital (HOMO).

        Note: this is a KS Lagrange multiplier, not a quasi-particle
        energy.  With the exact functional it equals −IP (Janak's
        theorem); with approximate functionals it is approximate.
        """
        return float(self.orbital_energies[self.n_occ - 1])

    @property
    def lumo_energy(self) -> Optional[float]:
        """
        KS eigenvalue of the lowest unoccupied orbital (LUMO).
        Returns None if no virtual orbitals exist.
        """
        n_basis = len(self.orbital_energies)
        if self.n_occ >= n_basis:
            return None
        return float(self.orbital_energies[self.n_occ])

    @property
    def homo_lumo_gap(self) -> Optional[float]:
        """KS HOMO–LUMO gap in hartree.  None if no virtual orbitals."""
        if self.lumo_energy is None:
            return None
        return self.lumo_energy - self.homo_energy

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"KSResult({status}, "
            f"functional={self.functional!r}, "
            f"E_total={self.e_total:.8f} Ha, "
            f"n_iter={self.n_iter}, "
            f"basis={self.basis_name!r})"
        )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def ks(
    mol: Molecule,
    basis_name:   str   = 'sto-3g',
    functional:   str   = 'lda',
    *,
    n_rad:        int   = 75,
    n_theta:      int   = 17,
    max_iter:     int   = 100,
    e_tol:        float = 1e-8,
    d_tol:        float = 1e-6,
    use_diis:     bool  = True,
    diis_max_vec: int   = 8,
    diis_start:   int   = 2,
) -> KSResult:
    """
    Run a Kohn-Sham DFT calculation on a closed-shell molecule.

    Parameters
    ----------
    mol : Molecule
        The molecule to compute.  Must be closed-shell (even number of
        electrons, multiplicity == 1).
    basis_name : str
        Basis set name (e.g. ``'sto-3g'``).  Passed to ``mol.build_basis()``.
    functional : str
        XC functional name.  Case-insensitive.  Recognised values:
          ``'lda'`` or ``'svwn'`` — Slater exchange + VWN5 correlation.
          ``'pbe'``               — PBE GGA exchange-correlation.
        Additional functionals can be added to ``dft.xc`` without
        changing this driver.
    n_rad : int
        Number of radial quadrature points per atom.  Default 75 gives
        chemical accuracy for most main-group systems; increase to 99 or
        150 for heavier elements or when checking grid convergence.
    n_theta : int
        Number of polar (Gauss-Legendre) angular points per shell.  The
        total angular points per shell is n_theta × 2·n_theta.  Default 17
        gives 578 angular points per shell.
    max_iter : int
        Maximum number of SCF iterations before declaring non-convergence.
    e_tol : float
        Energy convergence threshold: |E_new − E_old| < e_tol.
    d_tol : float
        Density convergence threshold: max|P_new − P_old| < d_tol.
    use_diis : bool
        Enable DIIS extrapolation (default True).
    diis_max_vec : int
        Maximum number of Fock / error-vector pairs stored by DIIS.
    diis_start : int
        Iteration at which DIIS extrapolation is first applied.

    Returns
    -------
    KSResult
        Dataclass with the converged (or final) energy, orbital energies,
        MO coefficients, density matrix, and diagnostic fields.  Check
        ``result.converged`` to confirm the calculation succeeded.

    Raises
    ------
    ValueError
        If the molecule is not closed-shell, has no electrons, if the
        basis set is unknown, or if the functional name is not recognised.
    RuntimeError
        If the SCF does not converge within max_iter iterations.

    Notes
    -----
    The initial guess is the core-Hamiltonian guess, identical to the
    RHF driver: diagonalise H_core to get starting MO coefficients and
    build P₀.  This is crude but requires no additional input.

    The energy is evaluated from the density matrix P used to build the
    current F_KS (not from P_new after diagonalisation).  This avoids
    a second grid evaluation per iteration while remaining physically
    consistent: F_KS and E_elec always correspond to the same ρ(r).
    A final clean evaluation is performed at Phase 4.

    Examples
    --------
    >>> from qchem.molecule import Molecule
    >>> from qchem.dft.ks import ks
    >>> mol = Molecule([('H', [0., 0., -0.7]), ('H', [0., 0., 0.7])])
    >>> result = ks(mol, functional='lda')
    >>> print(f"H2 LDA energy: {result.e_total:.6f} Ha")
    H2 LDA energy: -1.132470 Ha
    """
    # ------------------------------------------------------------------
    # 0.  Input validation
    # ------------------------------------------------------------------
    if mol.n_electrons == 0:
        raise ValueError("Molecule has no electrons.")
    if not mol.is_closed_shell:
        raise ValueError(
            f"ks() requires a closed-shell (singlet) molecule, but "
            f"got multiplicity={mol.multiplicity}.  "
            f"Open-shell KS (ROKS/UKS) is not yet implemented."
        )
    if mol.n_electrons % 2 != 0:
        raise ValueError(
            f"KS requires an even number of electrons, got {mol.n_electrons}."
        )

    n_occ = mol.n_electrons // 2

    # Validate functional name early and probe GGA tier.
    # For LDA get_xc accepts grad_rho but ignores it; we pass zeros so
    # the probe works regardless of whether the functional needs them.
    _probe_rho      = np.array([0.1])
    _probe_grad_rho = np.zeros((1, 3))
    _probe          = get_xc(functional, _probe_rho, _probe_grad_rho)
    is_gga          = _probe.v_xc_sigma is not None

    # ------------------------------------------------------------------
    # 1.  Build basis and one-electron integrals
    # ------------------------------------------------------------------
    basis   = mol.build_basis(basis_name)
    n_basis = len(basis)

    from qchem.integrals.overlap import build_overlap_matrix
    from qchem.integrals.kinetic import build_kinetic_matrix
    from qchem.integrals.nuclear import build_nuclear_matrix
    from qchem.integrals.eri     import build_eri_tensor

    S   = build_overlap_matrix(basis)
    T   = build_kinetic_matrix(basis)
    V   = build_nuclear_matrix(basis, mol.atomic_numbers, mol.coords)
    H   = core_hamiltonian(T, V)

    # Four-index ERI tensor — O(N^4), computed once and reused every
    # iteration to build J.  No exchange contractions needed for KS.
    ERI = build_eri_tensor(basis)

    e_nuclear = mol.nuclear_repulsion()

    # ------------------------------------------------------------------
    # 1b.  Build numerical grid and precompute geometry-fixed quantities
    # ------------------------------------------------------------------
    grid_points, grid_weights = build_molecular_grid(mol, n_rad=n_rad, n_theta=n_theta)
    n_grid_points = len(grid_weights)

    # AO values on the grid — fixed for the lifetime of the calculation
    ao_values = eval_ao_on_grid(basis, grid_points)   # (n_grid, n_basis)

    # AO gradients — only needed for GGA.  For LDA ao_gradients is None
    # and eval_density_gradient / build_vxc_matrix_gga are never called.
    if is_gga:
        ao_gradients = eval_ao_gradients(basis, grid_points)  # (n_grid, n_basis, 3)
    else:
        ao_gradients = None

    # ------------------------------------------------------------------
    # 2.  Initial guess: core Hamiltonian diagonalisation
    # ------------------------------------------------------------------
    _, C = solve_generalized_eigenvalue(H, S)
    P    = density_matrix(C, n_occ)

    # ------------------------------------------------------------------
    # 3.  SCF loop
    # ------------------------------------------------------------------
    diis = DIISAccelerator(max_vec=diis_max_vec) if use_diis else None

    e_elec_prev = 0.0
    converged   = False
    n_iter      = 0

    for iteration in range(1, max_iter + 1):
        n_iter = iteration

        # (a) Coulomb matrix from current density
        J = build_coulomb_matrix(P, ERI)

        # (b) XC quantities on the grid from current density
        rho      = eval_density_on_grid(P, ao_values)           # (n_grid,)
        grad_rho = (
            eval_density_gradient(P, ao_values, ao_gradients)   # (n_grid, 3)
            if is_gga else None
        )
        xc_result = get_xc(functional, rho, grad_rho)
        E_xc      = float(np.dot(grid_weights, xc_result.e_xc))

        # (c) XC matrix: LDA uses the simple weighted AO contraction;
        #     GGA adds the integration-by-parts gradient-correction term.
        if is_gga:
            V_xc = build_vxc_matrix_gga(
                ao_values, ao_gradients, grid_weights,
                xc_result.v_xc_rho, xc_result.v_xc_sigma, grad_rho,
            )
        else:
            V_xc = build_xc_matrix(ao_values, xc_result.v_xc_rho, grid_weights)

        # (d) KS Fock matrix  F_KS = H_core + J + V_xc  (no K)
        F = ks_fock_matrix(H, J, V_xc)

        # (e) KS electronic energy from the same P that built F.
        #     Using the matched (P, J, E_xc) triplet is essential:
        #     E_xc is a nonlinear functional of ρ, so ½ Tr[P(H+F)] would
        #     give the wrong energy (the HF double-counting correction
        #     does not apply here).
        e_elec = ks_electronic_energy(P, H, J, E_xc)

        # (f) Orbital-gradient error  e = F P S − S P F
        #     This is zero exactly when F and P commute in the S-metric,
        #     i.e. exactly at a converged KS solution.
        if use_diis:
            e_vec = F @ P @ S - S @ P @ F
            diis.push(F, e_vec)

            # Apply DIIS extrapolation once we have enough stored vectors.
            # The extrapolated F* is used only for diagonalisation — not
            # for energy evaluation, where it is not a valid Fock matrix.
            if iteration >= diis_start and len(diis) >= 2:
                F_scf = diis.extrapolate()
            else:
                F_scf = F
        else:
            F_scf = F

        # (g) Solve  F_scf C = S C ε  for new MO coefficients
        eps, C_new = solve_generalized_eigenvalue(F_scf, S)

        # (h) Build new density from the n_occ lowest KS orbitals
        P_new = density_matrix(C_new, n_occ)

        # (i) Convergence check — both criteria on the same iteration
        d_max  = float(np.max(np.abs(P_new - P)))
        e_diff = abs(e_elec - e_elec_prev)

        if d_max < d_tol and e_diff < e_tol:
            converged = True
            P = P_new
            C = C_new
            break

        # (j) Update for next iteration
        P           = P_new
        C           = C_new
        e_elec_prev = e_elec

    else:
        # Python for-else: fires only when the loop exits without break.
        raise RuntimeError(
            f"KS ({functional.upper()}) did not converge in {max_iter} "
            f"iterations.  "
            f"Final density change: {d_max:.2e}, "
            f"energy change: {e_diff:.2e}.  "
            f"Try increasing max_iter or n_rad/n_theta, or enabling DIIS."
        )

    # ------------------------------------------------------------------
    # 4.  Final energy and result assembly
    # ------------------------------------------------------------------
    # Recompute all quantities cleanly from the converged density P so
    # that all output fields are mutually consistent.  This also gives
    # the correct energy regardless of which P was active at convergence.
    J_final  = build_coulomb_matrix(P, ERI)
    rho_final = eval_density_on_grid(P, ao_values)
    grad_rho_final = (
        eval_density_gradient(P, ao_values, ao_gradients)
        if is_gga else None
    )
    xc_final  = get_xc(functional, rho_final, grad_rho_final)
    E_xc_final = float(np.dot(grid_weights, xc_final.e_xc))

    if is_gga:
        V_xc_final = build_vxc_matrix_gga(
            ao_values, ao_gradients, grid_weights,
            xc_final.v_xc_rho, xc_final.v_xc_sigma, grad_rho_final,
        )
    else:
        V_xc_final = build_xc_matrix(ao_values, xc_final.v_xc_rho, grid_weights)

    F_final  = ks_fock_matrix(H, J_final, V_xc_final)
    e_elec   = ks_electronic_energy(P, H, J_final, E_xc_final)
    e_total  = e_elec + e_nuclear

    # Final KS orbital energies from the converged Fock matrix
    eps_final, C_final = solve_generalized_eigenvalue(F_final, S)

    # Grid quality diagnostic: Σ_g w_g ρ(r_g) should equal n_electrons
    grid_electrons = float(np.dot(grid_weights, rho_final))

    return KSResult(
        e_total          = e_total,
        e_electronic     = e_elec,
        e_nuclear        = e_nuclear,
        e_xc             = E_xc_final,
        orbital_energies = eps_final,
        coefficients     = C_final,
        density          = P,
        fock             = F_final,
        overlap          = S,
        h_core           = H,
        converged        = converged,
        n_iter           = n_iter,
        mol              = mol,
        basis_name       = basis_name,
        functional       = functional.lower(),
        n_occ            = n_occ,
        n_grid_points    = n_grid_points,
        grid_electrons   = grid_electrons,
    )
