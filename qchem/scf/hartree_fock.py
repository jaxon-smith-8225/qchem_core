"""
qchem/scf/hartree_fock.py — Restricted Hartree-Fock SCF driver

Overview
--------
This module provides the top-level RHF entry point that orchestrates all
of the lower-level building blocks — integrals, density matrix, Fock
matrix, DIIS accelerator, and the generalised eigenvalue solver — into
a complete self-consistent field calculation.

The public interface is a single function:

    result = rhf(mol, basis_name='sto-3g', ...)

which returns an ``RHFResult`` dataclass carrying the converged energy,
orbital energies, MO coefficients, density matrix, and diagnostic
information.

Theory
------
Restricted Hartree-Fock for a closed-shell molecule with N electrons
solves the Roothaan-Hall equations:

    F C = S C ε                                           (1)

self-consistently, where:

    F_μν  = H^core_μν + G_μν(P)                          (2)
    G_μν  = Σ_λσ P_λσ [ (μν|λσ) − ½ (μλ|νσ) ]          (3)
    P_μν  = 2 Σ_i^{N/2} C_μi C_νi                       (4)

Because F depends on P and P depends on C which comes from F, equations
(1–4) must be iterated to self-consistency.

SCF Algorithm
-------------
1.  Compute the fixed one-electron integrals: S, T, V → H_core = T + V.
2.  Compute the fixed two-electron integrals: ERI tensor (μν|λσ).
3.  Form an initial guess P₀ by diagonalising H_core (core guess).
4.  Begin SCF iterations:
    a.  Build F from current P.
    b.  Compute the orbital-gradient error  e = FPS − SPF.
        At convergence, F and P commute, so this quantity → 0.
    c.  Push (F, e) into the DIIS accelerator and extrapolate to F*.
    d.  Solve F* C = S C ε for new MO coefficients.
    e.  Build P_new from the n_occ lowest MOs.
    f.  Check convergence:
          • max|P_new − P| < d_tol   (density change)
          • |E_new − E_old| < e_tol  (energy change)
        Both must pass on the same iteration.
    g.  If converged, exit; else P ← P_new.
5.  Compute E_total = E_elec(P, H_core, F) + V_nn.

DIIS (Direct Inversion in the Iterative Subspace)
--------------------------------------------------
The orbital-gradient error vector e = FPS − SPF (flattened to 1-D) is
used rather than the density-change vector because:
  - It is zero exactly at convergence (F and P commute when both are
    built from the same MO set).
  - It is dimensionally consistent with F, so the linear combination
    Σ_i c_i F_i is physically meaningful.
  - It converges quadratically near the solution.

DIIS is enabled by default but can be switched off via use_diis=False.
When off, the extrapolated F* is just the current F (no mixing).

References
----------
Roothaan, C. C. J. (1951). Rev. Mod. Phys. 23, 69.
Szabo & Ostlund, "Modern Quantum Chemistry", Dover (1996), Ch. 3.
Pulay, P. (1980). Chem. Phys. Lett. 73, 393.  [DIIS]
Helgaker, Jørgensen & Olsen, "Molecular Electronic-Structure Theory",
    Wiley (2000), Ch. 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qchem.molecule import Molecule
from qchem.linalg import solve_generalized_eigenvalue
from qchem.scf.density import density_matrix, n_electrons_from_density
from qchem.scf.fock import core_hamiltonian, fock_matrix, electronic_energy
from qchem.scf.diis import DIISAccelerator


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RHFResult:
    """
    Container for the output of a converged (or attempted) RHF calculation.

    Attributes
    ----------
    e_total : float
        Total electronic + nuclear repulsion energy in hartree.
    e_electronic : float
        Electronic contribution  ½ Tr[P(H_core + F)]  in hartree.
    e_nuclear : float
        Nuclear repulsion energy  Σ_{A<B} Z_A Z_B / R_AB  in hartree.
    orbital_energies : ndarray, shape (n_basis,)
        Orbital energies ε in ascending order (hartree).
    coefficients : ndarray, shape (n_basis, n_basis)
        MO coefficient matrix C.  Column i is the i-th MO expressed in
        the AO basis, ordered by ascending orbital energy.
    density : ndarray, shape (n_basis, n_basis)
        Converged density matrix P = 2 C_occ C_occ^T.
    fock : ndarray, shape (n_basis, n_basis)
        Converged Fock matrix F built from the converged P.
    overlap : ndarray, shape (n_basis, n_basis)
        Overlap matrix S (read-only reference, not recomputed).
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
    n_occ : int
        Number of occupied spatial orbitals  (= n_electrons // 2).
    """
    e_total:         float
    e_electronic:    float
    e_nuclear:       float
    orbital_energies: np.ndarray
    coefficients:    np.ndarray
    density:         np.ndarray
    fock:            np.ndarray
    overlap:         np.ndarray
    h_core:          np.ndarray
    converged:       bool
    n_iter:          int
    mol:             Molecule
    basis_name:      str
    n_occ:           int

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def homo_energy(self) -> float:
        """Energy of the highest occupied molecular orbital (HOMO)."""
        return float(self.orbital_energies[self.n_occ - 1])

    @property
    def lumo_energy(self) -> Optional[float]:
        """
        Energy of the lowest unoccupied molecular orbital (LUMO).
        Returns None if there are no virtual orbitals.
        """
        n_basis = len(self.orbital_energies)
        if self.n_occ >= n_basis:
            return None
        return float(self.orbital_energies[self.n_occ])

    @property
    def homo_lumo_gap(self) -> Optional[float]:
        """HOMO–LUMO gap in hartree.  None if no virtual orbitals exist."""
        if self.lumo_energy is None:
            return None
        return self.lumo_energy - self.homo_energy

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"RHFResult({status}, "
            f"E_total={self.e_total:.8f} Ha, "
            f"n_iter={self.n_iter}, "
            f"basis={self.basis_name!r})"
        )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def rhf(
    mol: Molecule,
    basis_name: str = 'sto-3g',
    *,
    max_iter:    int   = 100,
    e_tol:       float = 1e-8,
    d_tol:       float = 1e-6,
    use_diis:    bool  = True,
    diis_max_vec: int  = 8,
    diis_start:   int  = 2,
) -> RHFResult:
    """
    Run a Restricted Hartree-Fock calculation on a closed-shell molecule.

    Parameters
    ----------
    mol : Molecule
        The molecule to compute.  Must be closed-shell (even number of
        electrons and multiplicity == 1).
    basis_name : str
        Name of the basis set to use (e.g. 'sto-3g').  Passed directly
        to mol.build_basis().
    max_iter : int
        Maximum number of SCF iterations before declaring non-convergence.
    e_tol : float
        Energy convergence threshold: the SCF is considered converged
        when |E_new − E_old| < e_tol for two successive iterations that
        also satisfy the density criterion.
    d_tol : float
        Density convergence threshold: the SCF is considered converged
        when max|P_new − P_old| < d_tol.
    use_diis : bool
        Whether to use DIIS extrapolation to accelerate convergence.
        Default True.  Setting False falls back to plain Roothaan-Hall.
    diis_max_vec : int
        Maximum number of Fock/error vector pairs stored by DIIS.
    diis_start : int
        SCF iteration number at which DIIS extrapolation is first applied.
        For the first (diis_start − 1) iterations the current Fock matrix
        is used directly.  This gives the density a few steps to move away
        from the initial guess before DIIS takes over.

    Returns
    -------
    RHFResult
        Dataclass containing the converged (or final) energy, orbital
        energies, MO coefficients, density matrix, and diagnostic fields.
        Check result.converged to confirm the calculation succeeded.

    Raises
    ------
    ValueError
        If the molecule is not closed-shell (open-shell or odd electron
        count), if the molecule has no electrons, or if the basis set is
        unknown.
    RuntimeError
        If the SCF does not converge within max_iter iterations.

    Notes
    -----
    Convergence requires both criteria (energy and density) to be
    satisfied on the same iteration.  The density criterion is usually
    the tighter of the two in practice.

    The initial guess is the core-Hamiltonian guess: diagonalise H_core
    in the AO basis (equivalent to switching off all electron-electron
    repulsion) to get a starting set of MO coefficients, then build P₀.
    This is crude but universal — it requires no additional input.

    Examples
    --------
    >>> from qchem.molecule import Molecule
    >>> from qchem.scf.hartree_fock import rhf
    >>> mol = Molecule([('H', [0., 0., -0.7]), ('H', [0., 0., 0.7])])
    >>> result = rhf(mol)
    >>> print(f"H2 total energy: {result.e_total:.6f} Ha")
    H2 total energy: -1.116714 Ha
    """
    # ------------------------------------------------------------------
    # 0.  Input validation
    # ------------------------------------------------------------------
    if mol.n_electrons == 0:
        raise ValueError("Molecule has no electrons.")
    if not mol.is_closed_shell:
        raise ValueError(
            f"rhf() requires a closed-shell (singlet) molecule, but "
            f"got multiplicity={mol.multiplicity}.  "
            f"Use an unrestricted solver for open-shell systems."
        )
    if mol.n_electrons % 2 != 0:
        # Should not be reachable if is_closed_shell is correct, but be explicit.
        raise ValueError(
            f"RHF requires an even number of electrons, got {mol.n_electrons}."
        )

    n_occ = mol.n_electrons // 2

    # ------------------------------------------------------------------
    # 1.  Build the basis and compute all integrals
    # ------------------------------------------------------------------
    # build_basis raises KeyError for unknown basis names — let it propagate.
    basis = mol.build_basis(basis_name)
    n_basis = len(basis)

    # One-electron integrals (computed once)
    from qchem.integrals.overlap  import build_overlap_matrix
    from qchem.integrals.kinetic  import build_kinetic_matrix
    from qchem.integrals.nuclear  import build_nuclear_matrix
    from qchem.integrals.eri      import build_eri_tensor

    S   = build_overlap_matrix(basis)
    T   = build_kinetic_matrix(basis)
    V   = build_nuclear_matrix(basis, mol.atomic_numbers, mol.coords)
    H   = core_hamiltonian(T, V)

    # Two-electron integrals (O(N^4), computed once)
    print('before')
    ERI = build_eri_tensor(basis)
    print('after')

    # Nuclear repulsion energy (constant)
    e_nuclear = mol.nuclear_repulsion()

    # ------------------------------------------------------------------
    # 2.  Initial guess: core Hamiltonian diagonalisation
    # ------------------------------------------------------------------
    _, C = solve_generalized_eigenvalue(H, S)
    P = density_matrix(C, n_occ)

    # ------------------------------------------------------------------
    # 3.  SCF loop
    # ------------------------------------------------------------------
    diis = DIISAccelerator(max_vec=diis_max_vec) if use_diis else None

    e_elec_prev = 0.0
    converged   = False
    n_iter      = 0

    for iteration in range(1, max_iter + 1):
        n_iter = iteration

        # (a) Build Fock matrix from current density
        F = fock_matrix(H, P, ERI)

        # (b) Compute orbital-gradient error vector  e = FPS − SPF
        #     This is zero exactly at convergence (F and P commute).
        if use_diis:
            e_vec = F @ P @ S - S @ P @ F
            diis.push(F, e_vec)

            # (c) Extrapolate with DIIS once we have enough vectors
            if iteration >= diis_start and len(diis) >= 2:
                F_scf = diis.extrapolate()
            else:
                F_scf = F
        else:
            F_scf = F

        # (d) Solve generalised eigenvalue problem with (possibly extrapolated) F
        eps, C_new = solve_generalized_eigenvalue(F_scf, S)

        # (e) Build new density matrix from n_occ lowest MOs
        P_new = density_matrix(C_new, n_occ)

        # (f) Evaluate current energy using the un-extrapolated F
        #     (the extrapolated F* is not a valid Fock matrix for a
        #     specific P, so energetics must use the physical F(P))
        e_elec = electronic_energy(P_new, H, fock_matrix(H, P_new, ERI))

        # (g) Convergence check
        d_max  = float(np.max(np.abs(P_new - P)))
        e_diff = abs(e_elec - e_elec_prev)

        if d_max < d_tol and e_diff < e_tol:
            converged = True
            P = P_new
            C = C_new
            break

        # (h) Update for next iteration
        P           = P_new
        C           = C_new
        e_elec_prev = e_elec

    else:
        # Loop completed without break — max_iter reached
        raise RuntimeError(
            f"RHF did not converge in {max_iter} iterations. "
            f"Final density change: {d_max:.2e}, "
            f"energy change: {e_diff:.2e}. "
            f"Try increasing max_iter, loosening tolerances, or enabling DIIS."
        )

    # ------------------------------------------------------------------
    # 4.  Final energy and result
    # ------------------------------------------------------------------
    # Recompute F and energy from the final, converged P
    F_final  = fock_matrix(H, P, ERI)
    e_elec   = electronic_energy(P, H, F_final)
    e_total  = e_elec + e_nuclear

    # Final MO coefficients and orbital energies from the converged F
    eps_final, C_final = solve_generalized_eigenvalue(F_final, S)

    return RHFResult(
        e_total          = e_total,
        e_electronic     = e_elec,
        e_nuclear        = e_nuclear,
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
        n_occ            = n_occ,
    )
