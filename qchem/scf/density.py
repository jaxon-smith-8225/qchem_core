"""
qchem/scf/density.py — density matrix construction

Overview
--------
The one-particle density matrix P is the central quantity in every SCF
iteration.  For restricted Hartree-Fock (RHF) — closed-shell, singlet
systems — it is built directly from the occupied columns of the MO
coefficient matrix C:

    P_μν = 2 Σ_{i=1}^{N_occ} C_μi C_νi

or in matrix form:

    P = 2 C_occ C_occ^T

where C_occ is the (n_basis, n_occ) submatrix of C containing only the
occupied MO columns.

The factor of 2 accounts for the double occupation of each spatial orbital
by one alpha and one beta electron.

The density matrix feeds into:
    - the two-electron part of the Fock matrix  G_μν = Σ_λσ P_λσ [(μν|σλ) - ½(μλ|σν)]
    - the SCF energy expression  E_elec = ½ Tr[P (H_core + F)]
    - convergence checks via Tr[PS] = n_electrons

Functions
---------
density_matrix(C, n_occ)
    Build P for a closed-shell (RHF) system.

n_electrons_from_density(P, S)
    Compute Tr[PS] as a consistency check.  Should equal n_electrons
    to numerical precision at every SCF iteration.

References
----------
Szabo & Ostlund, "Modern Quantum Chemistry", Dover (1996),
    Eqs. 3.145, 3.154.
Helgaker, Jørgensen & Olsen, "Molecular Electronic-Structure Theory",
    Wiley (2000), Eq. 10.5.1.
"""

from __future__ import annotations

import numpy as np


def density_matrix(C: np.ndarray, n_occ: int) -> np.ndarray:
    """
    Build the RHF closed-shell density matrix P = 2 C_occ C_occ^T.

    Parameters
    ----------
    C : ndarray, shape (n_basis, n_mo)
        MO coefficient matrix.  Columns are molecular orbitals ordered by
        ascending energy (as returned by linalg.solve_generalized_eigenvalue).
    n_occ : int
        Number of occupied orbitals.  For a closed-shell system this is
        n_electrons // 2.

    Returns
    -------
    P : ndarray, shape (n_basis, n_basis)
        Real symmetric density matrix.

    Raises
    ------
    ValueError
        If n_occ is negative or exceeds the number of available MO columns.

    Notes
    -----
    The occupied block is the leftmost n_occ columns of C (lowest energy
    orbitals).  C itself is not consumed or modified.

    Examples
    --------
    >>> import numpy as np
    >>> # Minimal 2-basis, 1-occupied-orbital toy system
    >>> C = np.eye(2)
    >>> P = density_matrix(C, n_occ=1)
    >>> P
    array([[2., 0.],
           [0., 0.]])
    """
    n_basis, n_mo = C.shape

    if n_occ < 0:
        raise ValueError(f"n_occ must be >= 0, got {n_occ}")
    if n_occ > n_mo:
        raise ValueError(
            f"n_occ ({n_occ}) exceeds the number of MO columns ({n_mo})"
        )

    C_occ = C[:, :n_occ]               # shape (n_basis, n_occ)
    P = 2.0 * C_occ @ C_occ.T          # shape (n_basis, n_basis)
    return P


def n_electrons_from_density(P: np.ndarray, S: np.ndarray) -> float:
    """
    Compute the number of electrons as Tr[PS].

    For a converged (or internally consistent) density matrix this must
    equal n_electrons to within numerical precision.  Useful as a
    per-iteration sanity check inside the SCF loop.

    Parameters
    ----------
    P : ndarray, shape (n_basis, n_basis)
        Density matrix.
    S : ndarray, shape (n_basis, n_basis)
        Overlap matrix.

    Returns
    -------
    float
        Tr[P S] — should be n_electrons for a correct density matrix.

    Notes
    -----
    np.trace(P @ S) and np.einsum('ij,ji->', P, S) are equivalent;
    the einsum avoids materialising the full matrix product.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[2., 0.], [0., 0.]])
    >>> S = np.eye(2)
    >>> n_electrons_from_density(P, S)
    2.0
    """
    return float(np.einsum('ij,ji->', P, S))
