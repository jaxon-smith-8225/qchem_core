"""
qchem/scf/fock.py — core Hamiltonian and Fock matrix assembly

Overview
--------
This module builds the two matrices that drive every RHF SCF iteration:

1.  The **core Hamiltonian**  H_core = T + V

    A fixed one-electron matrix combining kinetic energy (T) and nuclear
    attraction (V).  It is assembled once before the SCF loop and never
    changes, because it does not depend on the electron density.

2.  The **Fock matrix**  F = H_core + G(P)

    The effective one-electron Hamiltonian.  It does depend on the density
    matrix P, which is why the SCF loop must iterate until F and P are
    mutually consistent.  The two-electron contribution G(P) is:

        G_μν = Σ_λσ P_λσ [ (μν|λσ) − ½ (μλ|νσ) ]
             = J_μν − ½ K_μν

    where J is the Coulomb matrix (classical mean-field repulsion) and K
    is the exchange matrix (a purely quantum, Pauli-exclusion effect that
    acts only between same-spin electrons).

Theory
------
Coulomb term:
    J_μν = Σ_λσ P_λσ (μν|λσ)

    The charge distribution of basis-function pair (μ,ν) repels the
    charge distribution of pair (λ,σ), weighted by how much (λ,σ) is
    occupied in the current density.

Exchange term:
    K_μν = Σ_λσ P_λσ (μλ|νσ)

    Arises from the antisymmetry of the wavefunction (Pauli exclusion).
    Has no classical analogue.  The factor of ½ in G = J − ½K accounts
    for the fact that, in a closed-shell system, exchange acts only
    between same-spin electrons (half the total).

Electronic energy:
    E_elec = ½ Tr[P (H_core + F)]

    The factor of ½ prevents double-counting the electron–electron
    repulsion, since G already sums over both electron indices.

Functions
---------
core_hamiltonian(T, V)
    H_core = T + V.

fock_matrix(H_core, P, ERI)
    F = H_core + G(P), where G is built from the full ERI tensor.

electronic_energy(P, H_core, F)
    E_elec = ½ Tr[P (H_core + F)].

References
----------
Szabo & Ostlund, "Modern Quantum Chemistry", Dover (1996),
    Eqs. 3.151, 3.154, 3.184.
Helgaker, Jørgensen & Olsen, "Molecular Electronic-Structure Theory",
    Wiley (2000), Eqs. 10.4.20, 10.5.5.
"""

from __future__ import annotations

import numpy as np


def core_hamiltonian(T: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Assemble the core Hamiltonian H_core = T + V.

    H_core is the one-electron part of the Fock operator.  It is
    independent of the electron density and is therefore built once
    before the SCF loop begins.

    Parameters
    ----------
    T : ndarray, shape (n, n)
        Kinetic energy matrix (symmetric), as returned by
        integrals.kinetic.build_kinetic_matrix.
    V : ndarray, shape (n, n)
        Nuclear attraction matrix (symmetric), as returned by
        integrals.nuclear.build_nuclear_matrix.

    Returns
    -------
    H_core : ndarray, shape (n, n)
        Symmetric core Hamiltonian matrix.

    Raises
    ------
    ValueError
        If T and V do not have the same shape, or if either is not
        2-dimensional.
    """
    T = np.asarray(T, dtype=float)
    V = np.asarray(V, dtype=float)

    if T.ndim != 2 or V.ndim != 2:
        raise ValueError(
            f"T and V must be 2-dimensional, got shapes {T.shape} and {V.shape}"
        )
    if T.shape != V.shape:
        raise ValueError(
            f"T and V must have the same shape, got {T.shape} and {V.shape}"
        )

    return T + V


def fock_matrix(
    H_core: np.ndarray,
    P: np.ndarray,
    ERI: np.ndarray,
) -> np.ndarray:
    """
    Build the Fock matrix F = H_core + G(P).

    G is the two-electron contribution:

        G_μν = Σ_λσ P_λσ [ (μν|λσ) − ½ (μλ|νσ) ]

    which decomposes into a Coulomb part J and an exchange part K:

        J_μν = Σ_λσ P_λσ (μν|λσ)
        K_μν = Σ_λσ P_λσ (μλ|νσ)
        G    = J − ½ K

    Parameters
    ----------
    H_core : ndarray, shape (n, n)
        Core Hamiltonian matrix, as returned by core_hamiltonian().
        This is not modified.
    P : ndarray, shape (n, n)
        Current density matrix, as returned by density.density_matrix().
    ERI : ndarray, shape (n, n, n, n)
        Two-electron repulsion integrals (μν|λσ), as returned by
        integrals.eri.build_eri_tensor().  Expected to be fully
        symmetric under all 8 index permutations.

    Returns
    -------
    F : ndarray, shape (n, n)
        Fock matrix.  Symmetric if H_core and P are symmetric (which
        they always are in a correct RHF calculation).

    Raises
    ------
    ValueError
        If matrix dimensions are mutually inconsistent.

    Notes
    -----
    The einsum contractions written out explicitly:

        J_μν = Σ_λσ P_λσ ERI[μ,ν,λ,σ]   →  'ls,mnls->mn'
        K_μν = Σ_λσ P_λσ ERI[μ,λ,ν,σ]   →  'ls,mlns->mn'

    The ERI tensor produced by build_eri_tensor() is fully symmetric,
    so no manual symmetry exploitation is needed here; every contraction
    index hits the correct pre-filled element.
    """
    H_core = np.asarray(H_core, dtype=float)
    P      = np.asarray(P,      dtype=float)
    ERI    = np.asarray(ERI,    dtype=float)

    n = H_core.shape[0]
    if H_core.shape != (n, n):
        raise ValueError(f"H_core must be square, got {H_core.shape}")
    if P.shape != (n, n):
        raise ValueError(
            f"P shape {P.shape} inconsistent with H_core shape {H_core.shape}"
        )
    if ERI.shape != (n, n, n, n):
        raise ValueError(
            f"ERI shape {ERI.shape} inconsistent with n={n}"
        )

    # Coulomb: J_μν = Σ_λσ P_λσ (μν|λσ)
    J = np.einsum('ls,mnls->mn', P, ERI)

    # Exchange: K_μν = Σ_λσ P_λσ (μλ|νσ)
    K = np.einsum('ls,mlns->mn', P, ERI)

    G = J - 0.5 * K
    return H_core + G


def electronic_energy(
    P: np.ndarray,
    H_core: np.ndarray,
    F: np.ndarray,
) -> float:
    """
    Compute the RHF electronic energy E_elec = ½ Tr[P (H_core + F)].

    The factor of ½ prevents double-counting the electron–electron
    repulsion: the G(P) term in F already includes contributions from
    both electrons in each pair, so summing over all P elements again
    would count each pair twice.

    Parameters
    ----------
    P : ndarray, shape (n, n)
        Density matrix.
    H_core : ndarray, shape (n, n)
        Core Hamiltonian matrix.
    F : ndarray, shape (n, n)
        Fock matrix corresponding to P.

    Returns
    -------
    float
        Electronic energy in hartree.

    Notes
    -----
    Equivalently:  E_elec = ½ Tr[P H_core] + ½ Tr[P F]
                           = E_one + ½ E_two

    where E_one is the one-electron energy and E_two the two-electron
    contribution.  This decomposition is useful for debugging.

    The einsum 'ij,ij->' is element-wise multiply and sum — equivalent
    to np.trace(P @ M) but avoids forming the full matrix product.
    """
    return 0.5 * float(np.einsum('ij,ij->', P, H_core + F))
