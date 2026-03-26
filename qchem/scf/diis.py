"""
qchem/scf/diis.py — DIIS convergence accelerator for the RHF SCF loop

Overview
--------
DIIS (Direct Inversion in the Iterative Subspace) is a convergence
accelerator introduced by Pulay (1980).  Instead of feeding the Fock
matrix from the current iteration directly into the next diagonalization,
DIIS builds an optimal linear combination of the last N Fock matrices —
the combination that minimises the norm of the residual error in the
subspace spanned by those iterations.

Error vector
------------
For RHF the natural residual is the orbital-gradient commutator in the
AO basis:

    e = F P S - S P F

At a converged solution F and P commute with respect to S, so e = 0
exactly.  DIIS stores a history of (F_i, e_i) pairs and at each step
seeks coefficients c_i such that

    || sum_i c_i e_i || is minimised,  subject to  sum_i c_i = 1.

The constraint prevents the trivial solution c = 0 and ensures genuine
extrapolation toward convergence rather than collapse.

Linear system
-------------
The optimality conditions (Lagrangian with multiplier λ) yield the
augmented (n+1) × (n+1) system:

    [ B   -1 ] [ c ]   [ 0 ]
    [-1^T  0 ] [ λ ] = [-1 ]

where B_ij = <e_i, e_j> = e_i · e_j.  The solution c[:n] gives the
extrapolation weights; λ is discarded.  The extrapolated Fock matrix is

    F* = sum_i c_i F_i.

This F* is then diagonalized in the usual way to produce new MO
coefficients and a new density matrix.

History cap and conditioning
----------------------------
The B matrix can become ill-conditioned as old, nearly-collinear error
vectors accumulate.  The standard remedy is a rolling window: when the
history reaches max_vec entries, the oldest (F, e) pair is evicted before
the new one is added.  Values of max_vec in the range 6–8 are typical
and sufficient for closed-shell molecules.

If the linear system is singular despite the window (e.g. very early
iterations with only one unique error direction), the method falls back
to returning the most recently stored Fock matrix unchanged.

Classes
-------
DIISAccelerator
    Stateful accumulator and extrapolator.  The SCF driver creates one
    instance per calculation and calls .push() then .extrapolate() every
    iteration.

References
----------
Pulay, P. Chem. Phys. Lett. 73, 393-398 (1980). — original DIIS paper.
Pulay, P. J. Comput. Chem. 3, 556-560 (1982). — DIIS for SCF.
Szabo & Ostlund, "Modern Quantum Chemistry", Dover (1996), pp. 146-150.
"""

from __future__ import annotations

import numpy as np


class DIISAccelerator:
    """
    DIIS convergence accelerator for the RHF SCF loop.

    Usage
    -----
    Create one instance before the SCF loop.  Inside the loop, after
    building the plain Fock matrix F and computing the error vector e,
    call push() and then extrapolate() to obtain the DIIS-improved Fock
    matrix to diagonalize::

        diis = DIISAccelerator(max_vec=8)

        for iteration in range(max_iter):
            F  = fock_matrix(H_core, P, ERI)
            e  = F @ P @ S - S @ P @ F       # orbital gradient
            diis.push(F, e)
            F_diis = diis.extrapolate()       # use this for diagonalization
            ...

    Parameters
    ----------
    max_vec : int, optional
        Maximum number of (F, e) pairs to retain in the rolling history.
        When the limit is reached, the oldest pair is discarded before
        the new one is stored.  Typical values: 6–8.  Default is 8.

    Raises
    ------
    ValueError
        If max_vec is less than 1.
    """

    def __init__(self, max_vec: int = 8) -> None:
        if max_vec < 1:
            raise ValueError(f"max_vec must be >= 1, got {max_vec}")
        self.max_vec = max_vec
        self._fock_history:  list[np.ndarray] = []
        self._error_history: list[np.ndarray] = []  # stored as 1-D vectors

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(self, F: np.ndarray, e: np.ndarray) -> None:
        """
        Add a new (Fock matrix, error vector) pair to the history.

        If the history already contains max_vec entries, the oldest pair
        is evicted before the new one is stored (rolling window).

        Parameters
        ----------
        F : ndarray, shape (n, n)
            Fock matrix for the current SCF iteration, as returned by
            fock.fock_matrix().
        e : ndarray, shape (n, n) or (n*n,)
            Error (orbital gradient) for this iteration:
                e = F P S - S P F
            May be passed as a 2-D matrix or a pre-flattened 1-D vector;
            it is stored internally as 1-D.

        Raises
        ------
        ValueError
            If F is not a 2-D square array, or if the number of elements
            in e does not equal n*n where n is inferred from F.
        ValueError
            If, after the first push, the shape of F is inconsistent with
            the previously stored shape.
        """
        F = np.asarray(F, dtype=float)
        e = np.asarray(e, dtype=float)

        if F.ndim != 2 or F.shape[0] != F.shape[1]:
            raise ValueError(
                f"F must be a 2-D square array, got shape {F.shape}"
            )
        n = F.shape[0]

        e_flat = e.ravel()
        if e_flat.size != n * n:
            raise ValueError(
                f"e must have n*n = {n*n} elements for an n={n} basis, "
                f"got {e_flat.size}"
            )

        if self._fock_history:
            stored_shape = self._fock_history[0].shape
            if F.shape != stored_shape:
                raise ValueError(
                    f"F shape {F.shape} is inconsistent with previously "
                    f"stored shape {stored_shape}"
                )

        # Evict oldest pair when at capacity
        if len(self._fock_history) >= self.max_vec:
            self._fock_history.pop(0)
            self._error_history.pop(0)

        self._fock_history.append(F.copy())
        self._error_history.append(e_flat.copy())

    def extrapolate(self) -> np.ndarray:
        """
        Build the DIIS-extrapolated Fock matrix F* = sum_i c_i F_i.

        The coefficients c_i minimise || sum_i c_i e_i || subject to
        sum_i c_i = 1, found by solving the augmented DIIS linear system.

        If only one vector is stored, the constraint forces c_1 = 1, so
        F_1 is returned directly without solving the system.

        If the linear system is singular (ill-conditioned B matrix), the
        method falls back to returning the most recently stored Fock
        matrix.  This is a safe, if unaccelerated, choice.

        Returns
        -------
        F_star : ndarray, shape (n, n)
            DIIS-extrapolated Fock matrix, ready to pass to
            linalg.solve_generalized_eigenvalue().

        Raises
        ------
        RuntimeError
            If called before any push().
        """
        if not self._fock_history:
            raise RuntimeError(
                "extrapolate() called before any push(); "
                "call push(F, e) at least once first."
            )

        if len(self._fock_history) == 1:
            return self._fock_history[0].copy()

        B = self._build_B()
        c = self._solve(B)

        F_star = np.einsum('i,ijk->jk', c, np.array(self._fock_history))
        return F_star

    def reset(self) -> None:
        """
        Clear all stored history.

        Useful when restarting a calculation with a new geometry or basis
        without constructing a new DIISAccelerator instance.
        """
        self._fock_history.clear()
        self._error_history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_B(self) -> np.ndarray:
        """
        Assemble the DIIS B matrix from stored error vectors.

        B_ij = <e_i | e_j> = e_i · e_j

        The matrix is symmetric and positive semi-definite (exactly
        positive definite if the error vectors are linearly independent).

        Returns
        -------
        B : ndarray, shape (n_vec, n_vec)
        """
        # Stack rows: shape (n_vec, n_basis^2)
        E = np.array(self._error_history)   # (n_vec, n²)
        return E @ E.T                      # (n_vec, n_vec), symmetric

    def _solve(self, B: np.ndarray) -> np.ndarray:
        """
        Solve the augmented DIIS linear system for the extrapolation
        coefficients.

        Constructs and solves:

            [ B   -1 ] [ c ]   [ 0 ]
            [-1^T  0 ] [ λ ] = [-1 ]

        Returns c[:n_vec].  If np.linalg.solve raises a LinAlgError
        (singular or nearly singular system), falls back to a uniform
        weight vector (all entries equal, summing to 1) so that the
        extrapolated Fock matrix is a simple average of the stored ones.
        This is safe but unaccelerated; it should not occur under normal
        circumstances.

        Parameters
        ----------
        B : ndarray, shape (n_vec, n_vec)

        Returns
        -------
        c : ndarray, shape (n_vec,)
            Extrapolation coefficients, guaranteed to sum to 1.
        """
        n = B.shape[0]

        # Build (n+1) × (n+1) augmented system
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = B
        A[:n, n]  = -1.0
        A[n, :n]  = -1.0
        # A[n, n] stays 0

        b = np.zeros(n + 1)
        b[n] = -1.0

        try:
            solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback: uniform weights — equivalent to a simple average
            return np.full(n, 1.0 / n)

        return solution[:n]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of (F, e) pairs currently in the history."""
        return len(self._fock_history)

    def __repr__(self) -> str:
        return (
            f"DIISAccelerator(max_vec={self.max_vec}, "
            f"stored={len(self)})"
        )
