"""
qchem/linalg.py — shared mathematical utilities

Contents
--------
double_factorial(n)
    The combinatorial function n!! = n·(n-2)·…·1, with the convention
    (-1)!! = 1 and 0!! = 1.  Used by norm_primitive and anywhere else
    Gaussian normalisation constants appear.

symmetric_orthogonalization(S)
    Löwdin's S^{-1/2} via eigendecomposition.  Produces an orthonormal
    basis that stays as close as possible to the original AO basis.
    This is the standard pre-conditioner for the HF / KS eigenvalue step.

canonical_orthogonalization(S, tol=1e-8)
    Alternative to Löwdin's scheme that discards near-linearly-dependent
    basis functions (eigenvalues of S below tol).  Useful for large or
    diffuse basis sets where S is nearly singular.

solve_generalized_eigenvalue(F, S)
    Solve  F C = S C ε  by transforming to an orthonormal basis using
    symmetric_orthogonalization, diagonalising, then back-transforming.
    Returns eigenvalues (ascending) and MO coefficient matrix C in the
    original AO basis.

Notes
-----
All routines operate on real-valued arrays (RHF / RKS convention).
The generalised eigenvalue solver will need extension for complex
integrals (GHF, relativistic, magnetic-field calculations) — this is
why magnetic.py is kept separate.
"""

import numpy as np
from scipy.special import factorial2 as _scipy_factorial2


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------

def double_factorial(n: int) -> float:
    """
    Double factorial with the convention that (-1)!! = 0!! = 1.

    n!! = n · (n-2) · (n-4) · … down to 1 (odd n) or 2 (even n).

    Parameters
    ----------
    n : int
        Non-negative integer, or -1.

    Returns
    -------
    float
        Value of n!!.

    Examples
    --------
    >>> double_factorial(-1)
    1.0
    >>> double_factorial(0)
    1.0
    >>> double_factorial(5)
    15.0   # 5·3·1
    >>> double_factorial(6)
    48.0   # 6·4·2
    """
    if n <= 0:
        return 1.0
    return float(_scipy_factorial2(n))


# ---------------------------------------------------------------------------
# Orthogonalization
# ---------------------------------------------------------------------------

def symmetric_orthogonalization(S: np.ndarray) -> np.ndarray:
    """
    Compute the Löwdin symmetric orthogonalization matrix X = S^{-1/2}.

    The transformation X maps the original (non-orthogonal) AO basis to
    an orthonormal basis:

        X^T S X = I

    while minimising the root-mean-square deviation from the original
    basis functions.

    Parameters
    ----------
    S : ndarray, shape (n, n)
        Symmetric, positive-definite overlap matrix.

    Returns
    -------
    X : ndarray, shape (n, n)
        The matrix S^{-1/2}.

    Raises
    ------
    np.linalg.LinAlgError
        If S is not positive definite (smallest eigenvalue ≤ 0).

    Notes
    -----
    Uses eigendecomposition S = U Λ U^T, so X = U Λ^{-1/2} U^T.
    For nearly-singular S (diffuse basis sets), prefer
    canonical_orthogonalization instead.
    """
    eigenvalues, U = np.linalg.eigh(S)
    if eigenvalues[0] <= 0.0:
        raise np.linalg.LinAlgError(
            f"Overlap matrix is not positive definite "
            f"(smallest eigenvalue = {eigenvalues[0]:.3e}). "
            "Use canonical_orthogonalization with a suitable tolerance."
        )
    X = U @ np.diag(eigenvalues ** -0.5) @ U.T
    return X


def canonical_orthogonalization(S: np.ndarray,
                                 tol: float = 1e-8) -> np.ndarray:
    """
    Canonical orthogonalization that discards near-linearly-dependent
    basis functions.

    Decomposes S = U Λ U^T and retains only eigenvectors whose eigenvalue
    exceeds tol, returning a rectangular transformation matrix X of shape
    (n, n_kept) such that X^T S X = I_{n_kept}.

    Parameters
    ----------
    S   : ndarray, shape (n, n)
        Symmetric overlap matrix.
    tol : float
        Eigenvalue threshold below which basis functions are dropped.

    Returns
    -------
    X : ndarray, shape (n, n_kept)
        Rectangular orthogonalization matrix.  n_kept ≤ n.

    Notes
    -----
    The MO coefficient matrix returned by solve_generalized_eigenvalue
    is always in the original AO basis regardless of which
    orthogonalization method was used internally.
    """
    eigenvalues, U = np.linalg.eigh(S)
    mask = eigenvalues > tol
    n_dropped = np.sum(~mask)
    if n_dropped:
        import warnings
        warnings.warn(
            f"canonical_orthogonalization: dropping {n_dropped} basis "
            f"function(s) with eigenvalue(s) below {tol:.1e}.",
            stacklevel=2,
        )
    kept = eigenvalues[mask]
    X = U[:, mask] @ np.diag(kept ** -0.5)
    return X


# ---------------------------------------------------------------------------
# Generalised eigenvalue problem
# ---------------------------------------------------------------------------

def solve_generalized_eigenvalue(
    F: np.ndarray,
    S: np.ndarray,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalised eigenvalue problem  F C = S C ε.

    This is the central linear-algebra step of every SCF iteration:
    given the Fock matrix F and the overlap matrix S, find the orbital
    energies ε and the MO coefficient matrix C (columns = MOs in the AO
    basis, ordered by ascending energy).

    Strategy
    --------
    1. Compute X = S^{-1/2} (symmetric or canonical, depending on tol).
    2. Form the orthonormal Fock matrix  F' = X^T F X.
    3. Diagonalise F' (standard symmetric eigenvalue problem).
    4. Back-transform: C = X C'.

    Parameters
    ----------
    F   : ndarray, shape (n, n)
        Fock (or Kohn-Sham) matrix in the AO basis, real symmetric.
    S   : ndarray, shape (n, n)
        Overlap matrix, real symmetric positive definite.
    tol : float
        Eigenvalue threshold passed to canonical_orthogonalization when
        S is nearly singular.  Ignored when S is well-conditioned.

    Returns
    -------
    epsilon : ndarray, shape (n,) or (n_kept,)
        Orbital energies in ascending order (Hartree).
    C : ndarray, shape (n, n) or (n, n_kept)
        MO coefficients in the AO basis (column i = MO i).

    Notes
    -----
    The function attempts symmetric_orthogonalization first (cheaper,
    preserves the full basis).  If S is not positive definite it falls
    back to canonical_orthogonalization automatically.
    """
    try:
        X = symmetric_orthogonalization(S)
    except np.linalg.LinAlgError:
        X = canonical_orthogonalization(S, tol=tol)

    F_prime = X.T @ F @ X
    epsilon, C_prime = np.linalg.eigh(F_prime)
    C = X @ C_prime
    return epsilon, C