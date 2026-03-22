import numpy as np
from ..linalg import double_factorial as _double_factorial


def norm_primitive(alpha: float, angular: tuple) -> float:
    """Normalization constant for a primitive Gaussian."""
    lx, ly, lz = angular
    L = lx + ly + lz
    prefactor = (2 * alpha / np.pi)**0.75 * (4 * alpha)**(L / 2)
    denom = np.sqrt(
        _double_factorial(2 * lx - 1) *
        _double_factorial(2 * ly - 1) *
        _double_factorial(2 * lz - 1)
    )
    return prefactor / denom


def overlap_1d(i: int, j: int, Ax: float, Bx: float,
               alpha: float, beta: float) -> float:
    """
    1D overlap integral between Gaussians with angular momenta i, j
    centered at Ax, Bx with exponents alpha, beta.
    Uses Obara-Saika recursion.
    """
    p = alpha + beta
    Px = (alpha * Ax + beta * Bx) / p
    XPA = Px - Ax
    XPB = Px - Bx
    XAB = Ax - Bx

    # Build table S[i][j] bottom-up
    S = np.zeros((i + 1, j + 1))
    S[0, 0] = np.sqrt(np.pi / p) * np.exp(-alpha * beta / p * XAB**2)

    # Increment i (vertical recurrence)
    for ii in range(1, i + 1):
        S[ii, 0] = XPA * S[ii - 1, 0]
        if ii > 1:
            S[ii, 0] += (ii - 1) / (2 * p) * S[ii - 2, 0]

    # Increment j (horizontal recurrence)
    for jj in range(1, j + 1):
        for ii in range(0, i + 1):
            S[ii, jj] = XPB * S[ii, jj - 1]
            if ii > 0:
                S[ii, jj] += ii / (2 * p) * S[ii - 1, jj - 1]
            if jj > 1:
                S[ii, jj] += (jj - 1) / (2 * p) * S[ii, jj - 2]

    return S[i, j]


def overlap_primitive(a: tuple, b: tuple, alpha: float, beta: float,
                      A: np.ndarray, B: np.ndarray) -> float:
    """
    Overlap integral between two primitive Gaussians.
    a, b: angular momentum tuples e.g. (1,0,0) for px
    A, B: centers as numpy arrays
    """
    sx = overlap_1d(a[0], b[0], A[0], B[0], alpha, beta)
    sy = overlap_1d(a[1], b[1], A[1], B[1], alpha, beta)
    sz = overlap_1d(a[2], b[2], A[2], B[2], alpha, beta)
    return sx * sy * sz


def overlap_contracted(shell_a: dict, shell_b: dict) -> float:
    """
    Overlap between two contracted basis functions.
    Each shell is a dict with keys:
      'center':       np.array([x, y, z])
      'angular':      tuple (lx, ly, lz)
      'exponents':    list of floats
      'coefficients': list of floats (contraction coefficients)
    """
    result = 0.0
    A, a = shell_a['center'], shell_a['angular']
    B, b = shell_b['center'], shell_b['angular']

    for alpha, ca in zip(shell_a['exponents'], shell_a['coefficients']):
        for beta, cb in zip(shell_b['exponents'], shell_b['coefficients']):
            Na = norm_primitive(alpha, a)
            Nb = norm_primitive(beta, b)
            result += Na * Nb * ca * cb * overlap_primitive(a, b, alpha, beta, A, B)

    return result


def build_overlap_matrix(basis: list) -> np.ndarray:
    """
    Build the full overlap matrix S for a list of contracted basis functions.

    Exploits Hermitian symmetry: only the upper triangle is computed and
    the result is reflected, halving the number of integral evaluations.
    """
    n = len(basis)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            S[i, j] = overlap_contracted(basis[i], basis[j])
            S[j, i] = S[i, j]
    return S