import numpy as np
from itertools import product
from .overlap import overlap_1d, norm_primitive


def kinetic_1d(i: int, j: int, Ax: float, Bx: float,
               alpha: float, beta: float) -> float:
    """
    1D kinetic energy integral via Obara-Saika.

    T(i,j) = beta*(2j+1)*S(i,j) - 2*beta²*S(i,j+2) - j*(j-1)/2*S(i,j-2)

    All S calls use the same centers and exponents as the overlap
    they're paired with — the angular momentum shifts are the only
    thing that changes.
    """
    # Centre term — always present
    result = beta * (2 * j + 1) * overlap_1d(i, j, Ax, Bx, alpha, beta)

    # Upper shift — always present
    result -= 2 * beta**2 * overlap_1d(i, j + 2, Ax, Bx, alpha, beta)

    # Lower shift — only exists when j >= 2
    if j >= 2:
        result -= 0.5 * j * (j - 1) * overlap_1d(i, j - 2, Ax, Bx, alpha, beta)

    return result


def kinetic_primitive(a: tuple, b: tuple, alpha: float, beta: float,
                      A: np.ndarray, B: np.ndarray) -> float:
    """
    Kinetic energy integral between two primitive Gaussians.
    Uses the 3D separation:
        T = Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
    """
    # Overlap integrals for each dimension
    Sx = overlap_1d(a[0], b[0], A[0], B[0], alpha, beta)
    Sy = overlap_1d(a[1], b[1], A[1], B[1], alpha, beta)
    Sz = overlap_1d(a[2], b[2], A[2], B[2], alpha, beta)

    # Kinetic integrals for each dimension
    Tx = kinetic_1d(a[0], b[0], A[0], B[0], alpha, beta)
    Ty = kinetic_1d(a[1], b[1], A[1], B[1], alpha, beta)
    Tz = kinetic_1d(a[2], b[2], A[2], B[2], alpha, beta)

    return Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz


def kinetic_contracted(shell_a: dict, shell_b: dict) -> float:
    """
    Kinetic energy integral between two contracted basis functions.
    Shell format matches overlap.py exactly.
    """
    result = 0.0
    A, a = shell_a['center'], shell_a['angular']
    B, b = shell_b['center'], shell_b['angular']

    for alpha, ca in zip(shell_a['exponents'], shell_a['coefficients']):
        for beta, cb in zip(shell_b['exponents'], shell_b['coefficients']):
            Na = norm_primitive(alpha, a)
            Nb = norm_primitive(beta, b)
            result += Na * Nb * ca * cb * kinetic_primitive(a, b, alpha, beta, A, B)

    return result


def build_kinetic_matrix(basis: list) -> np.ndarray:
    """
    Build the full kinetic energy matrix T for a list of contracted
    basis functions. Exploits Hermitian symmetry.
    """
    n = len(basis)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            T[i, j] = kinetic_contracted(basis[i], basis[j])
            T[j, i] = T[i, j]
    return T