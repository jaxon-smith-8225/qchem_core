"""
Nuclear attraction integrals using the Obara-Saika scheme.

For each nucleus C with charge Z_C, the single-centre contribution is:

    V_ij^C = -Z_C <phi_i | 1/|r - R_C| | phi_j>

and the full nuclear attraction matrix is V_ij = sum_C V_ij^C.

Unlike the overlap and kinetic integrals, the Coulomb operator 1/|r-R_C|
does not factorise into x, y, z parts, so the clean 3D product trick
no longer applies.  The integral is instead handled by the Fourier
representation of 1/r, which introduces the Boys function F_m(x).  Higher
angular momentum is built up in two stages:

Stage 1 — Vertical Recurrence Relation (VRR)
    All angular momentum is accumulated on centre A, producing auxiliary
    integrals [a|C]^(m) parametrised by a Boys index m.  The recurrence is:

        [a+1_i|C]^(m) = (P-A)_i [a|C]^(m)  -  (P-C)_i [a|C]^(m+1)
                       + a_i/(2p) ([a-1_i|C]^(m) - [a-1_i|C]^(m+1))

    seeded by [0|C]^(m) = (2π/p) exp(-αβ/p |AB|²) F_m(p |PC|²).

Stage 2 — Horizontal Recurrence Relation (HRR)
    Angular momentum is transferred from A to B without introducing any
    further Boys-function evaluations:

        [a | b+1_i] = [a+1_i | b]  +  (A-B)_i [a | b]

References
----------
Obara & Saika, J. Chem. Phys. 84, 3963 (1986).
Helgaker, Jørgensen & Olsen, "Molecular Electronic-Structure Theory",
    Wiley (2000), Chapter 9.
"""

import numpy as np
from .boys import boys_array
from .overlap import norm_primitive


# ---------------------------------------------------------------------------
# Stage 1: Vertical Recurrence Relation
# ---------------------------------------------------------------------------

def _vrr(a_max: tuple, p: float, PA: np.ndarray, PC: np.ndarray,
         K: float, F: np.ndarray) -> np.ndarray:
    """
    Build the 4-D auxiliary table T[ax, ay, az, m].

    Parameters
    ----------
    a_max : (ax_max, ay_max, az_max)
        Upper limits for each angular momentum component on A.
        For a primitive (a on A, b on B), pass (a+b) component-wise so
        that T covers everything the HRR will subsequently need.
    p     : combined exponent α + β
    PA    : P − A  (Gaussian product centre minus A)
    PC    : P − C  (Gaussian product centre minus nuclear position)
    K     : prefactor  exp(−αβ/p |AB|²) · 2π/p
    F     : Boys function array  F_m(p |PC|²),  length >= sum(a_max) + 1

    Returns
    -------
    T : ndarray, shape (ax_max+1, ay_max+1, az_max+1, L+1)
        T[ax, ay, az, m] = [ax, ay, az | C]^(m)
        Only entries with ax+ay+az+m <= L are filled; the rest stay zero.
    """
    ax_max, ay_max, az_max = a_max
    L = ax_max + ay_max + az_max

    T = np.zeros((ax_max + 1, ay_max + 1, az_max + 1, L + 1))

    # Base case: [0|C]^(m) = K * F_m
    for m in range(L + 1):
        T[0, 0, 0, m] = K * F[m]

    # Increment x: build T[ax, 0, 0, m]
    for ax in range(1, ax_max + 1):
        for m in range(L - ax + 1):
            T[ax, 0, 0, m] = (PA[0] * T[ax-1, 0, 0, m]
                              - PC[0] * T[ax-1, 0, 0, m+1])
            if ax >= 2:
                T[ax, 0, 0, m] += ((ax - 1) / (2 * p)
                                   * (T[ax-2, 0, 0, m] - T[ax-2, 0, 0, m+1]))

    # Increment y: build T[ax, ay, 0, m] for all filled ax
    for ay in range(1, ay_max + 1):
        for ax in range(ax_max + 1):
            for m in range(L - ax - ay + 1):
                T[ax, ay, 0, m] = (PA[1] * T[ax, ay-1, 0, m]
                                   - PC[1] * T[ax, ay-1, 0, m+1])
                if ay >= 2:
                    T[ax, ay, 0, m] += ((ay - 1) / (2 * p)
                                        * (T[ax, ay-2, 0, m] - T[ax, ay-2, 0, m+1]))

    # Increment z: build T[ax, ay, az, m] for all filled ax, ay
    for az in range(1, az_max + 1):
        for ax in range(ax_max + 1):
            for ay in range(ay_max + 1):
                for m in range(L - ax - ay - az + 1):
                    T[ax, ay, az, m] = (PA[2] * T[ax, ay, az-1, m]
                                        - PC[2] * T[ax, ay, az-1, m+1])
                    if az >= 2:
                        T[ax, ay, az, m] += ((az - 1) / (2 * p)
                                             * (T[ax, ay, az-2, m]
                                                - T[ax, ay, az-2, m+1]))

    return T


# ---------------------------------------------------------------------------
# Stage 2: Horizontal Recurrence Relation
# ---------------------------------------------------------------------------

def _hrr(T_vrr: np.ndarray, a: tuple, b: tuple,
         AB: np.ndarray) -> float:
    """
    Transfer angular momentum from A to B using the HRR.

    The HRR uses only the m=0 slice of T_vrr and requires no further
    Boys-function evaluations.

        [a | b+1_i] = [a+1_i | b]  +  (A-B)_i [a | b]

    Parameters
    ----------
    T_vrr : output of _vrr, shape (a[i]+b[i]+1, ..., L+1)
    a     : target angular momentum on A
    b     : target angular momentum on B
    AB    : A − B

    Returns
    -------
    float : the integral [a | b]
    """
    ax_max = a[0] + b[0]
    ay_max = a[1] + b[1]
    az_max = a[2] + b[2]

    # 6-D table H[ax, ay, az, bx, by, bz]
    H = np.zeros((ax_max + 1, ay_max + 1, az_max + 1,
                  b[0] + 1,   b[1] + 1,   b[2] + 1))

    # Seed from the m=0 slice of the VRR table
    for ax in range(ax_max + 1):
        for ay in range(ay_max + 1):
            for az in range(az_max + 1):
                H[ax, ay, az, 0, 0, 0] = T_vrr[ax, ay, az, 0]

    # Transfer angular momentum to bx
    # Bounds: ax runs 0..a[0]; ax+1 <= a[0]+1 <= a[0]+b[0] = ax_max  (b[0]>=1)
    for bx in range(1, b[0] + 1):
        for ax in range(a[0] + 1):
            for ay in range(ay_max + 1):
                for az in range(az_max + 1):
                    H[ax, ay, az, bx, 0, 0] = (
                        H[ax + 1, ay, az, bx - 1, 0, 0]
                        + AB[0] * H[ax, ay, az, bx - 1, 0, 0]
                    )

    # Transfer angular momentum to by
    # After the bx pass, H[ax, ay, az, bx, 0, 0] is valid for all bx.
    # Bounds: ay runs 0..a[1]; ay+1 <= a[1]+1 <= a[1]+b[1] = ay_max  (b[1]>=1)
    for by in range(1, b[1] + 1):
        for bx in range(b[0] + 1):
            for ax in range(a[0] + 1):
                for ay in range(a[1] + 1):
                    for az in range(az_max + 1):
                        H[ax, ay, az, bx, by, 0] = (
                            H[ax, ay + 1, az, bx, by - 1, 0]
                            + AB[1] * H[ax, ay, az, bx, by - 1, 0]
                        )

    # Transfer angular momentum to bz
    # Bounds: az runs 0..a[2]; az+1 <= a[2]+1 <= a[2]+b[2] = az_max  (b[2]>=1)
    for bz in range(1, b[2] + 1):
        for by in range(b[1] + 1):
            for bx in range(b[0] + 1):
                for ax in range(a[0] + 1):
                    for ay in range(a[1] + 1):
                        for az in range(a[2] + 1):
                            H[ax, ay, az, bx, by, bz] = (
                                H[ax, ay, az + 1, bx, by, bz - 1]
                                + AB[2] * H[ax, ay, az, bx, by, bz - 1]
                            )

    return H[a[0], a[1], a[2], b[0], b[1], b[2]]


# ---------------------------------------------------------------------------
# Primitive integral (single nucleus, no normalisation)
# ---------------------------------------------------------------------------

def nuclear_primitive_one_center(a: tuple, b: tuple,
                                 alpha: float, beta: float,
                                 A: np.ndarray, B: np.ndarray,
                                 C: np.ndarray) -> float:
    """
    Unnormalised nuclear attraction integral for one nucleus at C.

        <g_a | 1/|r - C| | g_b>     (positive; the -Z factor is applied later)

    Parameters
    ----------
    a, b   : angular momentum tuples, e.g. (1, 0, 0) for px
    alpha  : exponent of g_a
    beta   : exponent of g_b
    A, B   : Gaussian centres as numpy arrays
    C      : nuclear position as numpy array
    """
    p  = alpha + beta
    P  = (alpha * A + beta * B) / p
    PA = P - A
    PC = P - C
    AB = A - B

    L = sum(a) + sum(b)
    K = np.exp(-alpha * beta / p * float(np.dot(AB, AB))) * 2.0 * np.pi / p
    F = boys_array(L, p * float(np.dot(PC, PC)))

    a_max = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    T = _vrr(a_max, p, PA, PC, K, F)

    if sum(b) == 0:
        # No angular momentum on B: VRR result is final
        return float(T[a[0], a[1], a[2], 0])
    else:
        return float(_hrr(T, a, b, AB))


# ---------------------------------------------------------------------------
# Contracted integral
# ---------------------------------------------------------------------------

def nuclear_contracted(shell_a: dict, shell_b: dict,
                       charges: list, coords: list) -> float:
    """
    Nuclear attraction integral between two contracted basis functions.

    Parameters
    ----------
    shell_a, shell_b : shell dicts with keys
        'center'       : np.array([x, y, z])
        'angular'      : tuple (lx, ly, lz)
        'exponents'    : list of floats
        'coefficients' : list of floats
    charges : sequence of nuclear charges Z_C  (positive integers)
    coords  : sequence of nuclear positions, each array-like length 3

    Returns
    -------
    float : sum_C  -Z_C <phi_a | 1/|r - R_C| | phi_b>
    """
    result = 0.0
    A, a = shell_a['center'], shell_a['angular']
    B, b = shell_b['center'], shell_b['angular']
    coords_arr = [np.asarray(C, dtype=float) for C in coords]

    for alpha, ca in zip(shell_a['exponents'], shell_a['coefficients']):
        for beta, cb in zip(shell_b['exponents'], shell_b['coefficients']):
            Na = norm_primitive(alpha, a)
            Nb = norm_primitive(beta,  b)
            v_prim = sum(
                -Z * nuclear_primitive_one_center(a, b, alpha, beta, A, B, C)
                for Z, C in zip(charges, coords_arr)
            )
            result += Na * Nb * ca * cb * v_prim

    return result


# ---------------------------------------------------------------------------
# Full matrix
# ---------------------------------------------------------------------------

def build_nuclear_matrix(basis: list, charges: list,
                         coords: list) -> np.ndarray:
    """
    Build the full nuclear attraction matrix V for a list of contracted
    basis functions.  Exploits Hermitian symmetry.

    Parameters
    ----------
    basis   : list of shell dicts (same format as overlap.py / kinetic.py)
    charges : sequence of nuclear charges
    coords  : sequence of nuclear positions

    Returns
    -------
    V : ndarray, shape (n, n), symmetric
    """
    n = len(basis)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            V[i, j] = nuclear_contracted(basis[i], basis[j], charges, coords)
            V[j, i] = V[i, j]
    return V