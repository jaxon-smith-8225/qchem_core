"""
Electron repulsion integrals (ERI) — Obara-Saika scheme.

The four-centre two-electron integral in chemist's notation:

    (ab|cd) = ∬ φ_a(r₁) φ_b(r₁) (1/|r₁−r₂|) φ_c(r₂) φ_d(r₂) dr₁ dr₂

is evaluated in three stages.

Stage 1 — Vertical Recurrence Relation (VRR)
    Angular momentum is built up on centres A and C simultaneously,
    producing auxiliary integrals [a|c]^(m).  The recurrences are:

        [a+1ᵢ|c]^(m) = (P−A)ᵢ  [a|c]^(m)
                       − (ζ/p)(W−P)ᵢ  [a|c]^(m+1)
                       +  aᵢ/(2p)   ([a−1ᵢ|c]^(m)  − (ζ/p)[a−1ᵢ|c]^(m+1))
                       +  cᵢ/(2(p+q)) [a|c−1ᵢ]^(m+1)

        [a|c+1ᵢ]^(m) = (Q−C)ᵢ  [a|c]^(m)
                       − (ζ/q)(W−Q)ᵢ  [a|c]^(m+1)
                       +  cᵢ/(2q)   ([a|c−1ᵢ]^(m)  − (ζ/q)[a|c−1ᵢ]^(m+1))
                       +  aᵢ/(2(p+q)) [a−1ᵢ|c]^(m+1)

    seeded by  [0|0]^(m) = prefactor · F_m(T), where

        p = α+β,  q = γ+δ,  ζ = pq/(p+q),
        P = (αA+βB)/p,  Q = (γC+δD)/q,  W = (pP+qQ)/(p+q),
        T = ζ|P−Q|²,
        prefactor = 2π^(5/2) / (p·q·√(p+q)) · exp(−αβ/p|AB|²) · exp(−γδ/q|CD|²).

Stage 2 — Bra Horizontal Recurrence Relation (HRR)
    Angular momentum is transferred from A to B using the m=0 slice:

        [a | b+1ᵢ | c0] = [a+1ᵢ | b | c0]  +  (A−B)ᵢ [a | b | c0]

    The inner loop for each bx level runs ax from 0 to (a[0]+b[0]−bx),
    ensuring that the [ax+1, ..., bx−1, ...] entry from the preceding
    step is always available.  This correct bound is essential for
    b-functions with angular momentum ≥ 2 in any component.

Stage 3 — Ket Horizontal Recurrence Relation (HRR)
    Angular momentum is transferred from C to D by the identical rule:

        [ab | c | d+1ᵢ] = [ab | c+1ᵢ | d]  +  (C−D)ᵢ [ab | c | d]

References
----------
Obara & Saika, J. Chem. Phys. 84, 3963 (1986).
Helgaker, Jørgensen & Olsen, "Molecular Electronic-Structure Theory",
    Wiley (2000), §9.2.
Szabo & Ostlund, "Modern Quantum Chemistry", Dover (1989), Appendix A.
"""

import numpy as np
from .boys import boys_array
from .overlap import norm_primitive


# ---------------------------------------------------------------------------
# Stage 1 — Vertical Recurrence Relation
# ---------------------------------------------------------------------------

def _eri_vrr(a_max: tuple, c_max: tuple,
             p: float, q: float, zeta: float,
             PA: np.ndarray, WP: np.ndarray,
             QC: np.ndarray, WQ: np.ndarray,
             prefactor: float, F: np.ndarray) -> np.ndarray:
    """
    Build the 7-D auxiliary ERI table  R[ax, ay, az, cx, cy, cz, m].

    Angular momentum is grown in two passes:

    Pass A — build all [a | 0]^(m) by incrementing each component of a
             while holding c = 0.  The cross term cᵢ/(2(p+q)) vanishes,
             so these rows decouple from the ket.

    Pass B — build all [a | c]^(m) for c > 0 by incrementing each
             component of c.  The cross term aᵢ/(2(p+q)) references
             [a−1ᵢ | c−1ᵢ]^(m+1), which is already filled from Pass A
             or from a lower c-step.

    Within each component the m-loop runs from 0 to
    m_max − (total angular momentum accumulated so far), which both
    respects the triangular structure of the auxiliary table and
    keeps all right-hand-side m+1 accesses in-bounds.

    Parameters
    ----------
    a_max   : (ax_max, ay_max, az_max)   component-wise upper limits for A
    c_max   : (cx_max, cy_max, cz_max)   component-wise upper limits for C
    p, q    : combined bra / ket exponents
    zeta    : pq/(p+q)
    PA      : P − A
    WP      : W − P
    QC      : Q − C
    WQ      : W − Q
    prefactor : 2π^(5/2)/(p·q·√(p+q)) · K_AB · K_CD
    F       : boys_array output of length ≥ m_max + 1

    Returns
    -------
    R : ndarray, shape (ax_max+1, ay_max+1, az_max+1,
                        cx_max+1, cy_max+1, cz_max+1, m_max+1)
    """
    ax_max, ay_max, az_max = a_max
    cx_max, cy_max, cz_max = c_max
    m_max = ax_max + ay_max + az_max + cx_max + cy_max + cz_max

    R = np.zeros((ax_max+1, ay_max+1, az_max+1,
                  cx_max+1, cy_max+1, cz_max+1,
                  m_max+1))

    # Seed: [0|0]^(m) = prefactor · F_m(T)
    for m in range(m_max + 1):
        R[0, 0, 0, 0, 0, 0, m] = prefactor * F[m]

    # ---- Pass A: grow angular momentum on A (c held at zero) ----

    for ax in range(1, ax_max + 1):
        for m in range(m_max - ax + 1):
            R[ax, 0, 0, 0, 0, 0, m] = (
                PA[0] * R[ax-1, 0, 0, 0, 0, 0, m]
                - (zeta/p) * WP[0] * R[ax-1, 0, 0, 0, 0, 0, m+1]
            )
            if ax >= 2:
                R[ax, 0, 0, 0, 0, 0, m] += (ax-1) / (2*p) * (
                    R[ax-2, 0, 0, 0, 0, 0, m]
                    - (zeta/p) * R[ax-2, 0, 0, 0, 0, 0, m+1]
                )

    for ay in range(1, ay_max + 1):
        for ax in range(ax_max + 1):
            for m in range(m_max - ax - ay + 1):
                R[ax, ay, 0, 0, 0, 0, m] = (
                    PA[1] * R[ax, ay-1, 0, 0, 0, 0, m]
                    - (zeta/p) * WP[1] * R[ax, ay-1, 0, 0, 0, 0, m+1]
                )
                if ay >= 2:
                    R[ax, ay, 0, 0, 0, 0, m] += (ay-1) / (2*p) * (
                        R[ax, ay-2, 0, 0, 0, 0, m]
                        - (zeta/p) * R[ax, ay-2, 0, 0, 0, 0, m+1]
                    )

    for az in range(1, az_max + 1):
        for ax in range(ax_max + 1):
            for ay in range(ay_max + 1):
                for m in range(m_max - ax - ay - az + 1):
                    R[ax, ay, az, 0, 0, 0, m] = (
                        PA[2] * R[ax, ay, az-1, 0, 0, 0, m]
                        - (zeta/p) * WP[2] * R[ax, ay, az-1, 0, 0, 0, m+1]
                    )
                    if az >= 2:
                        R[ax, ay, az, 0, 0, 0, m] += (az-1) / (2*p) * (
                            R[ax, ay, az-2, 0, 0, 0, m]
                            - (zeta/p) * R[ax, ay, az-2, 0, 0, 0, m+1]
                        )

    # ---- Pass B: grow angular momentum on C ----

    for cx in range(1, cx_max + 1):
        for ax in range(ax_max + 1):
            for ay in range(ay_max + 1):
                for az in range(az_max + 1):
                    for m in range(m_max - ax - ay - az - cx + 1):
                        R[ax, ay, az, cx, 0, 0, m] = (
                            QC[0] * R[ax, ay, az, cx-1, 0, 0, m]
                            - (zeta/q) * WQ[0] * R[ax, ay, az, cx-1, 0, 0, m+1]
                        )
                        if cx >= 2:
                            R[ax, ay, az, cx, 0, 0, m] += (cx-1) / (2*q) * (
                                R[ax, ay, az, cx-2, 0, 0, m]
                                - (zeta/q) * R[ax, ay, az, cx-2, 0, 0, m+1]
                            )
                        if ax >= 1:
                            R[ax, ay, az, cx, 0, 0, m] += (
                                ax / (2*(p+q))
                                * R[ax-1, ay, az, cx-1, 0, 0, m+1]
                            )

    for cy in range(1, cy_max + 1):
        for cx in range(cx_max + 1):
            for ax in range(ax_max + 1):
                for ay in range(ay_max + 1):
                    for az in range(az_max + 1):
                        for m in range(m_max - ax - ay - az - cx - cy + 1):
                            R[ax, ay, az, cx, cy, 0, m] = (
                                QC[1] * R[ax, ay, az, cx, cy-1, 0, m]
                                - (zeta/q) * WQ[1] * R[ax, ay, az, cx, cy-1, 0, m+1]
                            )
                            if cy >= 2:
                                R[ax, ay, az, cx, cy, 0, m] += (cy-1) / (2*q) * (
                                    R[ax, ay, az, cx, cy-2, 0, m]
                                    - (zeta/q) * R[ax, ay, az, cx, cy-2, 0, m+1]
                                )
                            if ay >= 1:
                                R[ax, ay, az, cx, cy, 0, m] += (
                                    ay / (2*(p+q))
                                    * R[ax, ay-1, az, cx, cy-1, 0, m+1]
                                )

    for cz in range(1, cz_max + 1):
        for cy in range(cy_max + 1):
            for cx in range(cx_max + 1):
                for ax in range(ax_max + 1):
                    for ay in range(ay_max + 1):
                        for az in range(az_max + 1):
                            for m in range(m_max - ax - ay - az - cx - cy - cz + 1):
                                R[ax, ay, az, cx, cy, cz, m] = (
                                    QC[2] * R[ax, ay, az, cx, cy, cz-1, m]
                                    - (zeta/q) * WQ[2] * R[ax, ay, az, cx, cy, cz-1, m+1]
                                )
                                if cz >= 2:
                                    R[ax, ay, az, cx, cy, cz, m] += (cz-1) / (2*q) * (
                                        R[ax, ay, az, cx, cy, cz-2, m]
                                        - (zeta/q) * R[ax, ay, az, cx, cy, cz-2, m+1]
                                    )
                                if az >= 1:
                                    R[ax, ay, az, cx, cy, cz, m] += (
                                        az / (2*(p+q))
                                        * R[ax, ay, az-1, cx, cy, cz-1, m+1]
                                    )

    return R


# ---------------------------------------------------------------------------
# Stage 2 — Bra Horizontal Recurrence Relation
# ---------------------------------------------------------------------------

def _eri_hrr_bra(R0: np.ndarray, a: tuple, b: tuple,
                 AB: np.ndarray) -> np.ndarray:
    """
    Transfer angular momentum from A to B using the m=0 slice R0.

        [a | b+1ᵢ | c0] = [a+1ᵢ | b | c0]  +  (A−B)ᵢ [a | b | c0]

    The ax loop at level bx runs from 0 to (a[0]+b[0]−bx), the correct
    upper bound that ensures [ax+1, ..., bx−1, ...] — populated during
    the preceding bx step — is always present.  The analogous bounds
    apply to the ay and az loops for the by and bz transfers.

    The last three dimensions of H (and of R0) index the ket — they are
    carried through unchanged via numpy slice broadcasting.

    Parameters
    ----------
    R0 : m=0 slice of the VRR output,
         shape (a[0]+b[0]+1, a[1]+b[1]+1, a[2]+b[2]+1,
                cx_max+1, cy_max+1, cz_max+1)
    a  : target angular momentum on A
    b  : target angular momentum on B
    AB : A − B

    Returns
    -------
    ndarray, shape (cx_max+1, cy_max+1, cz_max+1)
        The [a | b | c]^(0) slice for all ket indices c.
    """
    ax_max = a[0] + b[0]
    ay_max = a[1] + b[1]
    az_max = a[2] + b[2]
    nc_shape = R0.shape[3:]       # (cx_max+1, cy_max+1, cz_max+1)

    # H[ax, ay, az, bx, by, bz, *ket]
    H = np.zeros((ax_max+1, ay_max+1, az_max+1,
                  b[0]+1, b[1]+1, b[2]+1) + nc_shape)

    # Seed: all angular momentum on A, none on B
    H[:, :, :, 0, 0, 0] = R0

    # Transfer bx
    for bx in range(1, b[0] + 1):
        for ax in range(a[0] + b[0] - bx + 1):
            H[ax, :, :, bx, 0, 0] = (
                H[ax+1, :, :, bx-1, 0, 0]
                + AB[0] * H[ax, :, :, bx-1, 0, 0]
            )

    # Transfer by
    for by in range(1, b[1] + 1):
        for bx in range(b[0] + 1):
            for ax in range(a[0] + 1):
                for ay in range(a[1] + b[1] - by + 1):
                    H[ax, ay, :, bx, by, 0] = (
                        H[ax, ay+1, :, bx, by-1, 0]
                        + AB[1] * H[ax, ay, :, bx, by-1, 0]
                    )

    # Transfer bz
    for bz in range(1, b[2] + 1):
        for by in range(b[1] + 1):
            for bx in range(b[0] + 1):
                for ax in range(a[0] + 1):
                    for ay in range(a[1] + 1):
                        for az in range(a[2] + b[2] - bz + 1):
                            H[ax, ay, az, bx, by, bz] = (
                                H[ax, ay, az+1, bx, by, bz-1]
                                + AB[2] * H[ax, ay, az, bx, by, bz-1]
                            )

    return H[a[0], a[1], a[2], b[0], b[1], b[2]]


# ---------------------------------------------------------------------------
# Stage 3 — Ket Horizontal Recurrence Relation
# ---------------------------------------------------------------------------

def _eri_hrr_ket(bra_slice: np.ndarray, c: tuple, d: tuple,
                 CD: np.ndarray) -> float:
    """
    Transfer angular momentum from C to D.

        [ab | c | d+1ᵢ] = [ab | c+1ᵢ | d]  +  (C−D)ᵢ [ab | c | d]

    Analogous loop-bound convention to _eri_hrr_bra.

    Parameters
    ----------
    bra_slice : shape (c[0]+d[0]+1, c[1]+d[1]+1, c[2]+d[2]+1)
                Output of _eri_hrr_bra (or a slice of R0 when b = 0).
    c  : target angular momentum on C
    d  : target angular momentum on D
    CD : C − D

    Returns
    -------
    float : the final (ab|cd) integral value
    """
    cx_max = c[0] + d[0]
    cy_max = c[1] + d[1]
    cz_max = c[2] + d[2]

    # G[cx, cy, cz, dx, dy, dz]
    G = np.zeros((cx_max+1, cy_max+1, cz_max+1,
                  d[0]+1, d[1]+1, d[2]+1))

    G[:, :, :, 0, 0, 0] = bra_slice

    # Transfer dx
    for dx in range(1, d[0] + 1):
        for cx in range(c[0] + d[0] - dx + 1):
            G[cx, :, :, dx, 0, 0] = (
                G[cx+1, :, :, dx-1, 0, 0]
                + CD[0] * G[cx, :, :, dx-1, 0, 0]
            )

    # Transfer dy
    for dy in range(1, d[1] + 1):
        for dx in range(d[0] + 1):
            for cx in range(c[0] + 1):
                for cy in range(c[1] + d[1] - dy + 1):
                    G[cx, cy, :, dx, dy, 0] = (
                        G[cx, cy+1, :, dx, dy-1, 0]
                        + CD[1] * G[cx, cy, :, dx, dy-1, 0]
                    )

    # Transfer dz
    for dz in range(1, d[2] + 1):
        for dy in range(d[1] + 1):
            for dx in range(d[0] + 1):
                for cx in range(c[0] + 1):
                    for cy in range(c[1] + 1):
                        for cz in range(c[2] + d[2] - dz + 1):
                            G[cx, cy, cz, dx, dy, dz] = (
                                G[cx, cy, cz+1, dx, dy, dz-1]
                                + CD[2] * G[cx, cy, cz, dx, dy, dz-1]
                            )

    return float(G[c[0], c[1], c[2], d[0], d[1], d[2]])


# ---------------------------------------------------------------------------
# Primitive ERI
# ---------------------------------------------------------------------------

def eri_primitive(a: tuple, b: tuple, c: tuple, d: tuple,
                  alpha: float, beta: float,
                  gamma: float, delta: float,
                  A: np.ndarray, B: np.ndarray,
                  C: np.ndarray, D: np.ndarray) -> float:
    """
    Unnormalized ERI between four primitive Gaussians.

    Parameters
    ----------
    a, b, c, d   : angular-momentum tuples, e.g. (1, 0, 0) for p_x
    alpha, beta  : Gaussian exponents of a, b
    gamma, delta : Gaussian exponents of c, d
    A, B, C, D   : centres as numpy arrays of length 3

    Returns
    -------
    float : unnormalized (ab|cd)
    """
    p    = alpha + beta
    q    = gamma + delta
    P    = (alpha * A + beta  * B) / p
    Q    = (gamma * C + delta * D) / q
    zeta = p * q / (p + q)
    W    = (p * P + q * Q) / (p + q)

    PA = P - A
    WP = W - P
    QC = Q - C
    WQ = W - Q
    AB = A - B
    CD = C - D

    K_AB = np.exp(-alpha * beta  / p * float(np.dot(AB, AB)))
    K_CD = np.exp(-gamma * delta / q * float(np.dot(CD, CD)))
    prefactor = (2.0 * np.pi**2.5
                 / (p * q * np.sqrt(p + q))
                 * K_AB * K_CD)

    a_max = (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    c_max = (c[0]+d[0], c[1]+d[1], c[2]+d[2])
    m_max = sum(a_max) + sum(c_max)

    T = zeta * float(np.dot(P - Q, P - Q))
    F = boys_array(m_max, T)

    R  = _eri_vrr(a_max, c_max, p, q, zeta, PA, WP, QC, WQ, prefactor, F)
    R0 = R[..., 0]   # m=0 slice, shape (*a_max+1, *c_max+1)

    # Bra HRR — skip if b carries no angular momentum
    if sum(b) == 0:
        slice_c = R0[a[0], a[1], a[2]]       # shape (cx_max+1, cy_max+1, cz_max+1)
    else:
        slice_c = _eri_hrr_bra(R0, a, b, AB)

    # Ket HRR — skip if d carries no angular momentum
    if sum(d) == 0:
        return float(slice_c[c[0], c[1], c[2]])
    else:
        return _eri_hrr_ket(slice_c, c, d, CD)


# ---------------------------------------------------------------------------
# Contracted ERI
# ---------------------------------------------------------------------------

def eri_contracted(shell_a: dict, shell_b: dict,
                   shell_c: dict, shell_d: dict) -> float:
    """
    ERI between four contracted basis functions.

    Shell format: same as overlap.py / kinetic.py / nuclear.py — each
    shell is a dict with keys
        'center'       : np.array([x, y, z])
        'angular'      : tuple (lx, ly, lz)
        'exponents'    : list of floats
        'coefficients' : list of floats

    Returns
    -------
    float : normalised, contracted (ab|cd)
    """
    result = 0.0
    A, a = shell_a['center'], shell_a['angular']
    B, b = shell_b['center'], shell_b['angular']
    C, c = shell_c['center'], shell_c['angular']
    D, d = shell_d['center'], shell_d['angular']

    for alpha, ca in zip(shell_a['exponents'], shell_a['coefficients']):
        Na = norm_primitive(alpha, a)
        for beta, cb in zip(shell_b['exponents'], shell_b['coefficients']):
            Nb = norm_primitive(beta, b)
            for gamma, cc in zip(shell_c['exponents'], shell_c['coefficients']):
                Nc = norm_primitive(gamma, c)
                for delta, cd in zip(shell_d['exponents'], shell_d['coefficients']):
                    Nd = norm_primitive(delta, d)
                    result += (Na * Nb * Nc * Nd
                               * ca * cb * cc * cd
                               * eri_primitive(a, b, c, d,
                                               alpha, beta, gamma, delta,
                                               A, B, C, D))
    return result


# ---------------------------------------------------------------------------
# Full ERI tensor
# ---------------------------------------------------------------------------

def build_eri_tensor(basis: list) -> np.ndarray:
    """
    Build the full (N, N, N, N) ERI tensor for a list of contracted basis
    functions, exploiting the 8-fold permutation symmetry:

        (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
                = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)

    Only unique shell quartets — those with compound bra index
    IJ = i(i+1)/2+j ≥ KL = k(k+1)/2+l — are evaluated.  All eight
    symmetry-equivalent elements are filled simultaneously.

    Parameters
    ----------
    basis : list of shell dicts (same format as the other integral modules)

    Returns
    -------
    ERI : ndarray, shape (N, N, N, N), fully symmetric under all
          8 permutations of the index pairs.
    """
    n = len(basis)
    ERI = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(i + 1):
            ij = i * (i + 1) // 2 + j
            for k in range(n):
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:
                        continue
                    val = eri_contracted(basis[i], basis[j],
                                         basis[k], basis[l])
                    # Apply all 8 symmetry copies simultaneously
                    for ii, jj, kk, ll in [
                        (i, j, k, l), (j, i, k, l),
                        (i, j, l, k), (j, i, l, k),
                        (k, l, i, j), (l, k, i, j),
                        (k, l, j, i), (l, k, j, i),
                    ]:
                        ERI[ii, jj, kk, ll] = val

    return ERI