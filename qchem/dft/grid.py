"""
qchem/dft/grid.py — numerical integration grid for Kohn-Sham DFT

Overview
--------
Kohn-Sham DFT replaces the exact exchange-correlation energy with an
approximate functional E_xc[ρ] that cannot be evaluated analytically
in a Gaussian AO basis.  Unlike the kinetic, overlap, nuclear-attraction
and Coulomb integrals — all of which reduce to closed-form expressions —
the XC integrand f(ρ(r), ∇ρ(r)) has no GTO primitive that can be
integrated exactly.  We therefore evaluate it by numerical quadrature:

    E_xc = ∫ f(ρ(r)) dr  ≈  Σ_g w_g f(ρ(r_g))

where the {r_g, w_g} pairs are the grid points and weights produced by
this module, and f is the XC energy density supplied by dft.xc.

The XC contribution to the Kohn-Sham matrix (the analogue of the
two-electron G matrix in HF) is:

    V_xc,μν = ∫ φ_μ(r) v_xc(ρ(r)) φ_ν(r) dr
            ≈ Σ_g w_g φ_μ(r_g) v_xc(r_g) φ_ν(r_g)

where v_xc = δE_xc/δρ is the XC potential (also from dft.xc).

Grid construction — three-step pipeline
-----------------------------------------
1.  **Atomic grids.** For each nucleus A, build a product grid on
    spherical shells: a radial quadrature (n_rad points on [0, ∞)) times
    an angular quadrature (n_ang points on the unit sphere).  The combined
    atomic weight at point r is:

        W_A(r) = w_rad(r) · r² · w_ang    (r² is the spherical Jacobian)

2.  **Becke partitioning.** When all atomic grids are superimposed, each
    grid point is counted once per atom.  Becke's scheme assigns a fuzzy
    Voronoi weight w_A(r) ∈ [0, 1] to atom A at point r such that

        Σ_A w_A(r) = 1  ∀ r.

    This resolves the overcounting: a point very close to nucleus A gets
    weight ≈ 1 for atom A and ≈ 0 for all others.

3.  **Molecular grid assembly.** The final weight for the g-th point of
    atom A is:

        total_weight = w_A(r_Ag) · W_A(r_Ag)

    The set of all (r_Ag, total_weight) across all atoms is the molecular
    grid passed to eval_ao_on_grid and build_xc_matrix.

Radial grid — Gauss-Chebyshev + Becke transform
-------------------------------------------------
We use the Gauss-Chebyshev quadrature of the second kind on x ∈ (-1, 1):

    x_i = cos(i π / (n+1)),   i = 1, …, n
    w_i = π / (n+1) · sin(i π / (n+1))

and map to r ∈ (0, ∞) via Becke's (1988) substitution:

    r_i = R_atom · (1 + x_i) / (1 - x_i)

The Jacobian dr/dx = 2R/(1-x)² transforms the weights to:

    w_rad,i = w_i · 2R / (1 - x_i)²

A Bragg-Slater radius R_atom is used for each element to adapt the grid
to the size of the atom.

Angular grid — Gauss-Legendre product rule
-------------------------------------------
The unit sphere integral ∫ f(θ, φ) sin θ dθ dφ is factored as:

    Σ_{j=1}^{n_θ} w_j · Σ_{k=1}^{n_φ} (2π/n_φ) · f(θ_j, φ_k)

where {cos θ_j, w_j} are the Gauss-Legendre nodes and weights on [-1,1]
(no sin θ factor — GL integrates in cos θ, so the Jacobian is absorbed)
and φ_k = 2π(k-1)/n_φ are equispaced azimuths.  The total number of
angular points is n_ang = n_θ · n_φ.  Setting n_φ = 2 · n_θ gives a
roughly isotropic distribution.  Production codes prefer Lebedev-Laikov
grids, which achieve higher angular accuracy with fewer points; those
require a large tabulation and are outside the scope of this module, but
the Becke-partitioned product grid is exact enough for GGA calculations
with n_rad ~ 75 and n_ang ~ 302.

Becke partitioning — fuzzy Voronoi cells
-----------------------------------------
For each pair of atoms A, B define the elliptic coordinate:

    μ_AB(r) = (|r − R_A| − |r − R_B|) / R_AB   ∈ (-1, 1)

Becke's polynomial smoothing step applies a 3-fold iterated mapping:

    f(x) = x(3 - x²)/2            (maps [-1,1] to [-1,1], steeper at 0)
    f_3(x) = f(f(f(x)))            (third iterate)
    s(μ) = (1 − f_3(μ)) / 2       ∈ [0,1]   (cell function)

An atomic heteronuclear correction scales μ before smoothing:

    χ_AB = R_A / R_B,   u_AB = (χ_AB - 1)/(χ_AB + 1)
    a_AB = u_AB / (u_AB² - 1)     (clipped to [-0.5, 0.5])
    ν_AB = μ_AB + a_AB(1 − μ_AB²) (shifted boundary toward smaller atom)

The Becke cell function for atom A is:

    P_A(r) = Π_{B ≠ A} s(ν_AB(r))

and the normalised Becke weight is:

    w_A(r) = P_A(r) / Σ_C P_C(r)

AO evaluation and density
--------------------------
Each contracted Cartesian GTO is:

    φ_μ(r) = N_μ · Σ_k c_k · (x-Ax)^lx (y-Ay)^ly (z-Az)^lz
                              · exp(-α_k |r − A|²)

where N_μ is the normalisation constant.  The electron density at a
grid point r is:

    ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)
          = ao(r) · P · ao(r)^T  (scalar contraction)

where ao(r) is the row vector of AO values and P is the density matrix.

Functions
---------
radial_grid_becke(n_rad, R_atom)
    Gauss-Chebyshev radial grid mapped to (0, ∞).

angular_grid_product(n_theta)
    Product Gauss-Legendre × equispaced angular grid on the unit sphere.

atomic_grid(element, n_rad, n_theta)
    Combine radial and angular grids for a single atom.

becke_partition_weights(points, atom_coords, atom_symbols)
    Fuzzy Voronoi weights w_A(r) for a batch of points.

build_molecular_grid(mol, n_rad, n_theta)
    Full molecular grid: (points, weights) with Becke partitioning.

eval_ao_on_grid(shells, points)
    Evaluate all contracted GTOs at a set of grid points.

eval_density_on_grid(P, ao_values)
    Compute ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r) at every grid point.

build_xc_matrix(ao_values, v_xc, weights)
    Assemble the XC contribution to the KS matrix:
    V_xc,μν = Σ_g w_g φ_μ(r_g) v_xc(r_g) φ_ν(r_g).

References
----------
Becke, A. D., J. Chem. Phys. 88, 2547 (1988).
    The original paper introducing atomic partitioning and the Becke
    transform.  The polynomial smoothing (Section II) and the
    heteronuclear correction (Appendix) are both implemented here.

Johnson, B. G.; Frisch, M. J., Chem. Phys. Lett. 216, 133 (1993).
    Efficient evaluation strategy for Kohn-Sham XC matrix elements.

Lebedev, V. I., Zh. Vychisl. Mat. Mat. Fiz. 15, 48 (1975).
    Spherical quadrature used in production DFT codes (not implemented
    here; product grid used instead).

Szabo & Ostlund, "Modern Quantum Chemistry", Dover (1996), Appendix A.
Helgaker, Jørgensen & Olsen, "Molecular Electronic-Structure Theory",
    Wiley (2000), Sections 8.3–8.5.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from ..basis import ATOMIC_NUMBER
from ..linalg import double_factorial as _double_factorial


# ---------------------------------------------------------------------------
# Bragg-Slater atomic radii (in bohr)
# ---------------------------------------------------------------------------
# Used both for the Becke radial transform (R_atom sets the length scale
# for each atom's radial grid) and for the heteronuclear correction in
# Becke partitioning.
#
# Source: Becke (1988), Table I; converted from Å to bohr using
#         1 Å = 1.8897259886 bohr (CODATA 2018).
# Row 1 noble gases are assigned H's radius as a conservative default.

_ANGSTROM_TO_BOHR: float = 1.8897259886

# fmt: off
_BRAGG_SLATER_ANGSTROM: dict[str, float] = {
    'H':  0.35,  'He': 0.35,
    'Li': 1.45,  'Be': 1.05,  'B':  0.85,  'C':  0.70,
    'N':  0.65,  'O':  0.60,  'F':  0.50,  'Ne': 0.38,
    'Na': 1.80,  'Mg': 1.50,  'Al': 1.25,  'Si': 1.10,
    'P':  1.00,  'S':  1.00,  'Cl': 1.00,  'Ar': 0.71,
    'K':  2.20,  'Ca': 1.80,  'Sc': 1.60,  'Ti': 1.40,
    'V':  1.35,  'Cr': 1.40,  'Mn': 1.40,  'Fe': 1.40,
    'Co': 1.35,  'Ni': 1.35,  'Cu': 1.35,  'Zn': 1.35,
    'Ga': 1.30,  'Ge': 1.25,  'As': 1.15,  'Se': 1.15,
    'Br': 1.15,  'Kr': 1.00,
}
# fmt: on

BRAGG_SLATER_RADII: dict[str, float] = {
    sym: r * _ANGSTROM_TO_BOHR for sym, r in _BRAGG_SLATER_ANGSTROM.items()
}
"""Bragg-Slater radii in **bohr** for elements H–Kr."""


def _bragg_radius(symbol: str) -> float:
    """
    Return the Bragg-Slater radius (bohr) for *symbol*.

    Falls back to 1.5 bohr for unknown elements rather than raising,
    so the grid degrades gracefully when heavy elements are present.
    """
    return BRAGG_SLATER_RADII.get(symbol, 1.5 * _ANGSTROM_TO_BOHR)


# ---------------------------------------------------------------------------
# GTO normalisation
# ---------------------------------------------------------------------------

def _norm_primitive(alpha: float, lx: int, ly: int, lz: int) -> float:
    """
    Normalisation constant for a single primitive Cartesian GTO.

    N(α, lx, ly, lz) = (2α/π)^(3/4) · (4α)^(L/2)
                        / √[(2lx-1)!! (2ly-1)!! (2lz-1)!!]

    where L = lx + ly + lz.  Matches the convention used throughout
    qchem.integrals (see integrals/overlap.py).

    Parameters
    ----------
    alpha : float
        Primitive exponent (bohr⁻²).
    lx, ly, lz : int
        Cartesian angular momentum components.

    Returns
    -------
    float
        Normalisation constant (dimensionless in atomic units).
    """
    L = lx + ly + lz
    prefactor = (2.0 * alpha / np.pi) ** 0.75 * (4.0 * alpha) ** (L / 2.0)
    denom = np.sqrt(
        _double_factorial(2 * lx - 1)
        * _double_factorial(2 * ly - 1)
        * _double_factorial(2 * lz - 1)
    )
    return prefactor / denom


# ---------------------------------------------------------------------------
# Radial grid
# ---------------------------------------------------------------------------

def radial_grid_becke(
    n_rad: int,
    R_atom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a radial quadrature grid on (0, ∞) via the Becke transform.

    The algorithm is:

    1.  Generate n_rad Gauss-Chebyshev type-2 nodes on (-1, 1):

            x_i = cos(i π / (n+1)),   i = 1, …, n
            w_i = π/(n+1) · sin(i π / (n+1))

    2.  Map to r ∈ (0, ∞) with Becke's transformation:

            r_i = R_atom · (1 + x_i) / (1 - x_i)

    3.  Absorb the Jacobian dr/dx = 2R/(1-x)² into the weights:

            w_rad,i = w_i · 2R / (1 - x_i)²

    The r² spherical Jacobian is NOT included here; it is multiplied in
    by atomic_grid, which combines radial and angular weights.

    Parameters
    ----------
    n_rad : int
        Number of radial quadrature points.  Typical values: 50 (fast,
        slightly inaccurate), 75 (standard), 100 (accurate).
    R_atom : float
        Bragg-Slater radius of the atom in bohr.  Sets the length scale
        of the mapping so that most quadrature weight falls near the atom.

    Returns
    -------
    r : ndarray, shape (n_rad,)
        Radial quadrature nodes in bohr, ordered from small to large r.
    w_rad : ndarray, shape (n_rad,)
        Radial quadrature weights (not including r²).

    References
    ----------
    Becke (1988), Eq. (25) and surrounding discussion.
    """
    i = np.arange(1, n_rad + 1)
    theta = i * np.pi / (n_rad + 1)

    # Gauss-Chebyshev type-2 nodes and weights on (-1, 1)
    x     = np.cos(theta)           # x_i = cos(iπ/(n+1))
    w_gc  = np.pi / (n_rad + 1) * np.sin(theta)

    # Becke mapping x → r
    r     = R_atom * (1.0 + x) / (1.0 - x)

    # Jacobian of the transformation
    jacobian = 2.0 * R_atom / (1.0 - x) ** 2

    w_rad = w_gc * jacobian
    return r, w_rad


# ---------------------------------------------------------------------------
# Angular grid
# ---------------------------------------------------------------------------

def angular_grid_product(n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a product Gauss-Legendre × equispaced angular grid on S².

    The unit-sphere integral ∫ f(Ω) dΩ is discretised as:

        Σ_{j=1}^{n_θ} w_j Σ_{k=1}^{n_φ} (2π/n_φ) · f(θ_j, φ_k)

    where {cos θ_j, w_j} are GL nodes and weights on [-1, 1] and
    φ_k = 2π(k-1)/n_φ with n_φ = 2 · n_θ.

    Because GL quadrature integrates with respect to the variable
    cos θ (not θ), the sin θ factor in dΩ = sin θ dθ dφ is already
    absorbed into the GL weight; no extra factor is needed.

    Production codes use Lebedev-Laikov grids, which achieve the same
    accuracy with fewer points by exploiting the full octahedral symmetry
    of the sphere.  The product grid used here is simpler to implement
    and adequate for moderate angular accuracy requirements.

    Parameters
    ----------
    n_theta : int
        Number of Gauss-Legendre polar nodes.  The total number of
        angular points returned is n_theta * (2 * n_theta).

    Returns
    -------
    xyz : ndarray, shape (n_ang, 3)
        Cartesian unit vectors of quadrature points on S².
    w_ang : ndarray, shape (n_ang,)
        Angular quadrature weights (sum ≈ 4π).
    """
    n_phi = 2 * n_theta

    # GL nodes (cos θ values) and weights on [-1, 1]
    cos_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)

    # Equispaced azimuths
    phi_vals = 2.0 * np.pi * np.arange(n_phi) / n_phi
    w_phi    = 2.0 * np.pi / n_phi

    # Build full tensor-product grid
    cos_t_full = np.repeat(cos_theta, n_phi)   # shape (n_ang,)
    sin_t_full = np.sqrt(np.maximum(1.0 - cos_t_full ** 2, 0.0))

    phi_full = np.tile(phi_vals, n_theta)       # shape (n_ang,)
    w_full   = np.repeat(w_theta, n_phi) * w_phi

    xyz = np.column_stack([
        sin_t_full * np.cos(phi_full),
        sin_t_full * np.sin(phi_full),
        cos_t_full,
    ])

    return xyz, w_full


# ---------------------------------------------------------------------------
# Atomic grid
# ---------------------------------------------------------------------------

def atomic_grid(
    element: str,
    n_rad: int = 75,
    n_theta: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 3-D quadrature grid for a single atom centred at the origin.

    The grid is the tensor product of a radial grid and an angular grid.
    The total weight at each point is:

        W_g = w_rad,i · r_i² · w_ang,j

    where r_i² is the spherical polar Jacobian.  The resulting (r, w)
    pairs satisfy:

        Σ_g W_g f(r_g) ≈ ∫₀^∞ ∫_{S²} f(r, Ω) r² dr dΩ

    Parameters
    ----------
    element : str
        Element symbol (case-insensitive), e.g. 'C' or 'o'.
    n_rad : int
        Number of radial points.  Default 75.
    n_theta : int
        Number of GL polar points.  Total angular points = n_theta · 2n_theta.
        Default 17 (→ 34 × 17 = 578 angular points).

    Returns
    -------
    points : ndarray, shape (n_rad * n_ang, 3)
        Cartesian coordinates of grid points in bohr, centred at origin.
    weights : ndarray, shape (n_rad * n_ang,)
        Quadrature weights including the r² Jacobian.

    Notes
    -----
    To place the grid at atom A located at R_A, add R_A to *points* after
    calling this function.  build_molecular_grid does this automatically.
    """
    sym  = element.strip().capitalize()
    R_a  = _bragg_radius(sym)

    r,    w_rad = radial_grid_becke(n_rad, R_a)
    xyz0, w_ang = angular_grid_product(n_theta)

    n_ang = len(w_ang)

    # r² · w_rad for each radial shell, broadcast over angular points
    w_shell = r ** 2 * w_rad             # shape (n_rad,)

    # Outer product: (n_rad, 1) * (1, n_ang) → (n_rad, n_ang)
    W = np.outer(w_shell, w_ang)          # shape (n_rad, n_ang)

    # Points: for each shell i, take r[i] * each unit vector
    # xyz0 shape: (n_ang, 3) → broadcast with r (n_rad,)
    pts = r[:, None, None] * xyz0[None, :, :]   # (n_rad, n_ang, 3)

    return pts.reshape(-1, 3), W.reshape(-1)


# ---------------------------------------------------------------------------
# Becke partitioning
# ---------------------------------------------------------------------------

def _becke_f(x: np.ndarray) -> np.ndarray:
    """One Becke smoothing step: f(x) = x(3 - x²)/2."""
    return x * (3.0 - x * x) * 0.5


def _becke_cell_function(mu: np.ndarray) -> np.ndarray:
    """
    Becke cell function s(μ) ∈ [0, 1].

    Applies the polynomial smoothing f three times and returns:
        s(μ) = (1 − f₃(μ)) / 2

    Parameters
    ----------
    mu : ndarray
        Elliptic coordinate(s) in (-1, 1).

    Returns
    -------
    ndarray
        Cell function values, same shape as *mu*.
    """
    nu = _becke_f(_becke_f(_becke_f(mu)))
    return 0.5 * (1.0 - nu)


def becke_partition_weights(
    points: np.ndarray,
    atom_coords: np.ndarray,
    atom_symbols: list[str],
) -> np.ndarray:
    """
    Compute Becke normalised partition weights w_A(r) for each atom.

    For each grid point r the function returns an array of shape
    (n_points, n_atoms) where row g contains the Becke weights
    {w_A(r_g)} with Σ_A w_A(r_g) = 1 for every g.

    Algorithm
    ---------
    For each pair (A, B):

    1.  Compute μ_AB(r) = (|r − R_A| − |r − R_B|) / R_AB.

    2.  Apply the heteronuclear correction (Becke 1988, Appendix):
            χ_AB = R_A / R_B,   u_AB = (χ_AB − 1)/(χ_AB + 1)
            a_AB = clip(u_AB / (u_AB² − 1), −0.5, 0.5)
            ν_AB = μ_AB + a_AB(1 − μ_AB²)

    3.  Accumulate P_A += log s(ν_AB) for B ≠ A  (log-sum for stability).

    4.  Normalise: w_A = exp(P_A) / Σ_B exp(P_B).

    Parameters
    ----------
    points : ndarray, shape (n_pts, 3)
        Cartesian coordinates of grid points in bohr.
    atom_coords : ndarray, shape (n_atoms, 3)
        Nuclear coordinates in bohr.
    atom_symbols : list of str
        Element symbols matching *atom_coords*.

    Returns
    -------
    w : ndarray, shape (n_pts, n_atoms)
        Becke partition weights.  Rows sum to 1.

    References
    ----------
    Becke (1988), Eqs. (13)–(22) and Appendix.
    """
    n_pts   = points.shape[0]
    n_atoms = atom_coords.shape[0]

    # Bragg-Slater radii for the correction factor
    R_bs = np.array([_bragg_radius(sym) for sym in atom_symbols])  # (n_atoms,)

    # Distances from every grid point to every nucleus: (n_pts, n_atoms)
    diff = points[:, None, :] - atom_coords[None, :, :]       # (n_pts, n_atoms, 3)
    dist = np.linalg.norm(diff, axis=-1)                       # (n_pts, n_atoms)

    # Log of the cell function product, accumulated for each atom A
    # log_P[g, A] = Σ_{B≠A} log s(ν_AB(r_g))
    log_P = np.zeros((n_pts, n_atoms))

    for A in range(n_atoms):
        for B in range(n_atoms):
            if A == B:
                continue

            R_AB = np.linalg.norm(atom_coords[A] - atom_coords[B])
            if R_AB < 1e-14:
                continue                  # Coincident nuclei: skip

            # Elliptic coordinate μ_AB (n_pts,)
            mu_AB = (dist[:, A] - dist[:, B]) / R_AB

            # Heteronuclear correction
            chi_AB = R_bs[A] / R_bs[B]
            u_AB   = (chi_AB - 1.0) / (chi_AB + 1.0)
            denom  = u_AB ** 2 - 1.0
            if abs(denom) < 1e-12:
                a_AB = 0.0             # Homonuclear: no shift
            else:
                a_AB = np.clip(u_AB / denom, -0.5, 0.5)

            nu_AB = mu_AB + a_AB * (1.0 - mu_AB ** 2)
            nu_AB = np.clip(nu_AB, -1.0, 1.0)   # Guard against floating-point drift

            s = _becke_cell_function(nu_AB)

            # Clip to avoid log(0) for points exactly on cell boundaries
            s = np.maximum(s, 1e-300)
            log_P[:, A] += np.log(s)

    # Exponentiate and normalise (subtract max for numerical stability)
    log_P -= log_P.max(axis=1, keepdims=True)
    P = np.exp(log_P)
    w = P / P.sum(axis=1, keepdims=True)

    return w


# ---------------------------------------------------------------------------
# Molecular grid
# ---------------------------------------------------------------------------

def build_molecular_grid(
    mol,
    n_rad: int = 75,
    n_theta: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the full molecular numerical integration grid.

    For each atom A:
        1.  Build an atomic grid centred at A.
        2.  Compute Becke weights w_A(r) for all grid points of all atoms.
        3.  Scale each atomic weight by the Becke factor for atom A.

    The final grid interleaves all atomic contributions.  Any point can
    carry weight from at most a few nearby atoms (the Becke weights decay
    quickly away from each nucleus), so no explicit pruning is needed.

    Parameters
    ----------
    mol : Molecule
        Molecular geometry.  Requires ``mol.atoms`` (list of (sym, coord)
        pairs in bohr) — compatible with qchem.molecule.Molecule.
    n_rad : int
        Radial points per atom.  Default 75.
    n_theta : int
        GL polar points per atom.  Total angular per atom = n_theta · 2n_theta.
        Default 17, giving 578 angular points.

    Returns
    -------
    points : ndarray, shape (n_grid, 3)
        Cartesian grid-point coordinates in bohr.
    weights : ndarray, shape (n_grid,)
        Final quadrature weights (atomic weights × Becke partition factor).

    Notes
    -----
    A typical small molecule (10 atoms, n_rad=75, n_theta=17) produces
    ~435 000 grid points.  The Becke-weight evaluation is O(n_atoms² ·
    n_grid) and dominates; for large molecules consider a distance-based
    cutoff that zeros the Becke weight for well-separated atom pairs.

    Examples
    --------
    >>> from qchem.molecule import Molecule
    >>> mol = Molecule([('O', [0,0,0]), ('H', [0,1.43,1.11]),
    ...                 ('H', [0,-1.43,1.11])])
    >>> pts, wts = build_molecular_grid(mol, n_rad=50, n_theta=11)
    >>> pts.shape
    (242, 3)   # n_atoms × n_rad × n_ang = 3 × 50 × 242 ÷ n_atoms (approx)
    """
    n_atoms   = mol.n_atoms
    coords    = np.array([r for _, r in mol.atoms])     # (n_atoms, 3)
    symbols   = [sym for sym, _ in mol.atoms]

    # --- Build all atomic grids before partitioning ---
    atom_pts_list: list[np.ndarray] = []
    atom_wts_list: list[np.ndarray] = []

    for idx, (sym, R_A) in enumerate(mol.atoms):
        pts, wts = atomic_grid(sym, n_rad=n_rad, n_theta=n_theta)
        atom_pts_list.append(pts + R_A)   # translate to atomic centre
        atom_wts_list.append(wts)

    # Stack: (n_total, 3) and (n_total,)
    n_per_atom  = len(atom_pts_list[0])
    all_points  = np.vstack(atom_pts_list)
    all_weights = np.concatenate(atom_wts_list)

    # --- Becke partition weights: (n_total, n_atoms) ---
    w_becke = becke_partition_weights(all_points, coords, symbols)

    # Apply: for points belonging to atom A, multiply by w_becke[:, A]
    final_weights = all_weights.copy()
    for A in range(n_atoms):
        sl = slice(A * n_per_atom, (A + 1) * n_per_atom)
        final_weights[sl] *= w_becke[sl, A]

    return all_points, final_weights


# ---------------------------------------------------------------------------
# Basis-function evaluation on the grid
# ---------------------------------------------------------------------------

def eval_ao_on_grid(
    shells: list[dict],
    points: np.ndarray,
) -> np.ndarray:
    """
    Evaluate all contracted AO basis functions at a set of grid points.

    For each contracted GTO:

        φ_μ(r) = N_μ · Σ_k c_k · Δx^lx Δy^ly Δz^lz · exp(-α_k |Δr|²)

    where Δr = r − A_μ is the displacement from the atomic centre.

    Parameters
    ----------
    shells : list of dict
        Basis function dicts as returned by ``basis.build_basis()``.
        Each dict must have keys ``'center'``, ``'angular'``,
        ``'exponents'``, and ``'coefficients'``.
    points : ndarray, shape (n_pts, 3)
        Cartesian grid-point coordinates in bohr.

    Returns
    -------
    ao_values : ndarray, shape (n_pts, n_basis)
        ao_values[g, μ] = φ_μ(r_g).

    Notes
    -----
    Performance note: for large grids and many basis functions the inner
    loop over primitives can be vectorised by pre-allocating a
    (n_pts, n_prims) exponential array.  The implementation here favours
    clarity over speed; a production version would use numba or Cython.
    """
    n_pts   = points.shape[0]
    n_basis = len(shells)
    ao      = np.zeros((n_pts, n_basis), dtype=float)

    for mu, shell in enumerate(shells):
        A  = np.asarray(shell['center'], dtype=float)   # (3,)
        lx, ly, lz = shell['angular']
        exps  = shell['exponents']
        coefs = shell['coefficients']

        delta = points - A[None, :]                 # (n_pts, 3)
        dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]
        r2 = dx * dx + dy * dy + dz * dz           # (n_pts,)

        # Cartesian prefactor (angular part)
        ang = (dx ** lx) * (dy ** ly) * (dz ** lz) # (n_pts,)

        val = np.zeros(n_pts, dtype=float)
        for alpha, c in zip(exps, coefs):
            N = _norm_primitive(alpha, lx, ly, lz)
            val += N * c * np.exp(-alpha * r2)

        ao[:, mu] = ang * val

    return ao


# ---------------------------------------------------------------------------
# Density evaluation
# ---------------------------------------------------------------------------

def eval_density_on_grid(
    P: np.ndarray,
    ao_values: np.ndarray,
) -> np.ndarray:
    """
    Compute the electron density ρ(r) at each grid point.

    The density is:

        ρ(r_g) = Σ_μν P_μν φ_μ(r_g) φ_ν(r_g)
               = ao_g · P · ao_g^T    (scalar for each g)

    Vectorised over all grid points simultaneously:

        ρ = diag(ao_values @ P @ ao_values.T)
          = einsum('gm,mn,gn->g', ao, P, ao)

    which is equivalent to the element-wise product of ``ao_values @ P``
    and ``ao_values``, summed over the basis index.

    Parameters
    ----------
    P : ndarray, shape (n_basis, n_basis)
        Density matrix.
    ao_values : ndarray, shape (n_pts, n_basis)
        AO values at each grid point, as returned by eval_ao_on_grid.

    Returns
    -------
    rho : ndarray, shape (n_pts,)
        Electron density at each grid point.  Should be non-negative
        everywhere; small negative values may appear near grid-point
        singularities and should be clamped with np.maximum(rho, 0.0).

    Notes
    -----
    Tr[P S] = ∫ ρ(r) dr  ≈  Σ_g w_g ρ(r_g) = n_electrons, which
    provides a useful numerical sanity check on the grid accuracy.
    """
    # (n_pts, n_basis) @ (n_basis, n_basis) → (n_pts, n_basis)
    Pao = ao_values @ P                      # P_{μν} φ_ν(r_g), summed over ν
    rho = np.einsum('gm,gm->g', Pao, ao_values)  # φ_μ · (Pao)_μ, summed over μ
    return np.maximum(rho, 0.0)


# ---------------------------------------------------------------------------
# XC matrix assembly
# ---------------------------------------------------------------------------

def build_xc_matrix(
    ao_values: np.ndarray,
    v_xc: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Assemble the XC contribution to the Kohn-Sham matrix.

    The XC matrix element is:

        V_xc,μν = ∫ φ_μ(r) v_xc(r) φ_ν(r) dr
                ≈ Σ_g w_g v_xc(r_g) φ_μ(r_g) φ_ν(r_g)

    This is the DFT analogue of the G(P) two-electron matrix in
    Hartree-Fock.  It feeds directly into the Kohn-Sham Fock matrix:

        F_KS = H_core + J + V_xc

    (no exchange K term; exchange is subsumed into V_xc for pure DFT).

    Parameters
    ----------
    ao_values : ndarray, shape (n_pts, n_basis)
        AO values at each grid point (from eval_ao_on_grid).
    v_xc : ndarray, shape (n_pts,)
        XC potential v_xc(ρ(r_g)) at each grid point (from dft.xc).
    weights : ndarray, shape (n_pts,)
        Final molecular grid weights (from build_molecular_grid).

    Returns
    -------
    V_xc : ndarray, shape (n_basis, n_basis)
        Symmetric XC matrix in the AO basis.

    Notes
    -----
    The contraction is:

        V_xc,μν = Σ_g (w_g v_xc,g) φ_μ(r_g) φ_ν(r_g)
                = ao.T @ diag(w · v_xc) @ ao

    which is computed as:

        wv = w_g · v_xc,g          (n_pts,)
        ao_wv = ao * wv[:, None]   (n_pts, n_basis)   — scaled rows
        V_xc = ao_wv.T @ ao        (n_basis, n_basis)

    This avoids forming the large (n_pts, n_pts) diagonal matrix.
    """
    wv = weights * v_xc                        # (n_pts,)
    ao_wv = ao_values * wv[:, None]            # (n_pts, n_basis)
    V_xc = ao_wv.T @ ao_values                 # (n_basis, n_basis)
    return V_xc