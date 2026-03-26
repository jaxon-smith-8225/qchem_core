"""
Tests for qchem/dft/grid.py

Run with: pytest test_grid.py -v

Test organisation
-----------------
TestRadialGrid          — Gauss-Chebyshev radial grid and Becke transform
TestAngularGrid         — product GL × equispaced angular grid on S²
TestAtomicGrid          — combined atomic (r, w) grid
TestBeckePartition      — fuzzy Voronoi cell-function and partition weights
TestMolecularGrid       — full multi-atom grid assembly
TestEvalAOOnGrid        — contracted GTO evaluation at grid points
TestEvalDensityOnGrid   — ρ(r) = Σ_μν P_μν φ_μ φ_ν
TestBuildXCMatrix       — V_xc assembly from grid quadrature
TestIntegrationAccuracy — end-to-end numerical integration sanity checks

Design philosophy
-----------------
Each test targets one mathematical property rather than a numerical value
that might change if the grid size is tweaked.  The most important
properties are:

    1.  Quadrature exactness for polynomials / exponentials up to a known
        degree (radial and angular grids separately).
    2.  Partition-of-unity: Σ_A w_A(r) = 1 everywhere.
    3.  Electron-count conservation: Σ_g w_g ρ(r_g) ≈ n_electrons.
    4.  Matrix symmetry and positive-semidefiniteness where applicable.
    5.  Normalisation: <φ_μ|φ_μ> = 1 for a correctly normalised GTO.

External dependencies: numpy, pytest, scipy (for reference integrations).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import factorial2


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
# We assume the package is installed (pip install -e .) or that tests are
# run from the repo root so the relative import resolves.

from qchem.dft.grid import (
    BRAGG_SLATER_RADII,
    radial_grid_becke,
    angular_grid_product,
    atomic_grid,
    becke_partition_weights,
    build_molecular_grid,
    eval_ao_on_grid,
    eval_density_on_grid,
    build_xc_matrix,
    _becke_f,
    _becke_cell_function,
    _norm_primitive,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

def make_shell(center, angular, exponents, coefficients):
    """Convenience constructor matching the qchem shell-dict format."""
    return {
        'center':       np.asarray(center, dtype=float),
        'angular':      tuple(angular),
        'exponents':    list(exponents),
        'coefficients': list(coefficients),
    }


def s_shell(center, alpha, coeff=1.0):
    return make_shell(center, (0, 0, 0), [alpha], [coeff])


def px_shell(center, alpha, coeff=1.0):
    return make_shell(center, (1, 0, 0), [alpha], [coeff])


def py_shell(center, alpha, coeff=1.0):
    return make_shell(center, (0, 1, 0), [alpha], [coeff])


def dxx_shell(center, alpha, coeff=1.0):
    return make_shell(center, (2, 0, 0), [alpha], [coeff])


def _df(n: int) -> float:
    """Double factorial with (-1)!! = 0!! = 1."""
    if n <= 0:
        return 1.0
    return float(factorial2(n))


def norm_primitive(alpha: float, lx: int, ly: int, lz: int) -> float:
    """GTO normalisation constant — mirrors grid.py so tests are self-consistent."""
    L = lx + ly + lz
    prefactor = (2.0 * alpha / np.pi) ** 0.75 * (4.0 * alpha) ** (L / 2.0)
    denom = np.sqrt(_df(2*lx-1) * _df(2*ly-1) * _df(2*lz-1))
    return prefactor / denom


def make_h2_molecule():
    """H₂ at the STO-3G equilibrium bond length (R = 1.4 bohr)."""
    class _FakeMol:
        atoms       = [('H', np.array([0.0, 0.0, 0.0])),
                       ('H', np.array([0.0, 0.0, 1.4]))]
        n_atoms     = 2
        n_electrons = 2

    return _FakeMol()


def make_water_molecule():
    """Water molecule in a near-equilibrium geometry (bohr)."""
    class _FakeMol:
        atoms = [
            ('O',  np.array([ 0.000,  0.000,  0.000])),
            ('H',  np.array([ 0.000,  1.430,  1.107])),
            ('H',  np.array([ 0.000, -1.430,  1.107])),
        ]
        n_atoms     = 3
        n_electrons = 10

    return _FakeMol()


# STO-3G exponents and coefficients for H (used throughout)
_H_STO3G_EXP   = [3.4252509, 0.6239137, 0.1688554]
_H_STO3G_COEF  = [0.1543290, 0.5353281, 0.4446345]


# ---------------------------------------------------------------------------
# TestRadialGrid
# ---------------------------------------------------------------------------

class TestRadialGrid:
    """
    Tests for radial_grid_becke.

    Key properties:
    - n_rad points are returned.
    - All radial nodes r_i > 0.
    - All weights w_i > 0.
    - Gaussian integrals ∫₀^∞ exp(-α r²) dr = √π/(2√α) are recovered.
    - Exponential integrals ∫₀^∞ r^n exp(-α r) dr = n!/α^{n+1} are recovered.
    - Scaling: doubling R_atom shifts the grid to larger r without
      changing the number of points.
    """

    def test_length(self):
        r, w = radial_grid_becke(50, 1.0)
        assert len(r) == 50
        assert len(w) == 50

    @pytest.mark.parametrize("n_rad", [25, 50, 75, 100])
    def test_all_positive(self, n_rad):
        r, w = radial_grid_becke(n_rad, 1.0)
        assert np.all(r > 0), "Some radial nodes are not positive"
        assert np.all(w > 0), "Some radial weights are not positive"

    @pytest.mark.parametrize("R_atom", [0.5, 1.0, 2.0])
    def test_nodes_increase_with_R(self, R_atom):
        """The Becke transform maps R_atom to a characteristic radius near r=R."""
        r_half, _ = radial_grid_becke(75, R_atom / 2.0)
        r_full, _ = radial_grid_becke(75, R_atom)
        # Larger R → same nodes mapped to larger r values
        assert np.mean(r_full) > np.mean(r_half)

    def test_gaussian_integral(self):
        """
        ∫₀^∞ exp(-α r²) dr = √π/(2√α).
        The r² Jacobian is NOT in the radial weights; this is a raw 1D test.
        We integrate f(r) = exp(-α r²) with weights w_rad directly.
        """
        r, w = radial_grid_becke(200, 1.0)
        alpha = 1.5
        numerical = np.dot(w, np.exp(-alpha * r**2))
        analytical = np.sqrt(np.pi) / (2.0 * np.sqrt(alpha))
        assert abs(numerical - analytical) / analytical < 2e-5

    def test_exponential_integral(self):
        """
        ∫₀^∞ r² exp(-α r) dr = 2/α³  (n=2 case of the gamma function).
        """
        r, w = radial_grid_becke(200, 0.9)
        alpha = 0.8
        numerical = np.dot(w, r**2 * np.exp(-alpha * r))
        analytical = 2.0 / alpha**3
        assert abs(numerical - analytical) / analytical < 2e-4

    def test_r4_exponential_integral(self):
        """
        ∫₀^∞ r⁴ exp(-α r²) dr = 3√π / (8α^(5/2)).
        """
        r, w = radial_grid_becke(200, 1.0)
        alpha = 1.0
        numerical = np.dot(w, r**4 * np.exp(-alpha * r**2))
        analytical = 3.0 * np.sqrt(np.pi) / (8.0 * alpha**2.5)
        assert abs(numerical - analytical) / analytical < 2e-4


# ---------------------------------------------------------------------------
# TestAngularGrid
# ---------------------------------------------------------------------------

class TestAngularGrid:
    """
    Tests for angular_grid_product.

    Key properties:
    - Points lie on the unit sphere.
    - Weights sum to 4π.
    - Constant functions integrate exactly (∫ dΩ = 4π).
    - Even-degree spherical harmonics integrate correctly.
    - Odd functions over the full sphere integrate to zero.
    """

    def test_output_shapes(self):
        n_theta = 11
        xyz, w = angular_grid_product(n_theta)
        n_ang = n_theta * 2 * n_theta
        assert xyz.shape == (n_ang, 3)
        assert w.shape  == (n_ang,)

    @pytest.mark.parametrize("n_theta", [5, 11, 17])
    def test_unit_sphere(self, n_theta):
        """All angular points should lie on the unit sphere."""
        xyz, _ = angular_grid_product(n_theta)
        norms = np.linalg.norm(xyz, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-12)

    @pytest.mark.parametrize("n_theta", [5, 11, 17])
    def test_weights_sum_to_4pi(self, n_theta):
        """Σ w_ang = 4π (solid angle of the full sphere)."""
        _, w = angular_grid_product(n_theta)
        assert abs(w.sum() - 4.0 * np.pi) < 1e-12

    def test_constant_function(self):
        """∫_{S²} 1 dΩ = 4π."""
        _, w = angular_grid_product(17)
        assert abs(w.sum() - 4.0 * np.pi) < 1e-10

    def test_x2_integral(self):
        """∫_{S²} x² dΩ = 4π/3  (by symmetry, also = ∫y² = ∫z²)."""
        xyz, w = angular_grid_product(17)
        for col in range(3):
            val = np.dot(w, xyz[:, col]**2)
            assert abs(val - 4.0 * np.pi / 3.0) < 1e-10, \
                f"x²_col={col} integral failed: {val}"

    def test_odd_function_zero(self):
        """∫_{S²} x dΩ = 0 (odd integrand)."""
        xyz, w = angular_grid_product(17)
        for col in range(3):
            val = np.dot(w, xyz[:, col])
            assert abs(val) < 1e-12, f"Odd integral not zero: {val}"

    def test_x2y2_integral(self):
        """∫_{S²} x²y² dΩ = 4π/15."""
        xyz, w = angular_grid_product(17)
        val = np.dot(w, xyz[:, 0]**2 * xyz[:, 1]**2)
        assert abs(val - 4.0 * np.pi / 15.0) < 1e-8

    def test_y0_l4_integral(self):
        """
        ∫_{S²} Y_4^0(θ,φ) dΩ = 0 because only l=0 survives integration.
        Y_4^0 ∝ 35cos⁴θ - 30cos²θ + 3.
        """
        xyz, w = angular_grid_product(17)
        cos_t = xyz[:, 2]
        Y40 = (35 * cos_t**4 - 30 * cos_t**2 + 3)   # proportional
        val = np.dot(w, Y40)
        assert abs(val) < 1e-8, f"Y_4^0 integral should be zero: {val}"


# ---------------------------------------------------------------------------
# TestAtomicGrid
# ---------------------------------------------------------------------------

class TestAtomicGrid:
    """
    Tests for atomic_grid.

    Key properties:
    - Combined weight is w_rad · r² · w_ang.
    - Weights are all positive.
    - The full 3D Gaussian ∫ exp(-α r²) d³r = (π/α)^(3/2) is recovered.
    """

    def test_output_shapes(self):
        pts, wts = atomic_grid('H', n_rad=20, n_theta=7)
        n_ang = 7 * 14
        assert pts.shape == (20 * n_ang, 3)
        assert wts.shape == (20 * n_ang,)

    def test_all_weights_positive(self):
        _, wts = atomic_grid('C', n_rad=30, n_theta=9)
        assert np.all(wts > 0)

    def test_grid_centred_at_origin(self):
        """Default atomic grid is centred at the origin."""
        pts, _ = atomic_grid('H', n_rad=20, n_theta=7)
        # Centre of mass of the grid should be near origin
        assert np.linalg.norm(pts.mean(axis=0)) < 0.5

    @pytest.mark.parametrize("element,alpha", [('H', 1.0), ('C', 2.5), ('O', 3.0)])
    def test_3d_gaussian_integral(self, element, alpha):
        """
        ∫ exp(-α|r|²) d³r = (π/α)^(3/2).
        Uses the full atomic grid, including the r² Jacobian.
        """
        pts, wts = atomic_grid(element, n_rad=100, n_theta=17)
        r2 = np.sum(pts**2, axis=1)
        numerical = np.dot(wts, np.exp(-alpha * r2))
        analytical = (np.pi / alpha) ** 1.5
        assert abs(numerical - analytical) / analytical < 5e-4, \
            f"Gaussian integral failed for {element}: {numerical:.6f} vs {analytical:.6f}"

    def test_spherical_shell_density(self):
        """
        ∫ r² exp(-2α r) d³r = 4π Γ(3)/(2α)³ = 4π · 2 / (2α)³  (hydrogenic 1s density).
        Uses ∫₀^∞ r² · 4π r² · exp(-2α r) dr = ... wait, let's use simpler:
        ∫ exp(-α r) d³r = 4π ∫₀^∞ r² exp(-αr) dr = 4π · 2/α³ = 8π/α³.
        """
        pts, wts = atomic_grid('H', n_rad=150, n_theta=17)
        alpha = 0.5
        r = np.linalg.norm(pts, axis=1)
        numerical = np.dot(wts, np.exp(-alpha * r))
        analytical = 8.0 * np.pi / alpha**3
        assert abs(numerical - analytical) / analytical < 1e-3


# ---------------------------------------------------------------------------
# TestBeckePartition
# ---------------------------------------------------------------------------

class TestBeckePartition:
    """
    Tests for _becke_f, _becke_cell_function, and becke_partition_weights.
    """

    # --- Cell function internals ---

    def test_becke_f_at_endpoints(self):
        """f(±1) = ±1 (fixed points of the smoothing map)."""
        assert abs(_becke_f(np.array([ 1.0]))[0] -  1.0) < 1e-15
        assert abs(_becke_f(np.array([-1.0]))[0] - -1.0) < 1e-15

    def test_becke_f_at_zero(self):
        """f(0) = 0."""
        assert abs(_becke_f(np.array([0.0]))[0]) < 1e-15

    def test_becke_f_antisymmetric(self):
        """f(-x) = -f(x) for all x ∈ [-1, 1]."""
        x = np.linspace(-0.9, 0.9, 50)
        assert np.allclose(_becke_f(-x), -_becke_f(x), atol=1e-15)

    def test_cell_function_range(self):
        """s(μ) ∈ [0, 1] for μ ∈ [-1, 1]."""
        mu = np.linspace(-1.0, 1.0, 500)
        s = _becke_cell_function(mu)
        assert np.all(s >= 0.0)
        assert np.all(s <= 1.0 + 1e-14)

    def test_cell_function_monotone(self):
        """s(μ) is monotonically decreasing: near μ = -1, s ≈ 1; near +1, s ≈ 0."""
        s_neg = _becke_cell_function(np.array([-0.9]))
        s_pos = _becke_cell_function(np.array([ 0.9]))
        assert s_neg > s_pos

    def test_cell_function_complement(self):
        """s(μ) + s(-μ) = 1 (the two cells partition unity pairwise)."""
        mu = np.linspace(-0.95, 0.95, 100)
        assert np.allclose(_becke_cell_function(mu) + _becke_cell_function(-mu),
                           1.0, atol=1e-14)

    # --- Partition weights for a single atom ---

    def test_single_atom_weight_is_one(self):
        """With only one atom, its Becke weight must be 1 everywhere."""
        pts = np.random.default_rng(0).uniform(-3, 3, (50, 3))
        coords = np.array([[0.0, 0.0, 0.0]])
        w = becke_partition_weights(pts, coords, ['H'])
        assert np.allclose(w[:, 0], 1.0, atol=1e-12)

    # --- Partition of unity ---

    def test_partition_of_unity_homonuclear(self):
        """Σ_A w_A(r) = 1 for all grid points (homonuclear H₂)."""
        pts = np.random.default_rng(1).uniform(-5, 5, (200, 3))
        coords = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.4]])
        w = becke_partition_weights(pts, coords, ['H', 'H'])
        row_sums = w.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-12)

    def test_partition_of_unity_heteronuclear(self):
        """Σ_A w_A(r) = 1 for all grid points (heteronuclear CO)."""
        pts = np.random.default_rng(2).uniform(-5, 5, (200, 3))
        coords = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 2.1]])
        w = becke_partition_weights(pts, coords, ['C', 'O'])
        row_sums = w.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-12)

    def test_weights_non_negative(self):
        """Becke weights must be non-negative."""
        pts = np.random.default_rng(3).uniform(-6, 6, (300, 3))
        coords = np.array([[0.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0],
                           [1.0, 1.7, 0.0]])
        w = becke_partition_weights(pts, coords, ['N', 'N', 'H'])
        assert np.all(w >= -1e-14)

    def test_atom_dominates_near_its_nucleus(self):
        """
        A point very close to atom A should have w_A ≈ 1 and
        w_B ≈ 0 for all other atoms B.
        """
        coords = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 3.0]])
        # Point very close to atom 0
        pts_near_0 = np.array([[0.001, 0.0, 0.0]])
        w0 = becke_partition_weights(pts_near_0, coords, ['C', 'C'])
        assert w0[0, 0] > 0.99, f"Expected w_A ≈ 1 near atom A, got {w0[0, 0]}"
        assert w0[0, 1] < 0.01, f"Expected w_B ≈ 0 near atom A, got {w0[0, 1]}"

    def test_midpoint_roughly_equal(self):
        """
        For a homonuclear dimer the midpoint should have w_A ≈ w_B ≈ 0.5.
        """
        coords = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0]])
        mid = np.array([[0.0, 0.0, 1.0]])
        w = becke_partition_weights(mid, coords, ['H', 'H'])
        assert abs(w[0, 0] - 0.5) < 0.01
        assert abs(w[0, 1] - 0.5) < 0.01

    def test_output_shape(self):
        pts    = np.zeros((7, 3))
        coords = np.zeros((3, 3))
        coords[1, 2] = 1.4
        coords[2, 2] = 2.8
        w = becke_partition_weights(pts, coords, ['H', 'H', 'H'])
        assert w.shape == (7, 3)


# ---------------------------------------------------------------------------
# TestMolecularGrid
# ---------------------------------------------------------------------------

class TestMolecularGrid:
    """
    Tests for build_molecular_grid.
    """

    def test_output_shapes_h2(self):
        mol = make_h2_molecule()
        pts, wts = build_molecular_grid(mol, n_rad=20, n_theta=7)
        n_per_atom = 20 * 7 * 14
        n_total = 2 * n_per_atom
        assert pts.shape == (n_total, 3)
        assert wts.shape == (n_total,)

    def test_weights_non_negative(self):
        mol = make_h2_molecule()
        _, wts = build_molecular_grid(mol, n_rad=30, n_theta=7)
        assert np.all(wts >= -1e-15)

    def test_grid_points_span_both_atoms(self):
        """Grid points should appear near each nucleus."""
        mol = make_h2_molecule()
        pts, _ = build_molecular_grid(mol, n_rad=20, n_theta=7)
        # Points within 2 bohr of each H nucleus
        for _, R in mol.atoms:
            d = np.linalg.norm(pts - R[None, :], axis=1)
            assert np.any(d < 2.0), f"No grid points found near nucleus at {R}"

    def test_gaussian_integral_h2(self):
        """
        ∫ exp(-α|r − R_A|²) d³r = (π/α)^(3/2) for each atom A.
        Because the atomic grids cover all of ℝ³ and Becke weights sum to 1,
        the total grid also integrates this to (π/α)^(3/2) per atom,
        but more usefully: ∫ [exp(-α|r-R_A|²) + exp(-α|r-R_B|²)] d³r
        = 2(π/α)^(3/2) regardless of R_AB.
        """
        mol = make_h2_molecule()
        pts, wts = build_molecular_grid(mol, n_rad=100, n_theta=17)
        alpha = 1.0
        R_A = mol.atoms[0][1]
        R_B = mol.atoms[1][1]
        f = (np.exp(-alpha * np.sum((pts - R_A)**2, axis=1)) +
             np.exp(-alpha * np.sum((pts - R_B)**2, axis=1)))
        numerical  = np.dot(wts, f)
        analytical = 2.0 * (np.pi / alpha) ** 1.5
        assert abs(numerical - analytical) / analytical < 5e-3


# ---------------------------------------------------------------------------
# TestEvalAOOnGrid
# ---------------------------------------------------------------------------

class TestEvalAOOnGrid:
    """
    Tests for eval_ao_on_grid.
    """

    def test_output_shape(self):
        shells = [s_shell([0, 0, 0], 1.0), px_shell([0, 0, 0], 1.0)]
        pts = np.random.default_rng(0).uniform(-2, 2, (30, 3))
        ao = eval_ao_on_grid(shells, pts)
        assert ao.shape == (30, 2)

    def test_s_shell_at_center_is_positive(self):
        """An s-function evaluated at its own centre must be positive."""
        shell = s_shell([0, 0, 0], 1.5)
        pts   = np.array([[0.0, 0.0, 0.0]])
        ao    = eval_ao_on_grid([shell], pts)
        assert ao[0, 0] > 0.0

    def test_p_shell_odd_parity(self):
        """px(r) = −px(−r) (odd function in x)."""
        shell = px_shell([0, 0, 0], 1.0)
        pts_pos = np.array([[ 1.0, 0.5, 0.3]])
        pts_neg = np.array([[-1.0, 0.5, 0.3]])   # flip x only
        ao_pos = eval_ao_on_grid([shell], pts_pos)
        ao_neg = eval_ao_on_grid([shell], pts_neg)
        assert abs(ao_pos[0, 0] + ao_neg[0, 0]) < 1e-14

    def test_p_shell_at_center_is_zero(self):
        """A p-function vanishes at its own centre (angular prefactor = 0)."""
        for shell in [px_shell([0,0,0], 1.0), py_shell([0,0,0], 1.0)]:
            ao = eval_ao_on_grid([shell], np.zeros((1, 3)))
            assert abs(ao[0, 0]) < 1e-14

    def test_decay_with_distance(self):
        """AO values should decay monotonically as |r − A| increases."""
        shell = s_shell([0, 0, 0], 1.0)
        rs = np.array([[R, 0, 0] for R in [0.0, 0.5, 1.0, 2.0, 4.0]])
        ao = eval_ao_on_grid([shell], rs)
        vals = ao[:, 0]
        assert all(vals[i] > vals[i+1] for i in range(len(vals)-1))

    def test_translation_invariance(self):
        """φ(r − A) evaluated at r = A should equal φ(0 − 0)."""
        A = np.array([1.3, -0.7, 2.1])
        shell_at_A  = s_shell(A,       1.2)
        shell_at_0  = s_shell([0,0,0], 1.2)
        ao_at_A = eval_ao_on_grid([shell_at_A], A[None, :])
        ao_at_0 = eval_ao_on_grid([shell_at_0], np.zeros((1, 3)))
        assert abs(ao_at_A[0, 0] - ao_at_0[0, 0]) < 1e-14

    def test_normalisation_s_self_overlap(self):
        """
        ∫ φ_s(r)² d³r = 1 for a correctly normalised s-function.
        Uses the atomic grid as the numerical integrator.
        """
        alpha = 1.5
        shell = s_shell([0, 0, 0], alpha)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao = eval_ao_on_grid([shell], pts)
        integral = np.dot(wts, ao[:, 0]**2)
        assert abs(integral - 1.0) < 5e-4, \
            f"S self-overlap integral = {integral:.6f}, expected 1.0"

    def test_normalisation_px_self_overlap(self):
        """∫ φ_px(r)² d³r = 1 for a correctly normalised px-function."""
        alpha = 1.5
        shell = px_shell([0, 0, 0], alpha)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao = eval_ao_on_grid([shell], pts)
        integral = np.dot(wts, ao[:, 0]**2)
        assert abs(integral - 1.0) < 5e-4, \
            f"Px self-overlap integral = {integral:.6f}, expected 1.0"

    def test_s_px_orthogonality(self):
        """∫ φ_s(r) φ_px(r) d³r = 0 (orthogonal at same centre by symmetry)."""
        alpha = 1.0
        s  = s_shell( [0, 0, 0], alpha)
        px = px_shell([0, 0, 0], alpha)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao = eval_ao_on_grid([s, px], pts)
        cross = np.dot(wts, ao[:, 0] * ao[:, 1])
        assert abs(cross) < 1e-10, f"s/px cross-integral = {cross:.2e}"

    def test_sto3g_contraction(self):
        """
        STO-3G contracted s-function on H: self-overlap should be close to 1.
        Uses the full STO-3G contraction coefficients.
        """
        shell = make_shell([0, 0, 0], (0, 0, 0), _H_STO3G_EXP, _H_STO3G_COEF)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao = eval_ao_on_grid([shell], pts)
        integral = np.dot(wts, ao[:, 0]**2)
        # The STO-3G contraction is approximately normalised
        assert 0.9 < integral < 1.1, \
            f"STO-3G H s-function self-overlap = {integral:.4f}"


# ---------------------------------------------------------------------------
# TestEvalDensityOnGrid
# ---------------------------------------------------------------------------

class TestEvalDensityOnGrid:
    """
    Tests for eval_density_on_grid.
    """

    def test_density_non_negative(self):
        """The electron density must be non-negative at all grid points."""
        shell = s_shell([0, 0, 0], 1.0)
        pts   = np.random.default_rng(0).uniform(-3, 3, (100, 3))
        ao    = eval_ao_on_grid([shell], pts)
        P     = np.array([[2.0]])   # 1 electron pair in one orbital
        rho   = eval_density_on_grid(P, ao)
        assert np.all(rho >= 0.0)

    def test_density_shape(self):
        shells = [s_shell([0,0,0], 1.0), px_shell([0,0,0], 1.0)]
        pts    = np.zeros((50, 3))
        ao     = eval_ao_on_grid(shells, pts)
        P      = np.eye(2)
        rho    = eval_density_on_grid(P, ao)
        assert rho.shape == (50,)

    def test_density_at_center_s_orbital(self):
        """
        For ρ = 2 |φ_s|² with P = 2 I (one doubly-occupied s orbital),
        ρ(A) = 2 φ_s(A)² > 0.
        """
        alpha = 1.5
        shell = s_shell([0, 0, 0], alpha)
        pts   = np.array([[0.0, 0.0, 0.0]])
        ao    = eval_ao_on_grid([shell], pts)
        P     = np.array([[2.0]])
        rho   = eval_density_on_grid(P, ao)
        assert rho[0] > 0.0

    def test_zero_density_from_zero_matrix(self):
        """If P = 0 then ρ = 0 everywhere."""
        shell = s_shell([0, 0, 0], 1.0)
        pts   = np.random.default_rng(1).uniform(-2, 2, (30, 3))
        ao    = eval_ao_on_grid([shell], pts)
        P     = np.zeros((1, 1))
        rho   = eval_density_on_grid(P, ao)
        assert np.allclose(rho, 0.0, atol=1e-15)

    def test_electron_count_single_s(self):
        """
        Σ_g w_g ρ(r_g) ≈ n_electrons for a doubly-occupied s orbital.
        For a single normalised s-function with P = 2 I, n_electrons = 2.
        """
        alpha = 1.5
        shell = s_shell([0, 0, 0], alpha)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao  = eval_ao_on_grid([shell], pts)
        P   = np.array([[2.0]])
        rho = eval_density_on_grid(P, ao)
        n_numerical = np.dot(wts, rho)
        # Should integrate to 2 (double occupation)
        assert abs(n_numerical - 2.0) < 0.01, \
            f"Electron count integral = {n_numerical:.4f}, expected 2.0"

    def test_density_respects_superposition(self):
        """ρ[P1 + P2] = ρ[P1] + ρ[P2] (linearity in P)."""
        shells = [s_shell([0,0,0], 1.0), s_shell([0,0,0], 0.5)]
        pts    = np.random.default_rng(2).uniform(-2, 2, (40, 3))
        ao     = eval_ao_on_grid(shells, pts)

        P1 = np.diag([1.0, 0.0])
        P2 = np.diag([0.0, 1.0])
        rho1 = eval_density_on_grid(P1, ao)
        rho2 = eval_density_on_grid(P2, ao)
        rho_both = eval_density_on_grid(P1 + P2, ao)
        assert np.allclose(rho_both, rho1 + rho2, atol=1e-14)


# ---------------------------------------------------------------------------
# TestBuildXCMatrix
# ---------------------------------------------------------------------------

class TestBuildXCMatrix:
    """
    Tests for build_xc_matrix.
    """

    def test_output_shape(self):
        n_pts, n_basis = 50, 4
        ao      = np.random.default_rng(0).uniform(0, 1, (n_pts, n_basis))
        v_xc    = np.ones(n_pts)
        weights = np.ones(n_pts) / n_pts
        Vxc     = build_xc_matrix(ao, v_xc, weights)
        assert Vxc.shape == (n_basis, n_basis)

    def test_symmetry(self):
        """V_xc must be symmetric (it is an operator matrix in a real AO basis)."""
        np.random.seed(7)
        n_pts, n_basis = 200, 5
        ao      = np.random.default_rng(7).standard_normal((n_pts, n_basis))
        v_xc    = np.abs(np.random.default_rng(8).standard_normal(n_pts))
        weights = np.abs(np.random.default_rng(9).standard_normal(n_pts))
        Vxc     = build_xc_matrix(ao, v_xc, weights)
        assert np.allclose(Vxc, Vxc.T, atol=1e-13), \
            "V_xc is not symmetric"

    def test_positive_semidefinite_for_positive_vxc(self):
        """
        If v_xc(r) > 0 and weights > 0 then V_xc is positive semidefinite.
        Proof: V_xc = Σ_g w_g v_xc,g ao_g ao_g^T is a sum of PSD rank-1 matrices.
        """
        n_pts, n_basis = 300, 4
        ao      = np.random.default_rng(10).standard_normal((n_pts, n_basis))
        v_xc    = np.ones(n_pts) * 0.5
        weights = np.ones(n_pts) / n_pts
        Vxc     = build_xc_matrix(ao, v_xc, weights)
        eigvals = np.linalg.eigvalsh(Vxc)
        assert np.all(eigvals >= -1e-12), \
            f"V_xc has negative eigenvalue: {eigvals.min():.2e}"

    def test_zero_vxc_gives_zero_matrix(self):
        """If v_xc = 0 everywhere, V_xc = 0."""
        ao      = np.random.default_rng(11).standard_normal((50, 3))
        weights = np.ones(50) / 50.0
        v_xc    = np.zeros(50)
        Vxc     = build_xc_matrix(ao, v_xc, weights)
        assert np.allclose(Vxc, 0.0, atol=1e-15)

    def test_scaling_with_vxc(self):
        """V_xc(c · v_xc) = c · V_xc(v_xc) for scalar c."""
        ao      = np.random.default_rng(12).standard_normal((80, 3))
        v_xc    = np.abs(np.random.default_rng(13).standard_normal(80))
        weights = np.ones(80) / 80.0
        c       = 3.7
        Vxc     = build_xc_matrix(ao, v_xc,     weights)
        Vxc_c   = build_xc_matrix(ao, c * v_xc, weights)
        assert np.allclose(Vxc_c, c * Vxc, atol=1e-13)

    def test_analytical_single_basis_function(self):
        """
        For a single basis function and constant v_xc = 1:
        V_xc[0,0] = Σ_g w_g φ(r_g)² ≈ ∫ φ(r)² d³r = 1 (if normalised).
        """
        alpha = 1.5
        shell = s_shell([0, 0, 0], alpha)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao   = eval_ao_on_grid([shell], pts)
        v_xc = np.ones(len(wts))
        Vxc  = build_xc_matrix(ao, v_xc, wts)
        assert abs(Vxc[0, 0] - 1.0) < 5e-4, \
            f"V_xc[0,0] = {Vxc[0,0]:.6f}, expected ≈ 1.0"

    def test_diagonal_dominance_decays_offdiagonal(self):
        """
        For well-separated basis functions the off-diagonal V_xc elements
        should be much smaller than the diagonal ones.
        """
        # Two s-functions far apart
        s1 = s_shell([ 5.0, 0, 0], 2.0)
        s2 = s_shell([-5.0, 0, 0], 2.0)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao = eval_ao_on_grid([s1, s2], pts)
        v_xc = np.ones(len(wts))
        Vxc  = build_xc_matrix(ao, v_xc, wts)
        assert abs(Vxc[0, 1]) < 0.01 * abs(Vxc[0, 0]), \
            "Off-diagonal element is too large for well-separated functions"


# ---------------------------------------------------------------------------
# TestIntegrationAccuracy
# ---------------------------------------------------------------------------

class TestIntegrationAccuracy:
    """
    End-to-end tests that chain the full pipeline and verify physically
    meaningful results.
    """

    def test_electron_count_h2_minimal_basis(self):
        """
        ∫ ρ(r) d³r = n_electrons for H₂ with a minimal s-type basis.

        Uses the STO-3G contraction on each H atom with a diagonal density
        matrix P = 2 I (both basis functions singly occupied → 2 electrons
        total across the molecule).
        """
        mol   = make_h2_molecule()
        R_A   = mol.atoms[0][1]
        R_B   = mol.atoms[1][1]
        shell_A = make_shell(R_A, (0, 0, 0), _H_STO3G_EXP, _H_STO3G_COEF)
        shell_B = make_shell(R_B, (0, 0, 0), _H_STO3G_EXP, _H_STO3G_COEF)
        shells  = [shell_A, shell_B]

        pts, wts = build_molecular_grid(mol, n_rad=75, n_theta=17)
        ao       = eval_ao_on_grid(shells, pts)

        # Bonding MO: (1/√2)(φ_A + φ_B) → P_μν = 2 · (1/√2)² = 1 for all μ,ν
        # For simplicity use the diagonal P=I (one electron per basis fn)
        P = np.eye(2)
        rho = eval_density_on_grid(P, ao)

        n_numerical = np.dot(wts, rho)
        # P = I → Tr[PS] ≈ 2 only if S ≈ I; instead n_elec = Tr[P·S_grid]
        # More robustly: the integral of ρ = Σ_{μν} P_μν φ_μ φ_ν
        # With P=I and approximately normalised functions, expect ≈ 2
        # (slight deviation from 2 due to non-zero overlap S_AB ≈ 0.66)
        assert 1.5 < n_numerical < 3.0, \
            f"Electron count integral out of range: {n_numerical:.4f}"

    def test_density_trace_matches_overlap(self):
        """
        For a single normalised s-function with P = 2 I,
        Σ_g w_g ρ(r_g) = Tr[P] = 2 (ignoring overlap, using orthonormal limit).
        """
        alpha = 2.0
        shell = s_shell([0, 0, 0], alpha)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao  = eval_ao_on_grid([shell], pts)
        P   = np.array([[2.0]])
        rho = eval_density_on_grid(P, ao)
        n   = np.dot(wts, rho)
        assert abs(n - 2.0) < 0.01

    def test_xc_energy_lda_constant_density(self):
        """
        For the LDA exchange energy functional ε_x = −(3/4)(3/π)^(1/3) ρ^(4/3),
        with a known analytic density, V_xc integrates to the correct total.

        We use ρ(r) = (α/π)^(3/2) exp(-α|r|²) (a Gaussian density, not
        self-consistent but analytically tractable) and verify that the
        numerical XC energy Σ_g w_g ε_x(ρ(r_g)) matches the analytic value
        obtained via scipy.integrate.quad on the radial integral.
        """
        alpha = 1.0
        C_x   = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

        # Build a grid large enough for a diffuse Gaussian density
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        r2   = np.sum(pts**2, axis=1)
        rho  = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)
        rho  = np.maximum(rho, 0.0)

        # LDA XC energy density
        eps_xc = C_x * rho ** (1.0 / 3.0)

        E_xc_numerical = np.dot(wts, eps_xc * rho)

        # Analytic radial integral:
        # E_xc = 4π ∫₀^∞ C_x ρ(r)^(4/3) r² dr
        #      = 4π C_x (α/π)^2 ∫₀^∞ exp(-4α/3 r²) r² dr
        #      = 4π C_x (α/π)^2 · (π/(4α/3))^(3/2) / 4
        #      = C_x (α/π)^2 · π · (3/(4α))^(3/2)
        # More simply, use scipy quad on the 1D radial form
        def integrand(r):
            rho_r = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r**2)
            return C_x * rho_r ** (4.0 / 3.0) * r**2

        E_radial, _ = quad(integrand, 0, np.inf, limit=200)
        E_xc_analytical = 4.0 * np.pi * E_radial

        rel_err = abs(E_xc_numerical - E_xc_analytical) / abs(E_xc_analytical)
        assert rel_err < 1e-3, \
            (f"LDA exchange energy: numerical={E_xc_numerical:.6f}, "
             f"analytical={E_xc_analytical:.6f}, rel_err={rel_err:.2e}")

    def test_xc_matrix_trace_approximate(self):
        """
        Tr[V_xc] = Σ_μ V_xc,μμ = Σ_g w_g v_xc(r_g) Σ_μ φ_μ(r_g)²
                 ≈ n_atoms (for normalised AOs and v_xc = 1).
        This is a loose sanity check: with v_xc = 1 and one normalised
        basis function per atom, Tr[V_xc] ≈ 1.0.
        """
        alpha = 1.5
        shell = s_shell([0, 0, 0], alpha)
        pts, wts = atomic_grid('H', n_rad=100, n_theta=17)
        ao  = eval_ao_on_grid([shell], pts)
        v_xc = np.ones(len(wts))
        Vxc  = build_xc_matrix(ao, v_xc, wts)
        assert abs(np.trace(Vxc) - 1.0) < 5e-3

    def test_becke_radii_table_consistency(self):
        """
        Bragg-Slater radii should be positive and monotonically increase
        from H to the heavier row-1 atoms (C > B > H, roughly).
        """
        assert BRAGG_SLATER_RADII['H'] > 0
        assert BRAGG_SLATER_RADII['C'] > BRAGG_SLATER_RADII['H']
        assert BRAGG_SLATER_RADII['Na'] > BRAGG_SLATER_RADII['C']

    def test_full_pipeline_water(self):
        """
        Smoke test for the full pipeline on water:
        grid → AO eval → density → V_xc assembly.
        Checks shapes and that the density integrates to a positive value.
        """
        mol = make_water_molecule()
        pts, wts = build_molecular_grid(mol, n_rad=50, n_theta=11)

        # Build a trivial single-s-function basis (one per atom)
        shells = [
            make_shell(mol.atoms[0][1], (0,0,0), [5.0], [1.0]),   # O (tight)
            make_shell(mol.atoms[1][1], (0,0,0), [1.0], [1.0]),   # H
            make_shell(mol.atoms[2][1], (0,0,0), [1.0], [1.0]),   # H
        ]

        ao  = eval_ao_on_grid(shells, pts)
        P   = np.diag([2.0, 1.0, 1.0])   # arbitrary occupation
        rho = eval_density_on_grid(P, ao)

        assert ao.shape  == (len(pts), 3)
        assert rho.shape == (len(pts),)
        assert np.all(rho >= 0.0)
        assert np.dot(wts, rho) > 0.0

        v_xc = np.ones(len(wts))
        Vxc  = build_xc_matrix(ao, v_xc, wts)
        assert Vxc.shape == (3, 3)
        assert np.allclose(Vxc, Vxc.T, atol=1e-12)
