"""
Tests for qchem/dft/ks.py

Structure
---------
TestInputValidation       — open-shell, zero electrons, unknown basis,
                            unknown functional, convergence failure
TestResultFields          — shape, type, flag, metadata, energy split
TestKSResult              — convenience properties: HOMO, LUMO, gap, repr,
                            KS-specific fields (e_xc, grid_electrons, functional)
TestAOGradients           — analytic ∇φ_μ vs central finite difference
TestDensityGradient       — analytic ∇ρ vs central finite difference
TestPhysicalH2            — LDA and PBE reference energies for H₂/STO-3G
TestPhysicalWater         — LDA and PBE reference energies for H₂O/STO-3G
TestPhysicalHe            — LDA and PBE for He (single atom, no V_nn)
TestPhysicalLiH           — LDA and PBE for LiH (heteronuclear Becke weights)
TestGridQuality           — integrated electron count, grid size, grid diagnostic
TestConvergenceBehaviour  — DIIS vs plain, tolerance control, grid resolution
TestSCFProperties         — commutator, idempotency, self-consistency, symmetry
TestInvariants            — energy additivity, reproducibility, LDA < HF ordering,
                            bond-length ordering, density PSD, E_xc sign

Reference values
----------------
All energies from a converged internal calculation at default grid
(n_rad=75, n_theta=17) with STO-3G basis.

H₂ (R = 1.4 bohr)
    LDA  E_total  = −1.15699428 Ha   E_xc  = −0.71487405 Ha
    PBE  E_total  = −1.18117282 Ha   E_xc  = −0.73905259 Ha

H₂O (STO-3G geometry, bohr)
    LDA  E_total  = −74.92805638 Ha  E_xc  = −9.07359830 Ha
    PBE  E_total  = −75.40176011 Ha  E_xc  = −9.54395291 Ha

He
    LDA  E_total  = −2.80959860 Ha   E_xc  = −1.05752762 Ha
    PBE  E_total  = −2.86046562 Ha   E_xc  = −1.10839464 Ha

LiH (R = 3.015 bohr)
    LDA  E_total  = −7.86564730 Ha   E_xc  = −2.15936574 Ha
    PBE  E_total  = −7.98077007 Ha   E_xc  = −2.28185406 Ha
"""

import numpy as np
import pytest

from qchem.molecule import Molecule
from qchem.dft.ks import (
    ks,
    KSResult,
    eval_ao_gradients,
    eval_density_gradient,
    build_coulomb_matrix,
    build_vxc_matrix_gga,
    ks_fock_matrix,
    ks_electronic_energy,
)


# ---------------------------------------------------------------------------
# Shared molecule helpers
# ---------------------------------------------------------------------------

def _h2(bond_bohr: float = 1.4) -> Molecule:
    r = bond_bohr / 2.0
    return Molecule([('H', [0., 0., -r]), ('H', [0., 0., r])])


def _water() -> Molecule:
    return Molecule([
        ('O', [0.000,  0.000,  0.000]),
        ('H', [0.000,  1.430,  1.107]),
        ('H', [0.000, -1.430,  1.107]),
    ])


def _he() -> Molecule:
    return Molecule([('He', [0., 0., 0.])])


def _lih() -> Molecule:
    return Molecule([('Li', [0., 0., 0.]), ('H', [0., 0., 3.015])])


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_open_shell_raises(self):
        """Doublet radical must be rejected — KS driver is closed-shell only."""
        mol = Molecule(
            [('O', [0., 0., 0.]), ('H', [0., 0., 1.8])],
            multiplicity=2,
        )
        with pytest.raises(ValueError, match="closed-shell"):
            ks(mol)

    def test_triplet_raises(self):
        mol = Molecule([('O', [0., 0., 0.])], multiplicity=3)
        with pytest.raises(ValueError, match="closed-shell"):
            ks(mol)

    def test_unknown_basis_raises(self):
        with pytest.raises(KeyError):
            ks(_h2(), basis_name='not-a-basis')

    def test_unknown_functional_raises(self):
        """Unrecognised functional name must raise ValueError from get_xc."""
        with pytest.raises(ValueError, match="[Uu]nknown"):
            ks(_h2(), functional='b3lyp')

    def test_unknown_functional_error_lists_recognised(self):
        """The error message should mention the valid functional names."""
        with pytest.raises(ValueError) as exc_info:
            ks(_h2(), functional='b3lyp')
        msg = str(exc_info.value)
        assert 'lda' in msg.lower() or 'pbe' in msg.lower()

    def test_convergence_failure_raises_runtime_error(self):
        """max_iter=1 should not converge for water."""
        with pytest.raises(RuntimeError, match="[Cc]onverg"):
            ks(_water(), max_iter=1)

    def test_convergence_error_message_contains_diagnostics(self):
        """RuntimeError should include diagnostic numbers, not just a bare message."""
        with pytest.raises(RuntimeError) as exc_info:
            ks(_water(), max_iter=1)
        msg = str(exc_info.value)
        assert 'e-' in msg.lower() or any(c.isdigit() for c in msg)

    def test_svwn_alias_accepted(self):
        """'svwn' is an alias for LDA; must not raise."""
        result = ks(_h2(), functional='svwn')
        assert result.converged

    def test_functional_name_case_insensitive(self):
        """Functional name should be normalised case-insensitively."""
        r_lower = ks(_h2(), functional='lda')
        r_upper = ks(_h2(), functional='LDA')
        assert r_lower.e_total == pytest.approx(r_upper.e_total, rel=1e-12)

    def test_tight_tolerance_still_converges(self):
        result = ks(_h2(), e_tol=1e-11, d_tol=1e-9)
        assert result.converged

    def test_loose_tolerance_converges_faster_than_tight(self):
        r_tight = ks(_water(), e_tol=1e-8,  d_tol=1e-6)
        r_loose = ks(_water(), e_tol=1e-4,  d_tol=1e-3)
        assert r_loose.n_iter <= r_tight.n_iter


# ---------------------------------------------------------------------------
# TestResultFields
# ---------------------------------------------------------------------------

class TestResultFields:

    @pytest.fixture(scope="class")
    def h2_lda(self):
        return ks(_h2(), functional='lda')

    def test_returns_ksresult(self, h2_lda):
        assert isinstance(h2_lda, KSResult)

    def test_converged_flag_is_true(self, h2_lda):
        assert h2_lda.converged is True

    def test_orbital_energies_shape(self, h2_lda):
        """H₂/STO-3G has 2 basis functions → 2 orbital energies."""
        assert h2_lda.orbital_energies.shape == (2,)

    def test_coefficients_shape(self, h2_lda):
        assert h2_lda.coefficients.shape == (2, 2)

    def test_density_shape(self, h2_lda):
        assert h2_lda.density.shape == (2, 2)

    def test_fock_shape(self, h2_lda):
        assert h2_lda.fock.shape == (2, 2)

    def test_overlap_shape(self, h2_lda):
        assert h2_lda.overlap.shape == (2, 2)

    def test_h_core_shape(self, h2_lda):
        assert h2_lda.h_core.shape == (2, 2)

    def test_e_total_is_float(self, h2_lda):
        assert isinstance(h2_lda.e_total, float)

    def test_e_electronic_is_float(self, h2_lda):
        assert isinstance(h2_lda.e_electronic, float)

    def test_e_nuclear_is_float(self, h2_lda):
        assert isinstance(h2_lda.e_nuclear, float)

    def test_e_xc_is_float(self, h2_lda):
        assert isinstance(h2_lda.e_xc, float)

    def test_n_iter_positive_int(self, h2_lda):
        assert isinstance(h2_lda.n_iter, int)
        assert h2_lda.n_iter >= 1

    def test_n_occ_correct_h2(self, h2_lda):
        """H₂ has 2 electrons → 1 occupied spatial orbital."""
        assert h2_lda.n_occ == 1

    def test_n_occ_correct_water(self):
        r = ks(_water(), functional='lda')
        assert r.n_occ == 5

    def test_basis_name_stored(self, h2_lda):
        assert h2_lda.basis_name == 'sto-3g'

    def test_functional_stored_lowercase(self, h2_lda):
        assert h2_lda.functional == 'lda'

    def test_functional_stored_pbe(self):
        r = ks(_h2(), functional='PBE')
        assert r.functional == 'pbe'

    def test_mol_stored(self, h2_lda):
        assert isinstance(h2_lda.mol, Molecule)

    def test_n_grid_points_positive(self, h2_lda):
        assert isinstance(h2_lda.n_grid_points, int)
        assert h2_lda.n_grid_points > 0

    def test_grid_electrons_is_float(self, h2_lda):
        assert isinstance(h2_lda.grid_electrons, float)

    def test_energy_split_adds_up(self, h2_lda):
        """E_total = E_electronic + E_nuclear exactly."""
        assert h2_lda.e_total == pytest.approx(
            h2_lda.e_electronic + h2_lda.e_nuclear, rel=1e-12
        )

    def test_water_orbital_energies_shape(self):
        r = ks(_water(), functional='lda')
        assert r.orbital_energies.shape == (7,)

    def test_electron_count_from_density(self, h2_lda):
        """Tr[PS] must equal n_electrons."""
        from qchem.scf.density import n_electrons_from_density
        n_e = n_electrons_from_density(h2_lda.density, h2_lda.overlap)
        assert n_e == pytest.approx(2.0, rel=1e-8)


# ---------------------------------------------------------------------------
# TestKSResult (convenience properties and KS-specific fields)
# ---------------------------------------------------------------------------

class TestKSResult:

    @pytest.fixture(scope="class")
    def h2_lda(self):
        return ks(_h2(), functional='lda')

    def test_homo_energy_equals_orbital_at_n_occ_minus_1(self, h2_lda):
        assert h2_lda.homo_energy == pytest.approx(
            h2_lda.orbital_energies[h2_lda.n_occ - 1]
        )

    def test_lumo_energy_equals_orbital_at_n_occ(self, h2_lda):
        assert h2_lda.lumo_energy == pytest.approx(
            h2_lda.orbital_energies[h2_lda.n_occ]
        )

    def test_homo_lumo_gap_positive(self, h2_lda):
        assert h2_lda.homo_lumo_gap > 0.0

    def test_homo_lumo_gap_equals_difference(self, h2_lda):
        assert h2_lda.homo_lumo_gap == pytest.approx(
            h2_lda.lumo_energy - h2_lda.homo_energy
        )

    def test_lumo_none_when_fully_occupied(self):
        """He/STO-3G has 1 basis function and 1 occupied orbital — no LUMO."""
        result = ks(_he(), functional='lda')
        assert result.lumo_energy is None
        assert result.homo_lumo_gap is None

    def test_repr_contains_converged(self, h2_lda):
        assert "converged" in repr(h2_lda).lower()

    def test_repr_contains_energy(self, h2_lda):
        assert "Ha" in repr(h2_lda)

    def test_repr_contains_functional(self, h2_lda):
        assert "lda" in repr(h2_lda)

    def test_repr_contains_basis(self, h2_lda):
        assert "sto-3g" in repr(h2_lda)

    def test_e_xc_negative_lda(self, h2_lda):
        """LDA exchange-correlation is always negative for physical densities."""
        assert h2_lda.e_xc < 0.0

    def test_e_xc_negative_pbe(self):
        r = ks(_h2(), functional='pbe')
        assert r.e_xc < 0.0

    def test_e_xc_pbe_more_negative_than_lda(self):
        """PBE adds gradient corrections that typically make E_xc more negative."""
        lda = ks(_h2(), functional='lda')
        pbe = ks(_h2(), functional='pbe')
        assert pbe.e_xc < lda.e_xc

    def test_n_grid_points_scales_with_atoms(self):
        """More atoms → more grid points (each atom has its own atomic grid)."""
        he_r  = ks(_he(),    functional='lda')
        h2_r  = ks(_h2(),   functional='lda')
        h2o_r = ks(_water(), functional='lda')
        assert he_r.n_grid_points < h2_r.n_grid_points < h2o_r.n_grid_points


# ---------------------------------------------------------------------------
# TestAOGradients
# ---------------------------------------------------------------------------

class TestAOGradients:
    """
    Verify eval_ao_gradients against central finite differences.

    The finite-difference step h = 1e-5 bohr gives truncation error O(h²)
    ≈ 1e-10; we therefore accept agreement to 1e-8.
    """

    FD_STEP = 1e-5
    ATOL    = 1e-8

    @pytest.fixture(scope="class")
    def basis_and_points(self):
        mol   = _h2()
        basis = mol.build_basis('sto-3g')
        pts   = np.array([
            [0.1,  0.2,  0.3],
            [0.5, -0.1,  0.2],
            [0.0,  0.0,  0.0],   # at the midpoint — tests lx=0 guards
            [0.0,  0.0,  0.7],   # near nucleus
        ])
        from qchem.dft.grid import eval_ao_on_grid
        ao   = eval_ao_on_grid(basis, pts)
        grad = eval_ao_gradients(basis, pts)
        return basis, pts, ao, grad

    def _fd_gradient(self, basis, pts, c):
        """Central finite-difference gradient in direction c."""
        from qchem.dft.grid import eval_ao_on_grid
        h   = self.FD_STEP
        pp  = pts.copy(); pp[:, c] += h
        pm  = pts.copy(); pm[:, c] -= h
        return (eval_ao_on_grid(basis, pp) - eval_ao_on_grid(basis, pm)) / (2.0 * h)

    def test_x_gradient_shape(self, basis_and_points):
        basis, pts, ao, grad = basis_and_points
        assert grad.shape == (pts.shape[0], len(basis), 3)

    def test_x_gradient(self, basis_and_points):
        basis, pts, ao, grad = basis_and_points
        fd = self._fd_gradient(basis, pts, 0)
        np.testing.assert_allclose(grad[:, :, 0], fd, atol=self.ATOL)

    def test_y_gradient(self, basis_and_points):
        basis, pts, ao, grad = basis_and_points
        fd = self._fd_gradient(basis, pts, 1)
        np.testing.assert_allclose(grad[:, :, 1], fd, atol=self.ATOL)

    def test_z_gradient(self, basis_and_points):
        basis, pts, ao, grad = basis_and_points
        fd = self._fd_gradient(basis, pts, 2)
        np.testing.assert_allclose(grad[:, :, 2], fd, atol=self.ATOL)

    def test_p_functions_gradient(self):
        """Water has p-functions (lx, ly, lz > 0) — test those too."""
        from qchem.dft.grid import eval_ao_on_grid
        mol   = _water()
        basis = mol.build_basis('sto-3g')
        pts   = np.array([[0.2, -0.3, 0.4], [0.0, 0.5, -0.1]])
        grad  = eval_ao_gradients(basis, pts)
        h     = self.FD_STEP
        for c in range(3):
            pp  = pts.copy(); pp[:, c] += h
            pm  = pts.copy(); pm[:, c] -= h
            fd  = (eval_ao_on_grid(basis, pp) - eval_ao_on_grid(basis, pm)) / (2.0 * h)
            np.testing.assert_allclose(grad[:, :, c], fd, atol=self.ATOL,
                                       err_msg=f"p-function gradient direction {c}")

    def test_gradient_zero_at_symmetry_point(self):
        """
        For H₂ along z, the x and y AO gradients must be zero at any
        point on the z-axis due to cylindrical symmetry.
        """
        mol   = _h2()
        basis = mol.build_basis('sto-3g')
        pts   = np.array([[0., 0., 0.1], [0., 0., 0.5]])
        grad  = eval_ao_gradients(basis, pts)
        np.testing.assert_allclose(grad[:, :, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(grad[:, :, 1], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# TestDensityGradient
# ---------------------------------------------------------------------------

class TestDensityGradient:
    """Verify eval_density_gradient against central finite differences on ρ."""

    FD_STEP = 1e-5
    ATOL    = 1e-8

    @pytest.fixture(scope="class")
    def setup(self):
        from qchem.scf.hartree_fock import rhf
        from qchem.dft.grid import eval_ao_on_grid, eval_density_on_grid
        mol   = _water()
        basis = mol.build_basis('sto-3g')
        P     = rhf(mol).density
        pts   = np.array([[0.1, 0.2, 0.3], [0.5, -0.1, 0.2], [0.2, 0.0, 0.0]])
        ao    = eval_ao_on_grid(basis, pts)
        aog   = eval_ao_gradients(basis, pts)
        return basis, P, pts, ao, aog

    def _fd_grad_rho(self, basis, P, pts, c):
        from qchem.dft.grid import eval_ao_on_grid, eval_density_on_grid
        h   = self.FD_STEP
        pp  = pts.copy(); pp[:, c] += h
        pm  = pts.copy(); pm[:, c] -= h
        rp  = eval_density_on_grid(P, eval_ao_on_grid(basis, pp))
        rm  = eval_density_on_grid(P, eval_ao_on_grid(basis, pm))
        return (rp - rm) / (2.0 * h)

    def test_shape(self, setup):
        basis, P, pts, ao, aog = setup
        grad_rho = eval_density_gradient(P, ao, aog)
        assert grad_rho.shape == (pts.shape[0], 3)

    def test_x_component(self, setup):
        basis, P, pts, ao, aog = setup
        analytic = eval_density_gradient(P, ao, aog)[:, 0]
        fd       = self._fd_grad_rho(basis, P, pts, 0)
        np.testing.assert_allclose(analytic, fd, atol=self.ATOL)

    def test_y_component(self, setup):
        basis, P, pts, ao, aog = setup
        analytic = eval_density_gradient(P, ao, aog)[:, 1]
        fd       = self._fd_grad_rho(basis, P, pts, 1)
        np.testing.assert_allclose(analytic, fd, atol=self.ATOL)

    def test_z_component(self, setup):
        basis, P, pts, ao, aog = setup
        analytic = eval_density_gradient(P, ao, aog)[:, 2]
        fd       = self._fd_grad_rho(basis, P, pts, 2)
        np.testing.assert_allclose(analytic, fd, atol=self.ATOL)

    def test_grad_rho_nonnegative_norm(self, setup):
        """‖∇ρ‖ must be non-negative everywhere (trivial, but a shape guard)."""
        basis, P, pts, ao, aog = setup
        grad_rho = eval_density_gradient(P, ao, aog)
        assert np.all(np.linalg.norm(grad_rho, axis=1) >= 0.0)


# ---------------------------------------------------------------------------
# TestPhysicalH2
# ---------------------------------------------------------------------------

class TestPhysicalH2:
    """Reference energies from a converged internal calculation at n_rad=75."""

    @pytest.fixture(scope="class")
    def lda(self):
        return ks(_h2(), functional='lda')

    @pytest.fixture(scope="class")
    def pbe(self):
        return ks(_h2(), functional='pbe')

    # --- LDA ---

    def test_lda_total_energy(self, lda):
        assert lda.e_total == pytest.approx(-1.1569942787, rel=1e-7)

    def test_lda_xc_energy(self, lda):
        assert lda.e_xc == pytest.approx(-0.7148740519, rel=1e-6)

    def test_lda_nuclear_repulsion(self, lda):
        assert lda.e_nuclear == pytest.approx(1.0 / 1.4, rel=1e-10)

    def test_lda_homo_energy(self, lda):
        assert lda.homo_energy == pytest.approx(-0.36599305, rel=1e-5)

    def test_lda_lumo_energy(self, lda):
        assert lda.lumo_energy == pytest.approx(0.38165427, rel=1e-5)

    def test_lda_homo_lumo_gap_positive(self, lda):
        assert lda.homo_lumo_gap > 0.0

    def test_lda_orbital_energies_ascending(self, lda):
        assert lda.orbital_energies[0] < lda.orbital_energies[1]

    def test_lda_converged(self, lda):
        assert lda.converged

    # --- PBE ---

    def test_pbe_total_energy(self, pbe):
        assert pbe.e_total == pytest.approx(-1.1811728164, rel=1e-7)

    def test_pbe_xc_energy(self, pbe):
        assert pbe.e_xc == pytest.approx(-0.7390525895, rel=1e-6)

    def test_pbe_homo_energy(self, pbe):
        assert pbe.homo_energy == pytest.approx(-0.37655623, rel=1e-5)

    def test_pbe_converged(self, pbe):
        assert pbe.converged

    # --- LDA vs PBE comparison ---

    def test_pbe_lower_than_lda(self, lda, pbe):
        """PBE adds gradient corrections and gives a lower energy than LDA for H₂."""
        assert pbe.e_total < lda.e_total

    def test_bond_length_energy_ordering(self):
        """Energy should be lowest at equilibrium, higher compressed or stretched."""
        r_eq   = ks(_h2(bond_bohr=1.4), functional='lda')
        r_str  = ks(_h2(bond_bohr=4.0), functional='lda')
        r_comp = ks(_h2(bond_bohr=0.8), functional='lda')
        assert r_str.e_total  > r_eq.e_total
        assert r_comp.e_total > r_eq.e_total


# ---------------------------------------------------------------------------
# TestPhysicalWater
# ---------------------------------------------------------------------------

class TestPhysicalWater:

    @pytest.fixture(scope="class")
    def lda(self):
        return ks(_water(), functional='lda')

    @pytest.fixture(scope="class")
    def pbe(self):
        return ks(_water(), functional='pbe')

    def test_lda_total_energy(self, lda):
        assert lda.e_total == pytest.approx(-74.9280563834, rel=1e-7)

    def test_lda_xc_energy(self, lda):
        assert lda.e_xc == pytest.approx(-9.0735983009, rel=1e-5)

    def test_lda_converged(self, lda):
        assert lda.converged

    def test_lda_n_occ(self, lda):
        assert lda.n_occ == 5

    def test_lda_orbital_energies_ascending(self, lda):
        assert np.all(np.diff(lda.orbital_energies) > 0)

    def test_lda_occupied_orbitals_negative(self, lda):
        """All 5 occupied KS eigenvalues should be negative."""
        assert np.all(lda.orbital_energies[:lda.n_occ] < 0.0)

    def test_pbe_total_energy(self, pbe):
        assert pbe.e_total == pytest.approx(-75.4017601087, rel=1e-7)

    def test_pbe_xc_energy(self, pbe):
        assert pbe.e_xc == pytest.approx(-9.5439529117, rel=1e-5)

    def test_pbe_converged(self, pbe):
        assert pbe.converged

    def test_pbe_lower_than_lda(self, lda, pbe):
        assert pbe.e_total < lda.e_total

    def test_lda_fock_symmetric(self, lda):
        np.testing.assert_allclose(lda.fock, lda.fock.T, atol=1e-12)

    def test_lda_density_symmetric(self, lda):
        np.testing.assert_allclose(lda.density, lda.density.T, atol=1e-12)


# ---------------------------------------------------------------------------
# TestPhysicalHe
# ---------------------------------------------------------------------------

class TestPhysicalHe:

    @pytest.fixture(scope="class")
    def lda(self):
        return ks(_he(), functional='lda')

    @pytest.fixture(scope="class")
    def pbe(self):
        return ks(_he(), functional='pbe')

    def test_lda_total_energy(self, lda):
        assert lda.e_total == pytest.approx(-2.8095985982, rel=1e-7)

    def test_lda_xc_energy(self, lda):
        assert lda.e_xc == pytest.approx(-1.0575276230, rel=1e-6)

    def test_lda_nuclear_repulsion_zero(self, lda):
        """Single atom — no nuclear repulsion."""
        assert lda.e_nuclear == pytest.approx(0.0, abs=1e-15)

    def test_lda_homo_energy(self, lda):
        assert lda.homo_energy == pytest.approx(-0.50823066, rel=1e-5)

    def test_lda_no_lumo(self, lda):
        """He/STO-3G: 1 basis function, fully occupied — no LUMO."""
        assert lda.lumo_energy is None
        assert lda.homo_lumo_gap is None

    def test_lda_one_basis_function(self, lda):
        assert lda.orbital_energies.shape == (1,)

    def test_pbe_total_energy(self, pbe):
        assert pbe.e_total == pytest.approx(-2.8604656194, rel=1e-7)

    def test_pbe_lower_than_lda(self, lda, pbe):
        assert pbe.e_total < lda.e_total


# ---------------------------------------------------------------------------
# TestPhysicalLiH
# ---------------------------------------------------------------------------

class TestPhysicalLiH:
    """LiH exercises heteronuclear Becke partitioning corrections."""

    @pytest.fixture(scope="class")
    def lda(self):
        return ks(_lih(), functional='lda')

    @pytest.fixture(scope="class")
    def pbe(self):
        return ks(_lih(), functional='pbe')

    def test_lda_total_energy(self, lda):
        assert lda.e_total == pytest.approx(-7.8656473005, rel=1e-7)

    def test_lda_xc_energy(self, lda):
        assert lda.e_xc == pytest.approx(-2.1593657403, rel=1e-5)

    def test_lda_converged(self, lda):
        assert lda.converged

    def test_lda_n_occ(self, lda):
        """Li (3e) + H (1e) = 4 electrons → 2 occupied orbitals."""
        assert lda.n_occ == 2

    def test_lda_orbital_energies_ascending(self, lda):
        assert np.all(np.diff(lda.orbital_energies) > 0)

    def test_pbe_total_energy(self, pbe):
        assert pbe.e_total == pytest.approx(-7.9807700677, rel=1e-7)

    def test_pbe_lower_than_lda(self, lda, pbe):
        assert pbe.e_total < lda.e_total


# ---------------------------------------------------------------------------
# TestGridQuality
# ---------------------------------------------------------------------------

class TestGridQuality:
    """The molecular grid must correctly integrate the electron density."""

    def test_h2_lda_grid_electron_count(self):
        """Σ_g w_g ρ(r_g) should equal 2 for H₂."""
        r = ks(_h2(), functional='lda')
        assert r.grid_electrons == pytest.approx(2.0, abs=1e-4)

    def test_h2_pbe_grid_electron_count(self):
        r = ks(_h2(), functional='pbe')
        assert r.grid_electrons == pytest.approx(2.0, abs=1e-4)

    def test_water_lda_grid_electron_count(self):
        """Water has 10 electrons."""
        r = ks(_water(), functional='lda')
        assert r.grid_electrons == pytest.approx(10.0, abs=1e-3)

    def test_he_lda_grid_electron_count(self):
        r = ks(_he(), functional='lda')
        assert r.grid_electrons == pytest.approx(2.0, abs=1e-4)

    def test_n_grid_points_reported_correctly(self):
        """n_grid_points should equal the actual number of grid points used."""
        from qchem.dft.grid import build_molecular_grid
        mol = _h2()
        pts, wts = build_molecular_grid(mol, n_rad=75, n_theta=17)
        r = ks(mol, functional='lda', n_rad=75, n_theta=17)
        assert r.n_grid_points == len(wts)

    def test_coarser_grid_gives_similar_energy(self):
        """A coarser grid (n_rad=50) should give an energy within ~1 mHa."""
        r_fine   = ks(_h2(), functional='lda', n_rad=75, n_theta=17)
        r_coarse = ks(_h2(), functional='lda', n_rad=50, n_theta=11)
        assert abs(r_fine.e_total - r_coarse.e_total) < 1e-3

    def test_finer_grid_has_more_points(self):
        r_fine   = ks(_h2(), functional='lda', n_rad=75, n_theta=17)
        r_coarse = ks(_h2(), functional='lda', n_rad=50, n_theta=11)
        assert r_fine.n_grid_points > r_coarse.n_grid_points


# ---------------------------------------------------------------------------
# TestConvergenceBehaviour
# ---------------------------------------------------------------------------

class TestConvergenceBehaviour:

    def test_diis_and_no_diis_agree_lda_h2(self):
        """DIIS and plain SCF must give the same energy to tight tolerance."""
        r_diis   = ks(_h2(), functional='lda', use_diis=True)
        r_nodiis = ks(_h2(), functional='lda', use_diis=False)
        assert r_diis.e_total == pytest.approx(r_nodiis.e_total, rel=1e-8)

    def test_diis_and_no_diis_agree_pbe_h2(self):
        r_diis   = ks(_h2(), functional='pbe', use_diis=True)
        r_nodiis = ks(_h2(), functional='pbe', use_diis=False)
        assert r_diis.e_total == pytest.approx(r_nodiis.e_total, rel=1e-8)

    def test_water_requires_diis_to_converge(self):
        """
        Plain Roothaan iteration oscillates for water LDA without DIIS.
        The XC grid potential introduces nonlinearity that the unaccelerated
        iteration cannot damp.  This verifies that DIIS is not merely a
        speed-up for KS-DFT — it is required for robust convergence.
        """
        # Without DIIS, water should fail (oscillates, density change ~0.5)
        with pytest.raises(RuntimeError, match="[Cc]onverg"):
            ks(_water(), functional='lda', use_diis=False, max_iter=30)
        # With DIIS, the same system converges cleanly
        r = ks(_water(), functional='lda', use_diis=True)
        assert r.converged

    def test_diis_converges_water_plain_does_not(self):
        """
        Positive control: DIIS converges water to a specific energy; the
        plain (no-DIIS) run fails with RuntimeError before reaching it.
        """
        r_diis = ks(_water(), functional='lda', use_diis=True)
        assert r_diis.e_total == pytest.approx(-74.9280563834, rel=1e-7)
        with pytest.raises(RuntimeError):
            ks(_water(), functional='lda', use_diis=False, max_iter=20)

    def test_n_iter_within_max_iter(self):
        r = ks(_h2(), functional='lda', max_iter=50)
        assert 1 <= r.n_iter <= 50

    def test_diis_max_vec_2_still_converges(self):
        """A very small DIIS window should still reach convergence."""
        r = ks(_water(), functional='lda', use_diis=True, diis_max_vec=2)
        assert r.converged

    def test_diis_start_delay_same_energy(self):
        r_early = ks(_water(), functional='lda', diis_start=2)
        r_late  = ks(_water(), functional='lda', diis_start=6)
        assert r_early.e_total == pytest.approx(r_late.e_total, rel=1e-7)

    def test_h2_plain_scf_converges_lda(self):
        r = ks(_h2(), functional='lda', use_diis=False)
        assert r.converged

    def test_h2_plain_scf_converges_pbe(self):
        r = ks(_h2(), functional='pbe', use_diis=False)
        assert r.converged


# ---------------------------------------------------------------------------
# TestSCFProperties
# ---------------------------------------------------------------------------

class TestSCFProperties:
    """Mathematical properties that must hold exactly at convergence."""

    @pytest.fixture(scope="class")
    def h2_lda(self):
        return ks(_h2(), functional='lda')

    @pytest.fixture(scope="class")
    def water_lda(self):
        return ks(_water(), functional='lda')

    @pytest.fixture(scope="class")
    def h2_pbe(self):
        return ks(_h2(), functional='pbe')

    # --- Fock/density commutator -----------------------------------------

    def test_h2_lda_commutator_near_zero(self, h2_lda):
        """At convergence FPS − SPF ≈ 0 for KS just as for HF."""
        F, P, S = h2_lda.fock, h2_lda.density, h2_lda.overlap
        comm = F @ P @ S - S @ P @ F
        np.testing.assert_allclose(comm, 0.0, atol=1e-9)

    def test_water_lda_commutator_near_zero(self, water_lda):
        F, P, S = water_lda.fock, water_lda.density, water_lda.overlap
        comm = F @ P @ S - S @ P @ F
        np.testing.assert_allclose(comm, 0.0, atol=1e-8)

    def test_h2_pbe_commutator_near_zero(self, h2_pbe):
        F, P, S = h2_pbe.fock, h2_pbe.density, h2_pbe.overlap
        comm = F @ P @ S - S @ P @ F
        np.testing.assert_allclose(comm, 0.0, atol=1e-9)

    # --- Generalised idempotency -----------------------------------------

    def test_h2_lda_generalised_idempotency(self, h2_lda):
        """PSP = 2P for a closed-shell KS density matrix (same as RHF)."""
        P, S = h2_lda.density, h2_lda.overlap
        np.testing.assert_allclose(P @ S @ P, 2.0 * P, atol=1e-9)

    def test_water_lda_generalised_idempotency(self, water_lda):
        P, S = water_lda.density, water_lda.overlap
        np.testing.assert_allclose(P @ S @ P, 2.0 * P, atol=1e-8)

    # --- Self-consistency: re-diagonalising the converged F gives back P -

    def test_h2_lda_self_consistent(self, h2_lda):
        """Diagonalising the converged F_KS must reproduce the converged P."""
        from qchem.scf.density import density_matrix
        from qchem.linalg import solve_generalized_eigenvalue
        F, S = h2_lda.fock, h2_lda.overlap
        _, C = solve_generalized_eigenvalue(F, S)
        P_check = density_matrix(C, n_occ=h2_lda.n_occ)
        np.testing.assert_allclose(P_check, h2_lda.density, atol=1e-9)

    def test_water_lda_self_consistent(self, water_lda):
        from qchem.scf.density import density_matrix
        from qchem.linalg import solve_generalized_eigenvalue
        F, S = water_lda.fock, water_lda.overlap
        _, C = solve_generalized_eigenvalue(F, S)
        P_check = density_matrix(C, n_occ=water_lda.n_occ)
        np.testing.assert_allclose(P_check, water_lda.density, atol=1e-8)

    # --- Fock and density symmetry ---------------------------------------

    def test_h2_lda_fock_symmetric(self, h2_lda):
        np.testing.assert_allclose(h2_lda.fock, h2_lda.fock.T, atol=1e-12)

    def test_h2_lda_density_symmetric(self, h2_lda):
        np.testing.assert_allclose(h2_lda.density, h2_lda.density.T, atol=1e-12)

    def test_water_lda_fock_symmetric(self, water_lda):
        np.testing.assert_allclose(water_lda.fock, water_lda.fock.T, atol=1e-12)

    def test_h2_pbe_fock_symmetric(self, h2_pbe):
        np.testing.assert_allclose(h2_pbe.fock, h2_pbe.fock.T, atol=1e-12)

    # --- Orbital energy ordering -----------------------------------------

    def test_h2_lda_orbital_energies_ascending(self, h2_lda):
        assert np.all(np.diff(h2_lda.orbital_energies) > 0)

    def test_water_lda_orbital_energies_ascending(self, water_lda):
        assert np.all(np.diff(water_lda.orbital_energies) > 0)

    # --- HOMO below LUMO -------------------------------------------------

    def test_h2_lda_homo_below_lumo(self, h2_lda):
        assert h2_lda.homo_energy < h2_lda.lumo_energy

    def test_water_lda_homo_below_lumo(self, water_lda):
        assert water_lda.homo_energy < water_lda.lumo_energy

    # --- Electron count from density -------------------------------------

    def test_water_lda_electron_count_from_density(self, water_lda):
        from qchem.scf.density import n_electrons_from_density
        n_e = n_electrons_from_density(water_lda.density, water_lda.overlap)
        assert n_e == pytest.approx(10.0, rel=1e-8)

    # --- h_core stored correctly (density-independent) -------------------

    def test_h_core_matches_fresh_computation(self, h2_lda):
        """The stored H_core must match a freshly computed one."""
        from qchem.integrals.kinetic import build_kinetic_matrix
        from qchem.integrals.nuclear import build_nuclear_matrix
        from qchem.scf.fock import core_hamiltonian
        mol   = _h2()
        basis = mol.build_basis('sto-3g')
        T = build_kinetic_matrix(basis)
        V = build_nuclear_matrix(basis, mol.atomic_numbers, mol.coords)
        H_fresh = core_hamiltonian(T, V)
        np.testing.assert_allclose(h2_lda.h_core, H_fresh, atol=1e-14)


# ---------------------------------------------------------------------------
# TestInvariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Properties that must hold for any correct KS calculation."""

    def test_energy_additivity_all_molecules(self):
        """E_total = E_electronic + E_nuclear to machine precision."""
        for mol in [_h2(), _water(), _he(), _lih()]:
            r = ks(mol, functional='lda')
            assert r.e_total == pytest.approx(
                r.e_electronic + r.e_nuclear, rel=1e-12
            ), f"Additivity failed for {mol.symbols}"

    def test_reproducibility_lda(self):
        """Two identical ks() calls must produce bit-identical results."""
        r1 = ks(_water(), functional='lda')
        r2 = ks(_water(), functional='lda')
        assert r1.e_total == r2.e_total
        np.testing.assert_array_equal(r1.orbital_energies, r2.orbital_energies)

    def test_reproducibility_pbe(self):
        r1 = ks(_h2(), functional='pbe')
        r2 = ks(_h2(), functional='pbe')
        assert r1.e_total == r2.e_total

    def test_lda_and_rhf_give_different_energies(self):
        """
        LDA-DFT and RHF use fundamentally different physics and must give
        different total energies.  LDA total energies are not variational
        bounds to the exact energy, so they can be either above or below
        the HF value depending on the molecule and basis set.
        """
        from qchem.scf.hartree_fock import rhf
        for mol in [_h2(), _water()]:
            ks_r = ks(mol, functional='lda')
            hf_r = rhf(mol)
            assert abs(ks_r.e_total - hf_r.e_total) > 1e-4, (
                f"LDA and HF unexpectedly agree for {mol.symbols}: "
                f"LDA={ks_r.e_total:.8f}, HF={hf_r.e_total:.8f}"
            )

    def test_atom_ordering_invariance(self):
        """Swapping the two H atoms in H₂ must not change the energy."""
        r1 = ks(Molecule([('H',[0.,0.,-0.7]),('H',[0.,0., 0.7])]), functional='lda')
        r2 = ks(Molecule([('H',[0.,0., 0.7]),('H',[0.,0.,-0.7])]), functional='lda')
        assert r1.e_total == pytest.approx(r2.e_total, rel=1e-10)

    def test_electronic_energy_negative(self):
        """E_electronic must be negative for all test molecules."""
        for mol in [_h2(), _water(), _he(), _lih()]:
            r = ks(mol, functional='lda')
            assert r.e_electronic < 0.0, (
                f"Positive electronic energy for {mol.symbols}: {r.e_electronic}"
            )

    def test_e_xc_negative_all_molecules(self):
        """Exchange-correlation energy is always negative for physical densities."""
        for mol in [_h2(), _water(), _he()]:
            for func in ['lda', 'pbe']:
                r = ks(mol, functional=func)
                assert r.e_xc < 0.0, (
                    f"Positive E_xc for {mol.symbols}/{func}: {r.e_xc}"
                )

    @pytest.mark.parametrize("mol", [_h2(), _water(), _he(), _lih()])
    def test_density_positive_semidefinite(self, mol):
        """P = 2 C_occ C_occ^T is always PSD regardless of functional."""
        r = ks(mol, functional='lda')
        eigvals = np.linalg.eigvalsh(r.density)
        assert np.all(eigvals >= -1e-10), (
            f"Negative density eigenvalue for {mol.symbols}: {eigvals.min():.3e}"
        )

    @pytest.mark.parametrize("mol", [_h2(), _water(), _he(), _lih()])
    def test_overlap_unchanged_by_scf(self, mol):
        """The stored overlap must match a freshly computed one (no mutation)."""
        from qchem.integrals.overlap import build_overlap_matrix
        r = ks(mol, functional='lda')
        S_fresh = build_overlap_matrix(mol.build_basis('sto-3g'))
        np.testing.assert_allclose(r.overlap, S_fresh, atol=1e-14)

    def test_build_coulomb_matrix_matches_einsum(self):
        """build_coulomb_matrix must equal the reference einsum for J."""
        from qchem.integrals.eri import build_eri_tensor
        mol   = _h2()
        basis = mol.build_basis('sto-3g')
        ERI   = build_eri_tensor(basis)
        from qchem.scf.hartree_fock import rhf
        P     = rhf(mol).density
        J     = build_coulomb_matrix(P, ERI)
        J_ref = np.einsum('ls,mnls->mn', P, ERI)
        np.testing.assert_allclose(J, J_ref, atol=1e-14)

    def test_ks_fock_no_exchange(self):
        """
        The KS Fock matrix F_KS = H + J + V_xc must differ from the HF
        Fock matrix F_HF = H + J − ½K by exactly −½K + V_xc.
        Concretely: for a closed-shell system with a non-trivial density,
        F_KS ≠ F_HF.
        """
        from qchem.scf.hartree_fock import rhf
        from qchem.scf.fock import fock_matrix
        from qchem.integrals.eri import build_eri_tensor
        from qchem.dft.grid import build_molecular_grid, eval_ao_on_grid, eval_density_on_grid, build_xc_matrix
        from qchem.dft.xc import get_xc

        mol   = _h2()
        basis = mol.build_basis('sto-3g')
        ERI   = build_eri_tensor(basis)
        P     = rhf(mol).density

        J = build_coulomb_matrix(P, ERI)

        pts, wts = build_molecular_grid(mol, n_rad=75, n_theta=17)
        ao  = eval_ao_on_grid(basis, pts)
        rho = eval_density_on_grid(P, ao)
        xc  = get_xc('lda', rho)
        V_xc = build_xc_matrix(ao, xc.v_xc_rho, wts)

        from qchem.scf.fock import core_hamiltonian
        from qchem.integrals.kinetic import build_kinetic_matrix
        from qchem.integrals.nuclear import build_nuclear_matrix
        T = build_kinetic_matrix(basis)
        V = build_nuclear_matrix(basis, mol.atomic_numbers, mol.coords)
        H = core_hamiltonian(T, V)

        F_ks = ks_fock_matrix(H, J, V_xc)
        F_hf = fock_matrix(H, P, ERI)

        # They must be different — KS has no K, HF has no V_xc
        assert not np.allclose(F_ks, F_hf, atol=1e-6), (
            "KS and HF Fock matrices are unexpectedly identical"
        )

    def test_ks_electronic_energy_not_half_trace_trick(self):
        """
        Demonstrate that the KS energy is NOT ½ Tr[P(H+F)].
        The correct KS formula gives a different (correct) result.
        """
        from qchem.integrals.eri import build_eri_tensor
        from qchem.integrals.kinetic import build_kinetic_matrix
        from qchem.integrals.nuclear import build_nuclear_matrix
        from qchem.scf.fock import core_hamiltonian
        from qchem.dft.grid import build_molecular_grid, eval_ao_on_grid, eval_density_on_grid, build_xc_matrix
        from qchem.dft.xc import get_xc

        mol   = _h2()
        r_ks  = ks(mol, functional='lda')

        P   = r_ks.density
        H   = r_ks.h_core
        F   = r_ks.fock

        # Wrong formula (HF shortcut) — should not match the KS energy
        e_wrong = 0.5 * float(np.einsum('ij,ij->', P, H + F))
        # Correct formula (stored in result)
        e_right = r_ks.e_electronic

        assert abs(e_wrong - e_right) > 1e-6, (
            f"KS and HF energy formulas unexpectedly agree: "
            f"e_wrong={e_wrong:.8f}, e_right={e_right:.8f}"
        )
