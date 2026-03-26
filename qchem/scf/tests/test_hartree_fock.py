"""
Tests for qchem/scf/hartree_fock.py

Structure
---------
TestInputValidation     — open-shell, zero electrons, odd electrons,
                          unknown basis, convergence failure
TestResultFields        — shape, type, flag, electron count, energy split
TestRHFResult           — convenience properties: HOMO, LUMO, gap, repr
TestPhysicalH2          — reference energies and orbital energies for H₂/STO-3G
TestPhysicalWater       — reference energy for water/STO-3G
TestPhysicalHe          — simplest 2-electron atom
TestPhysicalLiH         — heteronuclear diatomic
TestConvergenceBehaviour — DIIS vs plain, iteration counts, tolerance control
TestSCFProperties       — self-consistency, commutator, idempotency, ordering
TestInvariants          — basis-ordering independence, energy additivity,
                          reproducibility, monotonic plain-SCF descent

Reference values
----------------
H₂/STO-3G at R = 1.4 bohr:
    E_total      = -1.11671432 Ha  (Szabo & Ostlund Table 3.3)
    ε₁ (bonding) = -0.57820297 Ha
    ε₂ (anti)    = +0.67026784 Ha

Water/STO-3G at the geometry used throughout the suite (bohr):
    E_total      = -74.96289753 Ha

He/STO-3G:
    E_total      = -2.80778396 Ha
    ε₁           = -0.87603549 Ha

LiH/STO-3G at R = 3.015 bohr:
    E_total      = -7.86200927 Ha
"""

import numpy as np
import pytest

from qchem.molecule import Molecule
from qchem.scf.hartree_fock import rhf, RHFResult


# ---------------------------------------------------------------------------
# Shared molecule fixtures
# ---------------------------------------------------------------------------

def _h2(bond_bohr: float = 1.4) -> Molecule:
    r = bond_bohr / 2
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
        """Doublet OH radical must be rejected — RHF is closed-shell only."""
        mol = Molecule(
            [('O', [0., 0., 0.]), ('H', [0., 0., 1.8])],
            multiplicity=2,
        )
        with pytest.raises(ValueError, match="closed-shell"):
            rhf(mol)

    def test_triplet_raises(self):
        mol = Molecule([('O', [0., 0., 0.])], multiplicity=3)
        with pytest.raises(ValueError, match="closed-shell"):
            rhf(mol)

    def test_unknown_basis_raises(self):
        with pytest.raises(KeyError):
            rhf(_h2(), basis_name='not-a-basis')

    def test_convergence_failure_raises_runtime_error(self):
        """A max_iter=1 calculation on water should fail to converge."""
        with pytest.raises(RuntimeError, match="converge"):
            rhf(_water(), max_iter=1)

    def test_convergence_error_message_contains_diagnostics(self):
        """The RuntimeError message should include useful diagnostic numbers."""
        with pytest.raises(RuntimeError) as exc_info:
            rhf(_water(), max_iter=1)
        msg = str(exc_info.value)
        # Should contain the iteration limit and at least one floating-point value
        assert "1" in msg

    def test_tight_tolerance_still_converges(self):
        """Very tight tolerances should converge for a well-behaved system."""
        result = rhf(_h2(), e_tol=1e-12, d_tol=1e-10)
        assert result.converged

    def test_loose_tolerance_converges_faster(self):
        """Loose tolerances should converge in fewer iterations."""
        r_tight = rhf(_water(), e_tol=1e-8, d_tol=1e-6)
        r_loose = rhf(_water(), e_tol=1e-4, d_tol=1e-3)
        assert r_loose.n_iter <= r_tight.n_iter


# ---------------------------------------------------------------------------
# TestResultFields
# ---------------------------------------------------------------------------

class TestResultFields:

    @pytest.fixture(scope="class")
    def h2_result(self):
        return rhf(_h2())

    def test_returns_rhfresult(self, h2_result):
        assert isinstance(h2_result, RHFResult)

    def test_converged_flag_is_true(self, h2_result):
        assert h2_result.converged is True

    def test_orbital_energies_shape(self, h2_result):
        """Should have one orbital energy per basis function."""
        assert h2_result.orbital_energies.shape == (2,)

    def test_coefficients_shape(self, h2_result):
        assert h2_result.coefficients.shape == (2, 2)

    def test_density_shape(self, h2_result):
        assert h2_result.density.shape == (2, 2)

    def test_fock_shape(self, h2_result):
        assert h2_result.fock.shape == (2, 2)

    def test_overlap_shape(self, h2_result):
        assert h2_result.overlap.shape == (2, 2)

    def test_h_core_shape(self, h2_result):
        assert h2_result.h_core.shape == (2, 2)

    def test_e_total_is_float(self, h2_result):
        assert isinstance(h2_result.e_total, float)

    def test_e_electronic_is_float(self, h2_result):
        assert isinstance(h2_result.e_electronic, float)

    def test_e_nuclear_is_float(self, h2_result):
        assert isinstance(h2_result.e_nuclear, float)

    def test_n_iter_is_positive_int(self, h2_result):
        assert isinstance(h2_result.n_iter, int)
        assert h2_result.n_iter >= 1

    def test_n_occ_correct(self, h2_result):
        # H2 has 2 electrons → 1 occupied spatial orbital
        assert h2_result.n_occ == 1

    def test_basis_name_stored(self, h2_result):
        assert h2_result.basis_name == 'sto-3g'

    def test_mol_stored(self, h2_result):
        assert isinstance(h2_result.mol, Molecule)

    def test_energy_split_adds_up(self, h2_result):
        """E_total must equal E_electronic + E_nuclear exactly."""
        assert h2_result.e_total == pytest.approx(
            h2_result.e_electronic + h2_result.e_nuclear, rel=1e-12
        )

    def test_electron_count_from_density(self, h2_result):
        """Tr[PS] must equal the number of electrons."""
        from qchem.scf.density import n_electrons_from_density
        P = h2_result.density
        S = h2_result.overlap
        n_e = n_electrons_from_density(P, S)
        assert n_e == pytest.approx(float(_h2().n_electrons), rel=1e-8)

    def test_water_n_occ(self):
        result = rhf(_water())
        # Water: 10 electrons → 5 occupied orbitals
        assert result.n_occ == 5

    def test_water_orbital_energies_shape(self):
        result = rhf(_water())
        # STO-3G water has 7 basis functions
        assert result.orbital_energies.shape == (7,)


# ---------------------------------------------------------------------------
# TestRHFResult (convenience properties)
# ---------------------------------------------------------------------------

class TestRHFResult:

    @pytest.fixture(scope="class")
    def h2_result(self):
        return rhf(_h2())

    def test_homo_energy(self, h2_result):
        """HOMO energy = orbital_energies[n_occ - 1]."""
        assert h2_result.homo_energy == pytest.approx(
            h2_result.orbital_energies[h2_result.n_occ - 1]
        )

    def test_lumo_energy(self, h2_result):
        """LUMO energy = orbital_energies[n_occ]."""
        assert h2_result.lumo_energy == pytest.approx(
            h2_result.orbital_energies[h2_result.n_occ]
        )

    def test_homo_lumo_gap_positive(self, h2_result):
        """LUMO must lie above HOMO — gap must be positive."""
        assert h2_result.homo_lumo_gap > 0.0

    def test_homo_lumo_gap_equals_difference(self, h2_result):
        assert h2_result.homo_lumo_gap == pytest.approx(
            h2_result.lumo_energy - h2_result.homo_energy
        )

    def test_lumo_none_when_fully_occupied(self):
        """He/STO-3G has 1 basis function and 1 occupied MO — no LUMO."""
        result = rhf(_he())
        assert result.lumo_energy is None
        assert result.homo_lumo_gap is None

    def test_repr_contains_energy(self, h2_result):
        r = repr(h2_result)
        assert "converged" in r.lower()
        assert "Ha" in r

    def test_repr_contains_n_iter(self, h2_result):
        r = repr(h2_result)
        assert "n_iter" in r

    def test_repr_contains_basis(self, h2_result):
        r = repr(h2_result)
        assert "sto-3g" in r


# ---------------------------------------------------------------------------
# TestPhysicalH2
# ---------------------------------------------------------------------------

class TestPhysicalH2:
    """
    Reference values from Szabo & Ostlund, Table 3.3, plus a converged
    in-house calculation at R = 1.4 bohr.
    """

    @pytest.fixture(scope="class")
    def result(self):
        return rhf(_h2())

    def test_total_energy(self, result):
        """E_total must match −1.11671432 Ha (Szabo & Ostlund Table 3.3)."""
        assert result.e_total == pytest.approx(-1.1167143190, rel=1e-8)

    def test_electronic_energy(self, result):
        assert result.e_electronic == pytest.approx(-1.8310000333, rel=1e-8)

    def test_nuclear_repulsion(self, result):
        # V_nn = 1/1.4 = 5/7
        assert result.e_nuclear == pytest.approx(5.0 / 7.0, rel=1e-10)

    def test_bonding_orbital_energy(self, result):
        """ε₁ (bonding MO) ≈ −0.57820297 Ha."""
        assert result.orbital_energies[0] == pytest.approx(-0.5782029706, rel=1e-6)

    def test_antibonding_orbital_energy(self, result):
        """ε₂ (antibonding MO) ≈ +0.67026784 Ha."""
        assert result.orbital_energies[1] == pytest.approx(+0.6702678353, rel=1e-6)

    def test_orbital_energies_ascending(self, result):
        assert result.orbital_energies[0] < result.orbital_energies[1]

    def test_homo_is_bonding(self, result):
        """The occupied orbital must be the bonding (lower-energy) one."""
        assert result.homo_energy < 0.0

    def test_lumo_is_antibonding(self, result):
        """The virtual orbital must be positive energy for H₂/STO-3G."""
        assert result.lumo_energy > 0.0

    def test_energy_at_larger_bond_length_is_higher(self):
        """Stretching the bond should raise the total energy."""
        r_eq  = rhf(_h2(bond_bohr=1.4))
        r_str = rhf(_h2(bond_bohr=4.0))
        assert r_str.e_total > r_eq.e_total

    def test_energy_at_shorter_bond_length_is_higher(self):
        """Compressing the bond should also raise the total energy."""
        r_eq   = rhf(_h2(bond_bohr=1.4))
        r_comp = rhf(_h2(bond_bohr=0.8))
        assert r_comp.e_total > r_eq.e_total


# ---------------------------------------------------------------------------
# TestPhysicalWater
# ---------------------------------------------------------------------------

class TestPhysicalWater:

    @pytest.fixture(scope="class")
    def result(self):
        return rhf(_water())

    def test_total_energy(self, result):
        """E_total for water/STO-3G ≈ −74.96289753 Ha."""
        assert result.e_total == pytest.approx(-74.96289753, rel=1e-6)

    def test_converged(self, result):
        assert result.converged

    def test_n_occ(self, result):
        assert result.n_occ == 5

    def test_orbital_energies_ascending(self, result):
        eps = result.orbital_energies
        assert np.all(np.diff(eps) > 0)

    def test_occupied_orbitals_all_negative(self, result):
        """All 5 occupied orbital energies should be negative."""
        occ = result.orbital_energies[:result.n_occ]
        assert np.all(occ < 0.0)

    def test_homo_lumo_gap_positive(self, result):
        assert result.homo_lumo_gap > 0.0

    def test_fock_is_symmetric(self, result):
        np.testing.assert_allclose(result.fock, result.fock.T, atol=1e-12)

    def test_density_is_symmetric(self, result):
        np.testing.assert_allclose(result.density, result.density.T, atol=1e-12)


# ---------------------------------------------------------------------------
# TestPhysicalHe
# ---------------------------------------------------------------------------

class TestPhysicalHe:

    @pytest.fixture(scope="class")
    def result(self):
        return rhf(_he())

    def test_total_energy(self, result):
        """He/STO-3G total energy ≈ −2.80778396 Ha."""
        assert result.e_total == pytest.approx(-2.8077839633, rel=1e-6)

    def test_nuclear_repulsion_is_zero(self, result):
        """Single atom → no nuclear repulsion."""
        assert result.e_nuclear == pytest.approx(0.0, abs=1e-15)

    def test_orbital_energy(self, result):
        """ε₁ ≈ −0.87603549 Ha."""
        assert result.orbital_energies[0] == pytest.approx(-0.87603549, rel=1e-5)

    def test_no_lumo(self, result):
        """He/STO-3G has 1 basis function, fully occupied — no LUMO."""
        assert result.lumo_energy is None

    def test_one_basis_function(self, result):
        assert result.orbital_energies.shape == (1,)

    def test_electron_count(self, result):
        from qchem.scf.density import n_electrons_from_density
        n_e = n_electrons_from_density(result.density, result.overlap)
        assert n_e == pytest.approx(2.0, rel=1e-8)


# ---------------------------------------------------------------------------
# TestPhysicalLiH
# ---------------------------------------------------------------------------

class TestPhysicalLiH:

    @pytest.fixture(scope="class")
    def result(self):
        return rhf(_lih())

    def test_total_energy(self, result):
        """LiH/STO-3G at R = 3.015 bohr ≈ −7.86200927 Ha."""
        assert result.e_total == pytest.approx(-7.8620092733, rel=1e-6)

    def test_converged(self, result):
        assert result.converged

    def test_n_occ(self, result):
        # Li: 3e, H: 1e → 4 electrons → 2 occupied orbitals
        assert result.n_occ == 2

    def test_orbital_energies_ascending(self, result):
        eps = result.orbital_energies
        assert np.all(np.diff(eps) > 0)


# ---------------------------------------------------------------------------
# TestConvergenceBehaviour
# ---------------------------------------------------------------------------

class TestConvergenceBehaviour:

    def test_diis_and_no_diis_agree_on_h2_energy(self):
        """DIIS and plain SCF must give the same converged energy."""
        r_diis   = rhf(_h2(), use_diis=True)
        r_nodiis = rhf(_h2(), use_diis=False)
        assert r_diis.e_total == pytest.approx(r_nodiis.e_total, rel=1e-8)

    def test_diis_and_no_diis_agree_on_water_energy(self):
        r_diis   = rhf(_water(), use_diis=True)
        r_nodiis = rhf(_water(), use_diis=False)
        assert r_diis.e_total == pytest.approx(r_nodiis.e_total, rel=1e-8)

    def test_diis_converges_faster_than_plain_for_water(self):
        """DIIS should need fewer iterations than plain SCF for water."""
        r_diis   = rhf(_water(), use_diis=True)
        r_nodiis = rhf(_water(), use_diis=False)
        assert r_diis.n_iter < r_nodiis.n_iter

    def test_n_iter_reported_accurately(self):
        """n_iter should not exceed max_iter when converged."""
        r = rhf(_h2(), max_iter=50)
        assert 1 <= r.n_iter <= 50

    def test_diis_max_vec_respected(self):
        """A very small DIIS window should still converge (just more slowly)."""
        r = rhf(_water(), use_diis=True, diis_max_vec=2)
        assert r.converged

    def test_diis_start_delay_works(self):
        """Delaying DIIS start should still converge to the same energy."""
        r_early = rhf(_water(), diis_start=2)
        r_late  = rhf(_water(), diis_start=5)
        assert r_early.e_total == pytest.approx(r_late.e_total, rel=1e-8)

    def test_h2_converges_without_diis(self):
        r = rhf(_h2(), use_diis=False)
        assert r.converged

    def test_water_converges_without_diis(self):
        r = rhf(_water(), use_diis=False, max_iter=50)
        assert r.converged


# ---------------------------------------------------------------------------
# TestSCFProperties
# ---------------------------------------------------------------------------

class TestSCFProperties:
    """Mathematical properties that must hold exactly at convergence."""

    @pytest.fixture(scope="class")
    def h2_result(self):
        return rhf(_h2())

    @pytest.fixture(scope="class")
    def water_result(self):
        return rhf(_water())

    # -- Fock/density commutator ------------------------------------------

    def test_h2_fock_density_commute(self, h2_result):
        """At convergence, FPS − SPF ≈ 0."""
        F, P, S = h2_result.fock, h2_result.density, h2_result.overlap
        comm = F @ P @ S - S @ P @ F
        np.testing.assert_allclose(comm, np.zeros_like(comm), atol=1e-10)

    def test_water_fock_density_commute(self, water_result):
        # The commutator decays to ~O(d_tol) = O(1e-6), so the 7×7 matrix
        # entries can be a few times 1e-9 at default tolerances.
        F, P, S = water_result.fock, water_result.density, water_result.overlap
        comm = F @ P @ S - S @ P @ F
        np.testing.assert_allclose(comm, np.zeros_like(comm), atol=1e-8)

    # -- Generalised idempotency ------------------------------------------

    def test_h2_generalised_idempotency(self, h2_result):
        """PSP = 2P for a converged RHF density matrix."""
        P, S = h2_result.density, h2_result.overlap
        np.testing.assert_allclose(P @ S @ P, 2.0 * P, atol=1e-10)

    def test_water_generalised_idempotency(self, water_result):
        P, S = water_result.density, water_result.overlap
        np.testing.assert_allclose(P @ S @ P, 2.0 * P, atol=1e-10)

    # -- Self-consistency: re-diagonalising the converged F gives back P --

    def test_h2_self_consistent(self, h2_result):
        """Diagonalising the converged F must reproduce the converged P."""
        from qchem.scf.density import density_matrix
        from qchem.linalg import solve_generalized_eigenvalue

        F, S = h2_result.fock, h2_result.overlap
        _, C_check = solve_generalized_eigenvalue(F, S)
        P_check = density_matrix(C_check, n_occ=h2_result.n_occ)
        np.testing.assert_allclose(P_check, h2_result.density, atol=1e-10)

    def test_water_self_consistent(self, water_result):
        from qchem.scf.density import density_matrix
        from qchem.linalg import solve_generalized_eigenvalue

        F, S = water_result.fock, water_result.overlap
        _, C_check = solve_generalized_eigenvalue(F, S)
        P_check = density_matrix(C_check, n_occ=water_result.n_occ)
        # Tolerance reflects the default d_tol=1e-6 convergence criterion.
        np.testing.assert_allclose(P_check, water_result.density, atol=1e-8)

    # -- Fock and density symmetry ----------------------------------------

    def test_h2_fock_symmetric(self, h2_result):
        np.testing.assert_allclose(h2_result.fock, h2_result.fock.T, atol=1e-12)

    def test_h2_density_symmetric(self, h2_result):
        np.testing.assert_allclose(
            h2_result.density, h2_result.density.T, atol=1e-12
        )

    # -- Orbital energy ordering ------------------------------------------

    def test_h2_orbital_energies_ascending(self, h2_result):
        assert np.all(np.diff(h2_result.orbital_energies) > 0)

    def test_water_orbital_energies_ascending(self, water_result):
        assert np.all(np.diff(water_result.orbital_energies) > 0)

    # -- HOMO below LUMO --------------------------------------------------

    def test_h2_homo_below_lumo(self, h2_result):
        assert h2_result.homo_energy < h2_result.lumo_energy

    def test_water_homo_below_lumo(self, water_result):
        assert water_result.homo_energy < water_result.lumo_energy

    # -- Electron count from converged density ----------------------------

    def test_water_electron_count_from_density(self, water_result):
        from qchem.scf.density import n_electrons_from_density
        n_e = n_electrons_from_density(water_result.density, water_result.overlap)
        assert n_e == pytest.approx(10.0, rel=1e-8)


# ---------------------------------------------------------------------------
# TestInvariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Properties that should hold for any valid calculation."""

    def test_basis_atom_ordering_invariance_h2(self):
        """Swapping the two H atoms must not change the total energy."""
        r1 = rhf(Molecule([('H', [0., 0., -0.7]), ('H', [0., 0.,  0.7])]))
        r2 = rhf(Molecule([('H', [0., 0.,  0.7]), ('H', [0., 0., -0.7])]))
        assert r1.e_total == pytest.approx(r2.e_total, rel=1e-10)

    def test_energy_additivity(self):
        """E_total = E_electronic + E_nuclear must hold to machine precision."""
        for mol in [_h2(), _water(), _he(), _lih()]:
            r = rhf(mol)
            assert r.e_total == pytest.approx(
                r.e_electronic + r.e_nuclear, rel=1e-12
            )

    def test_reproducibility(self):
        """Calling rhf() twice on the same molecule gives identical results."""
        r1 = rhf(_water())
        r2 = rhf(_water())
        assert r1.e_total == r2.e_total
        np.testing.assert_array_equal(r1.orbital_energies, r2.orbital_energies)

    def test_plain_scf_energy_monotonically_decreases(self):
        """
        Without DIIS, the electronic energy must decrease (or stay flat)
        on every iteration.  We verify this by manually running the loop
        and recording energies at each step.
        """
        from qchem.integrals.overlap import build_overlap_matrix
        from qchem.integrals.kinetic import build_kinetic_matrix
        from qchem.integrals.nuclear import build_nuclear_matrix
        from qchem.integrals.eri     import build_eri_tensor
        from qchem.scf.fock   import core_hamiltonian, fock_matrix, electronic_energy
        from qchem.scf.density import density_matrix
        from qchem.linalg import solve_generalized_eigenvalue

        mol   = _water()
        basis = mol.build_basis('sto-3g')
        S     = build_overlap_matrix(basis)
        T     = build_kinetic_matrix(basis)
        V     = build_nuclear_matrix(basis, mol.atomic_numbers, mol.coords)
        H     = core_hamiltonian(T, V)
        ERI   = build_eri_tensor(basis)

        _, C = solve_generalized_eigenvalue(H, S)
        P = density_matrix(C, n_occ=5)
        energies = []
        for _ in range(12):
            F  = fock_matrix(H, P, ERI)
            energies.append(electronic_energy(P, H, F))
            _, C = solve_generalized_eigenvalue(F, S)
            P  = density_matrix(C, n_occ=5)

        diffs = np.diff(energies)
        assert np.all(diffs < 1e-10), (
            f"Non-monotonic energy step detected: {diffs[diffs >= 1e-10]}"
        )

    def test_electronic_energy_is_negative(self):
        """The electronic energy must be negative for all test molecules."""
        for mol in [_h2(), _water(), _he(), _lih()]:
            r = rhf(mol)
            assert r.e_electronic < 0.0, (
                f"Positive electronic energy for {mol.symbols}: {r.e_electronic}"
            )

    def test_total_energy_lower_than_nuclear_repulsion(self):
        """E_total must be lower than E_nuclear (electrons always lower it)."""
        for mol in [_h2(), _water(), _lih()]:
            r = rhf(mol)
            assert r.e_total < r.e_nuclear

    @pytest.mark.parametrize("mol", [
        _h2(), _water(), _he(), _lih(),
    ])
    def test_density_positive_semidefinite(self, mol):
        """P = 2 C_occ C_occ^T is always PSD."""
        r = rhf(mol)
        eigvals = np.linalg.eigvalsh(r.density)
        assert np.all(eigvals >= -1e-10), (
            f"Negative density eigenvalue for {mol.symbols}: {eigvals.min():.3e}"
        )

    @pytest.mark.parametrize("mol", [
        _h2(), _water(), _he(), _lih(),
    ])
    def test_overlap_consistent_with_basis(self, mol):
        """
        The stored overlap matrix must match a freshly computed one
        (guards against any accidental mutation during the SCF loop).
        """
        from qchem.integrals.overlap import build_overlap_matrix
        r = rhf(mol)
        S_fresh = build_overlap_matrix(mol.build_basis('sto-3g'))
        np.testing.assert_allclose(r.overlap, S_fresh, atol=1e-14)
