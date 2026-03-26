"""
Tests for qchem/molecule.py

Structure
---------
TestConstruction        — atoms stored correctly, bohr vs angstrom, symbol normalisation
TestProperties          — n_atoms, symbols, coords, atomic_numbers, is_closed_shell
TestElectronCounting    — n_electrons / n_alpha / n_beta for neutral, charged, open-shell
TestValidation          — every ValueError and KeyError the constructor can raise
TestNuclearRepulsion    — V_nn values and edge cases
TestBuildBasis          — AO counts and shell-dict format
TestFromXyz             — parser paths, Å conversion, charge/mult forwarding, error cases
TestInvariants          — relationships that must hold for any valid molecule
"""

import numpy as np
import pytest

from qchem.molecule import Molecule
from qchem.basis import BOHR_PER_ANGSTROM


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _h2(bond_bohr: float = 1.4) -> Molecule:
    r = bond_bohr / 2
    return Molecule([('H', [0., 0., -r]), ('H', [0., 0., r])])


def _water_bohr() -> Molecule:
    """Water at the geometry used across the test suite (bohr)."""
    return Molecule([
        ('O', [0.000,  0.000,  0.000]),
        ('H', [0.000,  1.430,  1.107]),
        ('H', [0.000, -1.430,  1.107]),
    ])


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_atoms_length(self):
        mol = _h2()
        assert len(mol.atoms) == 2

    def test_symbol_normalised_to_uppercase(self):
        mol = Molecule([('h', [0., 0., 0.]), ('h', [0., 0., 1.4])])
        assert mol.atoms[0][0] == 'H'
        assert mol.atoms[1][0] == 'H'

    def test_symbol_mixed_case(self):
        mol = Molecule([('he', [0., 0., 0.])])
        assert mol.atoms[0][0] == 'He'

    def test_coords_stored_as_numpy_array(self):
        mol = _h2()
        for _, r in mol.atoms:
            assert isinstance(r, np.ndarray)
            assert r.shape == (3,)
            assert r.dtype == float

    def test_coords_stored_in_bohr_by_default(self):
        r_bohr = [0., 0., 0.7]
        mol = Molecule([('H', [0., 0., 0.]), ('H', r_bohr)])
        np.testing.assert_allclose(mol.atoms[1][1], r_bohr)

    def test_angstrom_conversion_applied(self):
        r_ang = 0.74                  # H–H bond in Å
        mol = Molecule(
            [('H', [0., 0., 0.]), ('H', [0., 0., r_ang])],
            angstrom=True,
        )
        expected_bohr = r_ang * BOHR_PER_ANGSTROM
        np.testing.assert_allclose(mol.atoms[1][1][2], expected_bohr, rtol=1e-10)

    def test_angstrom_false_does_not_scale(self):
        r = [0., 0., 1.4]
        mol_bohr = Molecule([('H', [0., 0., 0.]), ('H', r)], angstrom=False)
        np.testing.assert_allclose(mol_bohr.atoms[1][1], r)

    def test_charge_stored(self):
        mol = Molecule([('H', [0., 0., 0.])], charge=1)
        assert mol.charge == 1

    def test_multiplicity_stored(self):
        mol = Molecule([('H', [0., 0., 0.])], multiplicity=2)
        assert mol.multiplicity == 2

    def test_list_coords_accepted(self):
        mol = Molecule([('H', [0., 0., 0.]), ('H', [0., 0., 1.4])])
        assert mol.n_atoms == 2

    def test_tuple_coords_accepted(self):
        mol = Molecule([('H', (0., 0., 0.)), ('H', (0., 0., 1.4))])
        assert mol.n_atoms == 2

    def test_atom_order_preserved(self):
        mol = _water_bohr()
        assert mol.symbols == ['O', 'H', 'H']


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_n_atoms(self):
        assert _h2().n_atoms == 2
        assert _water_bohr().n_atoms == 3

    def test_symbols(self):
        assert _h2().symbols == ['H', 'H']
        assert _water_bohr().symbols == ['O', 'H', 'H']

    def test_coords_shape(self):
        mol = _water_bohr()
        assert mol.coords.shape == (3, 3)

    def test_coords_values(self):
        mol = _h2(bond_bohr=1.4)
        expected = np.array([[0., 0., -0.7], [0., 0., 0.7]])
        np.testing.assert_allclose(mol.coords, expected)

    def test_atomic_numbers_h2(self):
        assert _h2().atomic_numbers == [1, 1]

    def test_atomic_numbers_water(self):
        assert _water_bohr().atomic_numbers == [8, 1, 1]

    def test_is_closed_shell_true_for_singlet(self):
        assert _h2().is_closed_shell is True

    def test_is_closed_shell_false_for_doublet(self):
        # OH radical: 9 electrons, doublet
        mol = Molecule(
            [('O', [0., 0., 0.]), ('H', [0., 0., 1.8])],
            multiplicity=2,
        )
        assert mol.is_closed_shell is False

    def test_is_closed_shell_false_for_triplet(self):
        # O atom triplet (ground state)
        mol = Molecule([('O', [0., 0., 0.])], multiplicity=3)
        assert mol.is_closed_shell is False


# ---------------------------------------------------------------------------
# TestElectronCounting
# ---------------------------------------------------------------------------

class TestElectronCounting:
    # Neutral closed-shell
    def test_h2_electron_count(self):
        assert _h2().n_electrons == 2

    def test_water_electron_count(self):
        assert _water_bohr().n_electrons == 10

    def test_closed_shell_alpha_beta_equal(self):
        mol = _h2()
        assert mol.n_alpha == mol.n_beta == 1

    def test_water_closed_shell_counts(self):
        mol = _water_bohr()
        assert mol.n_alpha == 5
        assert mol.n_beta == 5

    # Charged molecules
    def test_cation_reduces_electron_count(self):
        mol = Molecule([('H', [0., 0., 0.]), ('H', [0., 0., 1.4])], charge=1, multiplicity=2)
        assert mol.n_electrons == 1    # H2+ doublet

    def test_anion_increases_electron_count(self):
        mol = Molecule([('H', [0., 0., 0.]), ('H', [0., 0., 1.4])], charge=-1, multiplicity=2)
        assert mol.n_electrons == 3    # H2- doublet

    def test_large_negative_charge(self):
        # O2- : 8 + 2 = 10 - (-2) ... no, charge=-2 means 8+2=10 electrons
        mol = Molecule([('O', [0., 0., 0.])], charge=-2, multiplicity=1)
        assert mol.n_electrons == 10

    def test_positive_charge_n_alpha_n_beta_sum(self):
        mol = Molecule([('H', [0., 0., 0.]), ('H', [0., 0., 1.4])], charge=1, multiplicity=2)
        assert mol.n_alpha + mol.n_beta == mol.n_electrons

    # Open-shell
    def test_doublet_alpha_exceeds_beta_by_one(self):
        mol = Molecule(
            [('O', [0., 0., 0.]), ('H', [0., 0., 1.8])],
            multiplicity=2,
        )
        assert mol.n_alpha - mol.n_beta == 1

    def test_doublet_electron_counts(self):
        # OH: 9 electrons, doublet → n_alpha=5, n_beta=4
        mol = Molecule(
            [('O', [0., 0., 0.]), ('H', [0., 0., 1.8])],
            multiplicity=2,
        )
        assert mol.n_alpha == 5
        assert mol.n_beta == 4

    def test_triplet_alpha_exceeds_beta_by_two(self):
        # O atom: 8 electrons, triplet
        mol = Molecule([('O', [0., 0., 0.])], multiplicity=3)
        assert mol.n_alpha - mol.n_beta == 2

    def test_triplet_electron_counts(self):
        mol = Molecule([('O', [0., 0., 0.])], multiplicity=3)
        assert mol.n_alpha == 5
        assert mol.n_beta == 3

    def test_h_atom_doublet(self):
        mol = Molecule([('H', [0., 0., 0.])], multiplicity=2)
        assert mol.n_electrons == 1
        assert mol.n_alpha == 1
        assert mol.n_beta == 0

    def test_high_spin_nitrogen_quartet(self):
        # N atom: 7 electrons, quartet (S=3/2, 2S+1=4)
        mol = Molecule([('N', [0., 0., 0.])], multiplicity=4)
        assert mol.n_alpha == 5
        assert mol.n_beta == 2


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_multiplicity_zero_raises(self):
        with pytest.raises(ValueError, match="multiplicity"):
            Molecule([('H', [0., 0., 0.])], multiplicity=0)

    def test_multiplicity_negative_raises(self):
        with pytest.raises(ValueError, match="multiplicity"):
            Molecule([('H', [0., 0., 0.])], multiplicity=-1)

    def test_negative_electron_count_raises(self):
        # H has 1 electron; charge=2 → -1 electrons
        with pytest.raises(ValueError, match="electrons"):
            Molecule([('H', [0., 0., 0.])], charge=2)

    def test_parity_mismatch_even_electrons_doublet(self):
        # H2: 2 electrons (even), doublet requires 1 unpaired (odd) → mismatch
        with pytest.raises(ValueError, match="parity"):
            Molecule([('H', [0., 0., 0.]), ('H', [0., 0., 1.4])], multiplicity=2)

    def test_parity_mismatch_odd_electrons_singlet(self):
        # OH: 9 electrons (odd), singlet requires 0 unpaired (even) → mismatch
        with pytest.raises(ValueError, match="parity"):
            Molecule(
                [('O', [0., 0., 0.]), ('H', [0., 0., 1.8])],
                multiplicity=1,
            )

    def test_more_unpaired_than_electrons_raises(self):
        # H has 1 electron, triplet requires 2 unpaired
        with pytest.raises(ValueError):
            Molecule([('H', [0., 0., 0.])], multiplicity=3)

    def test_unknown_element_symbol_raises(self):
        with pytest.raises(KeyError):
            Molecule([('Xx', [0., 0., 0.])])

    def test_wrong_coord_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Molecule([('H', [0., 0.])])   # 2D instead of 3D

    def test_wrong_coord_shape_4d_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Molecule([('H', [0., 0., 0., 0.])])


# ---------------------------------------------------------------------------
# TestNuclearRepulsion
# ---------------------------------------------------------------------------

class TestNuclearRepulsion:
    def test_single_atom_is_zero(self):
        mol = Molecule([('He', [0., 0., 0.])])
        assert mol.nuclear_repulsion() == pytest.approx(0.0)

    def test_h2_at_1_4_bohr(self):
        # V_nn = Z_H * Z_H / r = 1/1.4 = 5/7
        mol = _h2(bond_bohr=1.4)
        assert mol.nuclear_repulsion() == pytest.approx(5 / 7, rel=1e-12)

    def test_h2_scales_inversely_with_distance(self):
        # Doubling distance should halve V_nn
        v1 = _h2(bond_bohr=1.4).nuclear_repulsion()
        v2 = _h2(bond_bohr=2.8).nuclear_repulsion()
        assert v1 == pytest.approx(2 * v2, rel=1e-10)

    def test_water_known_value(self):
        # Computed analytically from test geometry
        mol = _water_bohr()
        assert mol.nuclear_repulsion() == pytest.approx(9.1971984402, rel=1e-8)

    def test_symmetric_molecule_pair_contributions_equal(self):
        # In H2, both atom orderings give the same V_nn
        mol_fwd = Molecule([('H', [0., 0., -0.7]), ('H', [0., 0., 0.7])])
        mol_rev = Molecule([('H', [0., 0., 0.7]), ('H', [0., 0., -0.7])])
        assert mol_fwd.nuclear_repulsion() == pytest.approx(
            mol_rev.nuclear_repulsion(), rel=1e-12
        )

    def test_heteronuclear_scales_with_charges(self):
        # LiH: Z_Li=3, Z_H=1 → V_nn = 3/r
        # HH:  Z_H=1,  Z_H=1 → V_nn = 1/r
        # same distance, so LiH should be 3× larger
        r = 3.0
        mol_lih = Molecule([('Li', [0., 0., 0.]), ('H', [0., 0., r])])
        mol_hh  = Molecule([('H',  [0., 0., 0.]), ('H', [0., 0., r])])
        assert mol_lih.nuclear_repulsion() == pytest.approx(
            3 * mol_hh.nuclear_repulsion(), rel=1e-12
        )

    def test_return_type_is_float(self):
        assert isinstance(_h2().nuclear_repulsion(), float)


# ---------------------------------------------------------------------------
# TestBuildBasis
# ---------------------------------------------------------------------------

class TestBuildBasis:
    # STO-3G AO counts: H=1, He=1, Li=2, Be=2, B=5, C=5, N=5, O=5, F=5
    def test_h2_ao_count(self):
        assert len(_h2().build_basis('sto-3g')) == 2

    def test_water_ao_count(self):
        # O: 1s + 2sp (→ 1s + 3p) = 5; each H: 1s → 7 total
        assert len(_water_bohr().build_basis('sto-3g')) == 7

    def test_returns_list(self):
        assert isinstance(_h2().build_basis(), list)

    def test_each_shell_has_required_keys(self):
        basis = _h2().build_basis()
        required = {'center', 'angular', 'exponents', 'coefficients'}
        for shell in basis:
            assert required <= shell.keys(), f"Missing keys in shell: {shell.keys()}"

    def test_shell_center_matches_molecule_coords(self):
        mol = _h2()
        basis = mol.build_basis()
        # H2 STO-3G has one s function per H → 2 shells total
        np.testing.assert_allclose(basis[0]['center'], mol.atoms[0][1])
        np.testing.assert_allclose(basis[1]['center'], mol.atoms[1][1])

    def test_shell_exponents_are_positive(self):
        for shell in _water_bohr().build_basis():
            assert all(e > 0 for e in shell['exponents'])

    def test_unknown_basis_raises(self):
        with pytest.raises(KeyError):
            _h2().build_basis('not-a-basis')


# ---------------------------------------------------------------------------
# TestFromXyz
# ---------------------------------------------------------------------------

class TestFromXyz:
    _WATER_XYZ_FULL = """\
3
water
O  0.000  0.000  0.000
H  0.000  0.757 -0.471
H  0.000 -0.757 -0.471
"""
    _WATER_XYZ_NO_HEADER = """\
O  0.000  0.000  0.000
H  0.000  0.757 -0.471
H  0.000 -0.757 -0.471
"""

    def test_full_xyz_parses(self):
        mol = Molecule.from_xyz(self._WATER_XYZ_FULL)
        assert mol.n_atoms == 3

    def test_no_header_xyz_parses(self):
        mol = Molecule.from_xyz(self._WATER_XYZ_NO_HEADER)
        assert mol.n_atoms == 3

    def test_both_parsers_give_same_coords(self):
        m1 = Molecule.from_xyz(self._WATER_XYZ_FULL)
        m2 = Molecule.from_xyz(self._WATER_XYZ_NO_HEADER)
        np.testing.assert_allclose(m1.coords, m2.coords)

    def test_angstrom_conversion_applied(self):
        mol = Molecule.from_xyz(self._WATER_XYZ_FULL)
        # O is at origin in bohr too
        np.testing.assert_allclose(mol.atoms[0][1], [0., 0., 0.])
        # H should NOT be at raw Å values
        h_bohr = mol.atoms[1][1]
        h_ang  = np.array([0., 0.757, -0.471])
        assert not np.allclose(h_bohr, h_ang), \
            "Coordinates look like they were NOT converted from Å to bohr"
        np.testing.assert_allclose(h_bohr, h_ang * BOHR_PER_ANGSTROM, rtol=1e-6)

    def test_electron_count_correct(self):
        mol = Molecule.from_xyz(self._WATER_XYZ_FULL)
        assert mol.n_electrons == 10

    def test_symbol_order_preserved(self):
        mol = Molecule.from_xyz(self._WATER_XYZ_FULL)
        assert mol.symbols == ['O', 'H', 'H']

    def test_charge_forwarded(self):
        # H2O+ has 9 electrons (odd) → must be a doublet
        mol = Molecule.from_xyz(self._WATER_XYZ_FULL, charge=1, multiplicity=2)
        assert mol.charge == 1
        assert mol.n_electrons == 9

    def test_multiplicity_forwarded(self):
        # Water cation is a doublet (9 electrons)
        mol = Molecule.from_xyz(self._WATER_XYZ_FULL, charge=1, multiplicity=2)
        assert mol.multiplicity == 2

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            Molecule.from_xyz("")

    def test_blank_lines_only_raises(self):
        with pytest.raises(ValueError):
            Molecule.from_xyz("   \n\n   ")

    def test_wrong_column_count_raises(self):
        with pytest.raises(ValueError):
            Molecule.from_xyz("H 0.0 0.0")   # only 3 tokens

    def test_non_numeric_coords_raise(self):
        with pytest.raises(ValueError):
            Molecule.from_xyz("H x y z")

    def test_leading_trailing_whitespace_tolerated(self):
        xyz = "\n\n  3\n  water\n  O  0 0 0\n  H  0 0.757 -0.471\n  H  0 -0.757 -0.471\n\n"
        mol = Molecule.from_xyz(xyz)
        assert mol.n_atoms == 3

    def test_h2_from_xyz(self):
        xyz = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7414"
        mol = Molecule.from_xyz(xyz)
        assert mol.n_electrons == 2
        expected_dist = 0.7414 * BOHR_PER_ANGSTROM
        actual_dist = np.linalg.norm(mol.coords[1] - mol.coords[0])
        assert actual_dist == pytest.approx(expected_dist, rel=1e-8)


# ---------------------------------------------------------------------------
# TestInvariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Properties that must hold for *any* valid Molecule."""

    @pytest.mark.parametrize("mol", [
        _h2(),
        _water_bohr(),
        Molecule([('He', [0., 0., 0.])]),
        Molecule([('N', [0., 0., 0.])], multiplicity=4),
        Molecule([('O', [0., 0., 0.]), ('H', [0., 0., 1.8])], multiplicity=2),
        Molecule([('H', [0., 0., 0.]), ('H', [0., 0., 1.4])], charge=1, multiplicity=2),
    ])
    def test_alpha_plus_beta_equals_n_electrons(self, mol):
        assert mol.n_alpha + mol.n_beta == mol.n_electrons

    @pytest.mark.parametrize("mol", [
        _h2(),
        _water_bohr(),
        Molecule([('He', [0., 0., 0.])]),
        Molecule([('N', [0., 0., 0.])], multiplicity=4),
        Molecule([('O', [0., 0., 0.]), ('H', [0., 0., 1.8])], multiplicity=2),
    ])
    def test_alpha_geq_beta(self, mol):
        assert mol.n_alpha >= mol.n_beta

    @pytest.mark.parametrize("mol", [
        _h2(),
        _water_bohr(),
        Molecule([('He', [0., 0., 0.])]),
    ])
    def test_coords_shape_consistent_with_n_atoms(self, mol):
        assert mol.coords.shape == (mol.n_atoms, 3)

    @pytest.mark.parametrize("mol", [
        _h2(),
        _water_bohr(),
        Molecule([('He', [0., 0., 0.])]),
    ])
    def test_symbols_length_equals_n_atoms(self, mol):
        assert len(mol.symbols) == mol.n_atoms

    @pytest.mark.parametrize("mol", [
        _h2(),
        _water_bohr(),
        Molecule([('He', [0., 0., 0.])]),
    ])
    def test_atomic_numbers_length_equals_n_atoms(self, mol):
        assert len(mol.atomic_numbers) == mol.n_atoms

    @pytest.mark.parametrize("mol", [
        _h2(),
        _water_bohr(),
    ])
    def test_nuclear_repulsion_is_non_negative(self, mol):
        assert mol.nuclear_repulsion() >= 0.0
