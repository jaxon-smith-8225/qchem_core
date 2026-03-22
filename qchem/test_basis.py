"""
Tests for qchem/basis.py

Structure
---------
TestElementTable        — symbol ↔ number lookups and normalisation
TestBasisInfo           — basis_info() shape sanity for STO-3G
TestBuildBasis          — build_basis() output format and known AO counts
TestShellContent        — correct exponents / coefficients in shell dicts
TestSPShellExpansion    — SP shells expand to the right s + p functions
TestUnits               — angstrom=True path applies the correct conversion
TestNwchemParser        — load_nwchem() round-trip for a hand-crafted block
TestIntegralCompatibility — shell dicts pass straight into the integral layer
"""

import numpy as np
import pytest

from qchem.basis import (
    ATOMIC_NUMBER,
    ELEMENT_SYMBOL,
    BOHR_PER_ANGSTROM,
    angstrom_to_bohr,
    element_symbol,
    build_basis,
    basis_info,
    register_basis,
    load_nwchem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h2_atoms(bond_length_bohr: float = 1.4) -> list:
    """Two hydrogen atoms along z, symmetric about origin."""
    r = bond_length_bohr / 2
    return [
        ('H', np.array([0., 0., -r])),
        ('H', np.array([0., 0.,  r])),
    ]


def _water_atoms_bohr() -> list:
    """Approximate STO-3G optimised geometry for water, in bohr."""
    return [
        ('O', np.array([0.000,  0.000,  0.000])),
        ('H', np.array([0.000,  1.430,  1.107])),
        ('H', np.array([0.000, -1.430,  1.107])),
    ]


# ---------------------------------------------------------------------------
# TestElementTable
# ---------------------------------------------------------------------------

class TestElementTable:
    def test_hydrogen(self):
        assert ATOMIC_NUMBER['H'] == 1

    def test_neon(self):
        assert ATOMIC_NUMBER['Ne'] == 10

    def test_round_trip(self):
        for sym, z in ATOMIC_NUMBER.items():
            assert ELEMENT_SYMBOL[z] == sym

    def test_element_symbol_from_int(self):
        assert element_symbol(6) == 'C'

    def test_element_symbol_from_string(self):
        assert element_symbol('c') == 'C'
        with pytest.raises(KeyError):
            element_symbol('CARBON')   # full names are not valid symbols

    def test_element_symbol_unknown_raises(self):
        with pytest.raises(KeyError):
            element_symbol('Xx')

    def test_element_symbol_unknown_z_raises(self):
        with pytest.raises(KeyError):
            element_symbol(999)


# ---------------------------------------------------------------------------
# TestBasisInfo
# ---------------------------------------------------------------------------

class TestBasisInfo:
    """
    STO-3G AO counts per element (verified against standard references):
      H, He        → 1   (one s function)
      Li, Be, B,
      C, N, O, F,
      Ne           → 5   (1s  +  2s + 2px + 2py + 2pz)
    """

    def test_hydrogen_one_function(self):
        info = basis_info('sto-3g')
        assert info['H'] == 1

    def test_helium_one_function(self):
        info = basis_info('sto-3g')
        assert info['He'] == 1

    def test_second_row_five_functions(self):
        info = basis_info('sto-3g')
        for sym in ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']:
            assert info[sym] == 5, f"Expected 5 AOs for {sym}, got {info[sym]}"

    def test_unknown_basis_raises(self):
        with pytest.raises(KeyError):
            basis_info('not-a-basis')


# ---------------------------------------------------------------------------
# TestBuildBasis
# ---------------------------------------------------------------------------

class TestBuildBasis:
    def test_h2_sto3g_two_functions(self):
        basis = build_basis(_h2_atoms())
        assert len(basis) == 2

    def test_water_sto3g_seven_functions(self):
        # O → 5 AOs (1s + 2s + 2px + 2py + 2pz), 2×H → 1 each
        basis = build_basis(_water_atoms_bohr())
        assert len(basis) == 7

    def test_returns_list_of_dicts(self):
        basis = build_basis(_h2_atoms())
        for shell in basis:
            assert isinstance(shell, dict)

    def test_required_keys_present(self):
        required = {'center', 'angular', 'exponents', 'coefficients'}
        basis = build_basis(_h2_atoms())
        for shell in basis:
            assert required <= shell.keys()

    def test_center_is_array(self):
        basis = build_basis(_h2_atoms())
        for shell in basis:
            assert isinstance(shell['center'], np.ndarray)
            assert shell['center'].shape == (3,)

    def test_angular_is_three_tuple(self):
        basis = build_basis(_water_atoms_bohr())
        for shell in basis:
            ang = shell['angular']
            assert len(ang) == 3
            assert all(isinstance(x, int) for x in ang)

    def test_exponents_and_coefficients_same_length(self):
        basis = build_basis(_water_atoms_bohr())
        for shell in basis:
            assert len(shell['exponents']) == len(shell['coefficients'])

    def test_all_exponents_positive(self):
        basis = build_basis(_water_atoms_bohr())
        for shell in basis:
            assert all(e > 0 for e in shell['exponents'])

    def test_unknown_element_raises(self):
        with pytest.raises(KeyError):
            build_basis([('Xx', np.zeros(3))])

    def test_unknown_basis_raises(self):
        with pytest.raises(KeyError):
            build_basis(_h2_atoms(), basis_name='not-a-basis')

    def test_case_insensitive_basis_name(self):
        b1 = build_basis(_h2_atoms(), 'sto-3g')
        b2 = build_basis(_h2_atoms(), 'STO-3G')
        assert len(b1) == len(b2)

    def test_default_basis_is_sto3g(self):
        b_default = build_basis(_h2_atoms())
        b_explicit = build_basis(_h2_atoms(), 'sto-3g')
        assert len(b_default) == len(b_explicit)

    def test_atom_ordering_preserved(self):
        """First shell in basis should sit at first atom's centre."""
        atoms = _h2_atoms(bond_length_bohr=1.4)
        basis = build_basis(atoms)
        np.testing.assert_allclose(basis[0]['center'], atoms[0][1])
        np.testing.assert_allclose(basis[1]['center'], atoms[1][1])


# ---------------------------------------------------------------------------
# TestShellContent
# ---------------------------------------------------------------------------

class TestShellContent:
    """Verify that the STO-3G data made it through unscathed."""

    def test_hydrogen_exponents(self):
        basis = build_basis([('H', np.zeros(3))])
        np.testing.assert_allclose(
            basis[0]['exponents'],
            [3.4252509, 0.6239137, 0.1688554],
            rtol=1e-6,
        )

    def test_hydrogen_coefficients(self):
        basis = build_basis([('H', np.zeros(3))])
        np.testing.assert_allclose(
            basis[0]['coefficients'],
            [0.1543290, 0.5353281, 0.4446345],
            rtol=1e-6,
        )

    def test_oxygen_1s_exponents(self):
        """First shell on O should be the 1s core with the large exponents."""
        basis = build_basis([('O', np.zeros(3))])
        np.testing.assert_allclose(
            basis[0]['exponents'],
            [130.7093200, 23.8088610, 6.4436083],
            rtol=1e-6,
        )

    def test_oxygen_2s_coefficients(self):
        """Second shell (2s part of SP) should carry the SP s-coefficients."""
        basis = build_basis([('O', np.zeros(3))])
        np.testing.assert_allclose(
            basis[1]['coefficients'],
            [-0.0999672, 0.3995128, 0.7001155],
            rtol=1e-6,
        )

    def test_carbon_2p_coefficients(self):
        """p-component shells should carry the SP p-coefficients."""
        basis = build_basis([('C', np.zeros(3))])
        # basis[0] = 1s, basis[1] = 2s, basis[2..4] = 2px 2py 2pz
        for i in (2, 3, 4):
            np.testing.assert_allclose(
                basis[i]['coefficients'],
                [0.1559163, 0.6076837, 0.3919574],
                rtol=1e-6,
            )

    def test_three_primitives_per_shell_sto3g(self):
        basis = build_basis([('C', np.zeros(3))])
        for shell in basis:
            assert len(shell['exponents']) == 3


# ---------------------------------------------------------------------------
# TestSPShellExpansion
# ---------------------------------------------------------------------------

class TestSPShellExpansion:
    """SP shells must expand to exactly one s + three p functions."""

    def test_sp_expands_to_four_functions(self):
        # Carbon has 1 core-s shell + 1 SP shell → 1 + 4 = 5 AOs
        basis = build_basis([('C', np.zeros(3))])
        assert len(basis) == 5

    def test_s_component_has_correct_angular(self):
        basis = build_basis([('C', np.zeros(3))])
        assert basis[1]['angular'] == (0, 0, 0)  # 2s from SP

    def test_p_components_angular_momenta(self):
        basis = build_basis([('C', np.zeros(3))])
        # Indices 2, 3, 4 are px, py, pz
        expected = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for i, ang in zip((2, 3, 4), expected):
            assert basis[i]['angular'] == ang

    def test_sp_shares_exponents(self):
        """s and p parts of an SP shell must have identical exponents."""
        basis = build_basis([('N', np.zeros(3))])
        s_exps = basis[1]['exponents']   # 2s
        for i in (2, 3, 4):             # 2px, 2py, 2pz
            assert basis[i]['exponents'] == s_exps

    def test_sp_different_coefficients(self):
        """s and p parts of an SP shell must have DIFFERENT coefficients."""
        basis = build_basis([('N', np.zeros(3))])
        assert basis[1]['coefficients'] != basis[2]['coefficients']


# ---------------------------------------------------------------------------
# TestUnits
# ---------------------------------------------------------------------------

class TestUnits:
    def test_bohr_per_angstrom_value(self):
        """Sanity check the conversion constant itself."""
        # 1 bohr = 0.529177 Å  →  1 Å ≈ 1.88973 bohr
        assert abs(BOHR_PER_ANGSTROM - 1.8897259886) < 1e-8

    def test_angstrom_conversion_function(self):
        coords = np.array([1.0, 0.0, 0.0])
        converted = angstrom_to_bohr(coords)
        np.testing.assert_allclose(converted, [BOHR_PER_ANGSTROM, 0.0, 0.0])

    def test_angstrom_flag_scales_centers(self):
        r_ang = 0.74 / 2          # half H-H bond in Å
        r_bohr = r_ang * BOHR_PER_ANGSTROM
        atoms_ang = [
            ('H', np.array([0., 0., -r_ang])),
            ('H', np.array([0., 0.,  r_ang])),
        ]
        basis = build_basis(atoms_ang, angstrom=True)
        np.testing.assert_allclose(basis[0]['center'][2], -r_bohr, rtol=1e-10)
        np.testing.assert_allclose(basis[1]['center'][2],  r_bohr, rtol=1e-10)

    def test_angstrom_flag_does_not_affect_exponents(self):
        """Exponents are dimensionless; the flag must not scale them."""
        atoms_ang = [('H', np.array([0., 0., 0.]))]
        b_ang  = build_basis(atoms_ang, angstrom=True)
        b_bohr = build_basis([('H', np.zeros(3))])
        np.testing.assert_allclose(b_ang[0]['exponents'], b_bohr[0]['exponents'])


# ---------------------------------------------------------------------------
# TestNwchemParser
# ---------------------------------------------------------------------------

_MINI_NWCHEM = """\
#----------------------------------------------------------------------
# Basis Set Exchange
# Minimal hand-crafted STO-3G block for H only (for parser testing)
#----------------------------------------------------------------------
BASIS "ao basis" PRINT
#
# Element : Hydrogen
#
H    S
      3.42525091             0.15432897
      0.62391373             0.53532814
      0.16885540             0.44463454
END
"""


class TestNwchemParser:
    def test_registers_under_given_name(self):
        load_nwchem(_MINI_NWCHEM, 'mini-test-reg')
        info = basis_info('mini-test-reg')
        assert 'H' in info

    def test_correct_number_of_functions(self):
        load_nwchem(_MINI_NWCHEM, 'mini-test-count')
        info = basis_info('mini-test-count')
        assert info['H'] == 1

    def test_exponents_parsed_correctly(self):
        load_nwchem(_MINI_NWCHEM, 'mini-test-exp')
        basis = build_basis([('H', np.zeros(3))], 'mini-test-exp')
        np.testing.assert_allclose(
            basis[0]['exponents'],
            [3.42525091, 0.62391373, 0.16885540],
            rtol=1e-7,
        )

    def test_coefficients_parsed_correctly(self):
        load_nwchem(_MINI_NWCHEM, 'mini-test-coeff')
        basis = build_basis([('H', np.zeros(3))], 'mini-test-coeff')
        np.testing.assert_allclose(
            basis[0]['coefficients'],
            [0.15432897, 0.53532814, 0.44463454],
            rtol=1e-7,
        )


# ---------------------------------------------------------------------------
# TestRegisterBasis
# ---------------------------------------------------------------------------

class TestRegisterBasis:
    def test_custom_basis_appears_in_info(self):
        register_basis('custom-s', {
            'H': [{'type': 's',
                   'exponents': [1.0],
                   'coefficients': [1.0],
                   'p_coefficients': None}],
        })
        info = basis_info('custom-s')
        assert info['H'] == 1

    def test_custom_basis_case_insensitive(self):
        register_basis('CUSTOM-UPPER', {
            'H': [{'type': 's',
                   'exponents': [1.0],
                   'coefficients': [1.0],
                   'p_coefficients': None}],
        })
        info = basis_info('custom-upper')
        assert info['H'] == 1


# ---------------------------------------------------------------------------
# TestIntegralCompatibility
# ---------------------------------------------------------------------------

class TestIntegralCompatibility:
    """
    Pass build_basis output directly to the integral builders and confirm
    we get numerically sensible matrices.  This is an end-to-end smoke
    test for the basis → integral pipeline.
    """

    def test_h2_overlap_is_symmetric(self):
        from qchem.integrals.overlap import build_overlap_matrix
        basis = build_basis(_h2_atoms())
        S = build_overlap_matrix(basis)
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_h2_overlap_diagonal_is_one(self):
        """Normalised basis functions have ⟨φ|φ⟩ = 1."""
        from qchem.integrals.overlap import build_overlap_matrix
        basis = build_basis(_h2_atoms())
        S = build_overlap_matrix(basis)
        np.testing.assert_allclose(np.diag(S), np.ones(2), atol=1e-6)

    def test_h2_overlap_known_offdiag(self):
        """
        H₂ STO-3G overlap at R = 1.4 bohr is approximately 0.6593.
        Reference: Szabo & Ostlund, Modern Quantum Chemistry, Table 3.8.
        """
        from qchem.integrals.overlap import build_overlap_matrix
        basis = build_basis(_h2_atoms(bond_length_bohr=1.4))
        S = build_overlap_matrix(basis)
        assert abs(S[0, 1] - 0.6593) < 5e-4

    def test_water_overlap_shape(self):
        from qchem.integrals.overlap import build_overlap_matrix
        basis = build_basis(_water_atoms_bohr())
        S = build_overlap_matrix(basis)
        assert S.shape == (7, 7)

    def test_water_overlap_positive_definite(self):
        from qchem.integrals.overlap import build_overlap_matrix
        basis = build_basis(_water_atoms_bohr())
        S = build_overlap_matrix(basis)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals > 0)

    def test_water_kinetic_symmetric(self):
        from qchem.integrals.kinetic import build_kinetic_matrix
        basis = build_basis(_water_atoms_bohr())
        T = build_kinetic_matrix(basis)
        np.testing.assert_allclose(T, T.T, atol=1e-12)

    def test_water_kinetic_positive_definite(self):
        """Kinetic energy matrix must be positive definite."""
        from qchem.integrals.kinetic import build_kinetic_matrix
        basis = build_basis(_water_atoms_bohr())
        T = build_kinetic_matrix(basis)
        eigvals = np.linalg.eigvalsh(T)
        assert np.all(eigvals > 0)

    def test_water_nuclear_symmetric(self):
        from qchem.integrals.nuclear import build_nuclear_matrix
        atoms = _water_atoms_bohr()
        basis = build_basis(atoms)
        charges = [8, 1, 1]
        coords  = [a[1] for a in atoms]
        V = build_nuclear_matrix(basis, charges, coords)
        np.testing.assert_allclose(V, V.T, atol=1e-12)

    def test_water_nuclear_diagonal_negative(self):
        """
        Diagonal elements V[i,i] = <φ_i|-∑_C Z_C/|r-R_C||φ_i> must be negative:
        each basis function sits inside the molecular charge cloud and sees a
        net attractive nuclear potential.  Off-diagonal elements between p
        functions on different centres can be positive (the sign of the angular
        part can flip the integral), so we only assert the diagonal here.
        """
        from qchem.integrals.nuclear import build_nuclear_matrix
        atoms = _water_atoms_bohr()
        basis = build_basis(atoms)
        charges = [8, 1, 1]
        coords  = [a[1] for a in atoms]
        V = build_nuclear_matrix(basis, charges, coords)
        assert np.all(np.diag(V) < 0)

    def test_linalg_solve_h2(self):
        """
        For a 2×2 symmetric matrix, solve_generalized_eigenvalue should
        return two eigenvalues and a 2×2 coefficient matrix.
        """
        from qchem.integrals.overlap import build_overlap_matrix
        from qchem.integrals.kinetic import build_kinetic_matrix
        from qchem.linalg import solve_generalized_eigenvalue
        basis = build_basis(_h2_atoms())
        S = build_overlap_matrix(basis)
        T = build_kinetic_matrix(basis)  # use T as a stand-in Fock matrix
        eps, C = solve_generalized_eigenvalue(T, S)
        assert eps.shape == (2,)
        assert C.shape  == (2, 2)
        assert eps[0] <= eps[1]          # ascending order