"""
Tests for qchem/scf/fock.py

Structure
---------
TestCoreHamiltonian     — shape, symmetry, values, validation
TestFockMatrix          — structure, index correctness, zero-P case, validation
TestElectronicEnergy    — formula correctness, sign, units
TestPhysicalH2          — end-to-end checks against Szabo & Ostlund H2/STO-3G reference
TestPhysicalWater       — end-to-end checks for water/STO-3G
TestInvariants          — properties that must hold for any valid inputs
"""

import numpy as np
import pytest

from qchem.scf.fock import core_hamiltonian, fock_matrix, electronic_energy


# ---------------------------------------------------------------------------
# Shared fixtures and reference data
# ---------------------------------------------------------------------------

def _h2_ingredients():
    """
    Full set of H2/STO-3G matrices at R=1.4 bohr.

    Reference values cross-checked against Szabo & Ostlund Table 3.1 and
    a converged RHF calculation.
    """
    from qchem.molecule import Molecule
    from qchem.integrals.overlap import build_overlap_matrix
    from qchem.integrals.kinetic import build_kinetic_matrix
    from qchem.integrals.nuclear import build_nuclear_matrix
    from qchem.integrals.eri import build_eri_tensor

    mol = Molecule([('H', [0., 0., -0.7]), ('H', [0., 0., 0.7])])
    basis = mol.build_basis('sto-3g')
    S   = build_overlap_matrix(basis)
    T   = build_kinetic_matrix(basis)
    V   = build_nuclear_matrix(basis, mol.atomic_numbers, mol.coords)
    ERI = build_eri_tensor(basis)
    return mol, S, T, V, ERI


def _water_ingredients():
    """Water/STO-3G matrices at the standard test geometry (bohr)."""
    from qchem.molecule import Molecule
    from qchem.integrals.kinetic import build_kinetic_matrix
    from qchem.integrals.nuclear import build_nuclear_matrix
    from qchem.integrals.eri import build_eri_tensor

    mol = Molecule([
        ('O', [0.000,  0.000,  0.000]),
        ('H', [0.000,  1.430,  1.107]),
        ('H', [0.000, -1.430,  1.107]),
    ])
    basis = mol.build_basis('sto-3g')
    T   = build_kinetic_matrix(basis)
    V   = build_nuclear_matrix(basis, mol.atomic_numbers, mol.coords)
    ERI = build_eri_tensor(basis)
    return mol, T, V, ERI


def _converge_scf(H_core, S, ERI, n_occ, max_iter=100, tol=1e-10):
    """
    Minimal SCF loop using fock_matrix and density_matrix.
    Returns the converged P, F, and electronic energy.
    """
    from qchem.scf.density import density_matrix
    from qchem.linalg import solve_generalized_eigenvalue

    n = H_core.shape[0]
    P = np.zeros((n, n))
    for _ in range(max_iter):
        F = fock_matrix(H_core, P, ERI)
        _, C = solve_generalized_eigenvalue(F, S)
        P_new = density_matrix(C, n_occ)
        if np.max(np.abs(P_new - P)) < tol:
            return P_new, F
        P = P_new
    raise RuntimeError("SCF did not converge")


# ---------------------------------------------------------------------------
# TestCoreHamiltonian
# ---------------------------------------------------------------------------

class TestCoreHamiltonian:

    def test_returns_sum_of_T_and_V(self):
        T = np.array([[1., 2.], [2., 3.]])
        V = np.array([[0.5, 0.1], [0.1, 0.5]])
        H = core_hamiltonian(T, V)
        np.testing.assert_allclose(H, T + V)

    def test_output_shape(self):
        n = 5
        T = np.eye(n)
        V = np.zeros((n, n))
        assert core_hamiltonian(T, V).shape == (n, n)

    def test_symmetric_for_symmetric_inputs(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((4, 4))
        T = A + A.T
        B = rng.standard_normal((4, 4))
        V = B + B.T
        H = core_hamiltonian(T, V)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_does_not_modify_inputs(self):
        T = np.eye(2)
        V = np.eye(2)
        T_copy, V_copy = T.copy(), V.copy()
        core_hamiltonian(T, V)
        np.testing.assert_array_equal(T, T_copy)
        np.testing.assert_array_equal(V, V_copy)

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="shape"):
            core_hamiltonian(np.eye(2), np.eye(3))

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError):
            core_hamiltonian(np.ones(4), np.ones(4))

    def test_h2_hcore_diagonal(self):
        _, _, T, V, _ = _h2_ingredients()
        H = core_hamiltonian(T, V)
        # T11 ≈ 0.7600, V11 ≈ -1.8804 → H11 ≈ -1.1204
        assert H[0, 0] == pytest.approx(-1.12040887, rel=1e-6)
        assert H[1, 1] == pytest.approx(-1.12040887, rel=1e-6)

    def test_h2_hcore_offdiagonal(self):
        _, _, T, V, _ = _h2_ingredients()
        H = core_hamiltonian(T, V)
        assert H[0, 1] == pytest.approx(-0.95837985, rel=1e-6)


# ---------------------------------------------------------------------------
# TestFockMatrix
# ---------------------------------------------------------------------------

class TestFockMatrix:

    def test_zero_density_gives_hcore(self):
        """With P=0, G=0 and F must equal H_core exactly."""
        _, _, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P = np.zeros_like(H)
        F = fock_matrix(H, P, ERI)
        np.testing.assert_allclose(F, H, atol=1e-14)

    def test_output_shape(self):
        _, _, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P = np.zeros((2, 2))
        assert fock_matrix(H, P, ERI).shape == (2, 2)

    def test_f_is_symmetric(self):
        """F must be symmetric for any symmetric P."""
        _, _, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        rng = np.random.default_rng(7)
        A = rng.standard_normal((2, 2))
        P = A + A.T   # random symmetric density
        F = fock_matrix(H, P, ERI)
        np.testing.assert_allclose(F, F.T, atol=1e-12)

    def test_g_is_symmetric(self):
        """G = F - H_core must be symmetric."""
        _, _, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        rng = np.random.default_rng(11)
        A = rng.standard_normal((2, 2))
        P = A + A.T
        F = fock_matrix(H, P, ERI)
        G = F - H
        np.testing.assert_allclose(G, G.T, atol=1e-12)

    def test_does_not_modify_inputs(self):
        _, _, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P = np.zeros((2, 2))
        H_copy, P_copy, ERI_copy = H.copy(), P.copy(), ERI.copy()
        fock_matrix(H, P, ERI)
        np.testing.assert_array_equal(H, H_copy)
        np.testing.assert_array_equal(P, P_copy)
        np.testing.assert_array_equal(ERI, ERI_copy)

    def test_linear_in_P(self):
        """G(aP) = a * G(P) since G is linear in P."""
        _, _, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        rng = np.random.default_rng(3)
        A = rng.standard_normal((2, 2))
        P = A + A.T
        a = 2.5
        F1 = fock_matrix(H, P, ERI)
        F2 = fock_matrix(H, a * P, ERI)
        # G(aP) = a*G(P)  ⟹  F(aP) - H = a*(F(P) - H)
        G1 = F1 - H
        G2 = F2 - H
        np.testing.assert_allclose(G2, a * G1, rtol=1e-12)

    def test_p_shape_mismatch_raises(self):
        _, _, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        with pytest.raises(ValueError, match="shape|inconsistent"):
            fock_matrix(H, np.zeros((3, 3)), ERI)

    def test_eri_shape_mismatch_raises(self):
        _, _, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P = np.zeros((2, 2))
        bad_ERI = np.zeros((3, 3, 3, 3))
        with pytest.raises(ValueError, match="shape|inconsistent"):
            fock_matrix(H, P, bad_ERI)

    def test_h2_converged_fock_diagonal(self):
        """Diagonal of converged F for H2/STO-3G matches reference."""
        mol, S, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=1)
        # Reference: F = [[-0.36553729, -0.59388531], [...]]
        assert F[0, 0] == pytest.approx(-0.36553729, rel=1e-6)
        assert F[1, 1] == pytest.approx(-0.36553729, rel=1e-6)

    def test_h2_converged_fock_offdiagonal(self):
        mol, S, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=1)
        assert F[0, 1] == pytest.approx(-0.59388531, rel=1e-6)

    def test_coulomb_and_exchange_index_order(self):
        """
        Explicitly verify J and K using the ERI symmetry of H2.

        For H2/STO-3G with the converged P, the Coulomb and exchange
        contributions can be computed via the known ERI values:
          ERI[0,0,0,0] = ERI[1,1,1,1] ≈ 0.77461
          ERI[0,0,1,1] = ERI[1,1,0,0] ≈ 0.56968
          ERI[0,1,0,1] = ERI[0,1,1,0] ≈ 0.29703
        This test catches any transposition of the J/K einsum indices.
        """
        mol, S, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=1)

        # Recompute J and K explicitly from converged P and ERI
        J = np.einsum('ls,mnls->mn', P, ERI)
        K = np.einsum('ls,mlns->mn', P, ERI)
        G_expected = J - 0.5 * K
        F_expected = H + G_expected

        np.testing.assert_allclose(F, F_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# TestElectronicEnergy
# ---------------------------------------------------------------------------

class TestElectronicEnergy:

    def test_zero_density_gives_zero(self):
        n = 3
        P = np.zeros((n, n))
        H = np.eye(n)
        F = np.eye(n)
        assert electronic_energy(P, H, F) == pytest.approx(0.0)

    def test_factor_of_half(self):
        """E = ½ Tr[P(H+F)]: with P=I, H=I, F=I → E = ½ * n * 2 = n."""
        n = 3
        P = np.eye(n)
        H = np.eye(n)
        F = np.eye(n)
        assert electronic_energy(P, H, F) == pytest.approx(float(n))

    def test_returns_float(self):
        P = np.eye(2)
        H = np.eye(2)
        F = np.eye(2)
        result = electronic_energy(P, H, F)
        assert isinstance(result, float)

    def test_formula_matches_trace(self):
        """E = ½ Tr[P(H+F)] matches np.trace explicitly (symmetric inputs)."""
        rng = np.random.default_rng(42)
        # H_core and F are always symmetric in RHF — the formula only
        # equals np.trace(P @ M) when M is symmetric.
        A = rng.standard_normal((4, 4));  P = A + A.T
        B = rng.standard_normal((4, 4));  H = B + B.T
        C = rng.standard_normal((4, 4));  F = C + C.T
        expected = 0.5 * np.trace(P @ (H + F))
        assert electronic_energy(P, H, F) == pytest.approx(expected, rel=1e-12)

    def test_h2_electronic_energy(self):
        """E_elec for converged H2/STO-3G must match reference to 6 decimal places."""
        mol, S, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=1)
        E_elec = electronic_energy(P, H, F)
        # Reference from converged SCF: -1.8310000333 Ha
        assert E_elec == pytest.approx(-1.8310000333, rel=1e-8)

    def test_h2_total_energy(self):
        """E_total = E_elec + V_nn for H2/STO-3G must match -1.1167143190 Ha."""
        mol, S, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=1)
        E_total = electronic_energy(P, H, F) + mol.nuclear_repulsion()
        # Szabo & Ostlund Table 3.3: E = -1.1175 Ha (their geometry differs
        # slightly; with R=1.4 bohr exactly our reference is -1.1167143190)
        assert E_total == pytest.approx(-1.1167143190, rel=1e-8)


# ---------------------------------------------------------------------------
# TestPhysicalH2
# ---------------------------------------------------------------------------

class TestPhysicalH2:

    def test_hcore_is_symmetric(self):
        _, _, T, V, _ = _h2_ingredients()
        H = core_hamiltonian(T, V)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_fock_is_symmetric_at_convergence(self):
        mol, S, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=1)
        np.testing.assert_allclose(F, F.T, atol=1e-12)

    def test_converged_density_is_self_consistent(self):
        """
        At convergence, the Fock matrix built from P must reproduce
        the same P after diagonalisation (to within SCF tolerance).
        """
        from qchem.scf.density import density_matrix
        from qchem.linalg import solve_generalized_eigenvalue

        mol, S, T, V, ERI = _h2_ingredients()
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=1)

        _, C = solve_generalized_eigenvalue(F, S)
        P_check = density_matrix(C, n_occ=1)
        np.testing.assert_allclose(P_check, P, atol=1e-9)


# ---------------------------------------------------------------------------
# TestPhysicalWater
# ---------------------------------------------------------------------------

class TestPhysicalWater:

    def test_hcore_shape(self):
        _, T, V, _ = _water_ingredients()
        H = core_hamiltonian(T, V)
        assert H.shape == (7, 7)

    def test_hcore_is_symmetric(self):
        _, T, V, _ = _water_ingredients()
        H = core_hamiltonian(T, V)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_fock_is_symmetric_at_convergence(self):
        from qchem.integrals.overlap import build_overlap_matrix
        from qchem.molecule import Molecule

        mol, T, V, ERI = _water_ingredients()
        mol_obj = Molecule([
            ('O', [0.000,  0.000,  0.000]),
            ('H', [0.000,  1.430,  1.107]),
            ('H', [0.000, -1.430,  1.107]),
        ])
        S = build_overlap_matrix(mol_obj.build_basis('sto-3g'))
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=5)
        np.testing.assert_allclose(F, F.T, atol=1e-12)

    def test_water_total_energy(self):
        """E_total for water/STO-3G must match converged reference -74.9629 Ha."""
        from qchem.integrals.overlap import build_overlap_matrix
        from qchem.molecule import Molecule

        mol_obj = Molecule([
            ('O', [0.000,  0.000,  0.000]),
            ('H', [0.000,  1.430,  1.107]),
            ('H', [0.000, -1.430,  1.107]),
        ])
        _, T, V, ERI = _water_ingredients()
        S = build_overlap_matrix(mol_obj.build_basis('sto-3g'))
        H = core_hamiltonian(T, V)
        P, F = _converge_scf(H, S, ERI, n_occ=5)
        E_total = electronic_energy(P, H, F) + mol_obj.nuclear_repulsion()
        assert E_total == pytest.approx(-74.96289753, rel=1e-6)


# ---------------------------------------------------------------------------
# TestInvariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Properties that must hold for any valid (H_core, P, ERI) combination."""

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_fock_symmetric_for_symmetric_inputs(self, n):
        rng = np.random.default_rng(n * 17)
        A = rng.standard_normal((n, n));  H = A + A.T
        B = rng.standard_normal((n, n));  P = B + B.T
        # Build a trivially symmetric ERI: ERI[i,j,k,l] = (i+j+k+l+1)^{-1}
        ERI = np.zeros((n, n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        ERI[i,j,k,l] = 1.0 / (i + j + k + l + 1)
        F = fock_matrix(H, P, ERI)
        np.testing.assert_allclose(F, F.T, atol=1e-12)

    @pytest.mark.parametrize("n,n_occ", [(2, 1), (4, 2)])
    def test_electronic_energy_non_positive_for_attractive_hcore(self, n, n_occ):
        """
        For a purely attractive (negative diagonal) H_core and small G,
        E_elec should be negative.
        """
        from qchem.scf.density import density_matrix
        from qchem.linalg import solve_generalized_eigenvalue

        H = -1.0 * np.eye(n)       # attractive one-electron Hamiltonian
        ERI = np.zeros((n, n, n, n))
        S = np.eye(n)
        P = np.zeros((n, n))
        F = fock_matrix(H, P, ERI)   # G=0, so F=H
        _, C = solve_generalized_eigenvalue(F, S)
        P = density_matrix(C, n_occ)
        F = fock_matrix(H, P, ERI)
        E = electronic_energy(P, H, F)
        assert E < 0.0
