"""
Tests for qchem/scf/density.py

Structure
---------
TestDensityMatrix           — shape, symmetry, values, occupation edge cases
TestNElectronsFromDensity   — Tr[PS] correctness and edge cases
TestInvariants              — properties that must hold for any valid (P, S) pair
TestPhysical                — end-to-end checks using real H2/STO-3G integrals
"""

import numpy as np
import pytest

from qchem.scf.density import density_matrix, n_electrons_from_density


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_orthonormal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random n×n orthonormal matrix via QR decomposition."""
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    return Q


def _h2_sto3g_ingredients():
    """
    Real STO-3G S matrix and a credible C for H2.
    Returns (S, C, n_occ) where C[:,0] is bonding MO, C[:,1] antibonding.
    """
    from qchem.molecule import Molecule
    from qchem.integrals.overlap import build_overlap_matrix
    from qchem.linalg import solve_generalized_eigenvalue
    import numpy as np

    mol = Molecule([('H', [0., 0., -0.7]), ('H', [0., 0., 0.7])])
    basis = mol.build_basis('sto-3g')
    S = build_overlap_matrix(basis)

    # Build a simple Hückel-like symmetric F for the guess
    # (we just need a valid C, not the converged one)
    F = np.array([[-0.5, -0.3], [-0.3, -0.5]])
    _, C = solve_generalized_eigenvalue(F, S)
    return S, C, 1   # H2 has 1 occupied MO in RHF


# ---------------------------------------------------------------------------
# TestDensityMatrix
# ---------------------------------------------------------------------------

class TestDensityMatrix:

    def test_output_shape_square(self):
        C = np.eye(4)
        P = density_matrix(C, n_occ=2)
        assert P.shape == (4, 4)

    def test_output_shape_matches_n_basis(self):
        n_basis = 6
        C = np.random.default_rng(0).standard_normal((n_basis, n_basis))
        P = density_matrix(C, n_occ=3)
        assert P.shape == (n_basis, n_basis)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        C = rng.standard_normal((5, 5))
        P = density_matrix(C, n_occ=3)
        np.testing.assert_allclose(P, P.T, atol=1e-14)

    def test_factor_of_two(self):
        # With one orbital and identity C, P[0,0] must be exactly 2
        C = np.eye(2)
        P = density_matrix(C, n_occ=1)
        assert P[0, 0] == pytest.approx(2.0)
        assert P[1, 1] == pytest.approx(0.0)

    def test_zero_occupied_orbitals(self):
        # n_occ=0 → zero matrix (e.g. bare nuclear framework)
        C = np.eye(3)
        P = density_matrix(C, n_occ=0)
        np.testing.assert_array_equal(P, np.zeros((3, 3)))

    def test_all_orbitals_occupied(self):
        # n_occ == n_basis, orthonormal C → P = 2 * I
        C = np.eye(3)
        P = density_matrix(C, n_occ=3)
        np.testing.assert_allclose(P, 2.0 * np.eye(3), atol=1e-14)

    def test_n_occ_uses_only_first_columns(self):
        # Only the first n_occ columns of C should matter.
        # Changing columns beyond n_occ must not affect P.
        rng = np.random.default_rng(7)
        C = rng.standard_normal((4, 4))
        C2 = C.copy()
        C2[:, 2:] = rng.standard_normal((4, 2))   # scramble virtual columns
        P1 = density_matrix(C, n_occ=2)
        P2 = density_matrix(C2, n_occ=2)
        np.testing.assert_allclose(P1, P2, atol=1e-14)

    def test_does_not_modify_C(self):
        C = np.eye(3)
        C_copy = C.copy()
        density_matrix(C, n_occ=2)
        np.testing.assert_array_equal(C, C_copy)

    def test_n_occ_negative_raises(self):
        with pytest.raises(ValueError, match="n_occ"):
            density_matrix(np.eye(3), n_occ=-1)

    def test_n_occ_exceeds_n_mo_raises(self):
        with pytest.raises(ValueError, match="n_occ"):
            density_matrix(np.eye(3), n_occ=4)

    def test_returns_ndarray(self):
        P = density_matrix(np.eye(2), n_occ=1)
        assert isinstance(P, np.ndarray)

    def test_dtype_is_float(self):
        P = density_matrix(np.eye(2), n_occ=1)
        assert np.issubdtype(P.dtype, np.floating)

    def test_rectangular_C_n_basis_gt_n_mo(self):
        # More basis functions than MOs is valid (after canonical orthogonalisation)
        C = np.zeros((6, 4))
        C[0, 0] = 1.0
        C[1, 1] = 1.0
        P = density_matrix(C, n_occ=2)
        assert P.shape == (6, 6)


# ---------------------------------------------------------------------------
# TestNElectronsFromDensity
# ---------------------------------------------------------------------------

class TestNElectronsFromDensity:

    def test_identity_overlap_one_occupied(self):
        C = np.eye(2)
        P = density_matrix(C, n_occ=1)
        assert n_electrons_from_density(P, np.eye(2)) == pytest.approx(2.0)

    def test_identity_overlap_two_occupied(self):
        C = np.eye(4)
        P = density_matrix(C, n_occ=2)
        assert n_electrons_from_density(P, np.eye(4)) == pytest.approx(4.0)

    def test_zero_density_gives_zero(self):
        P = np.zeros((3, 3))
        S = np.eye(3)
        assert n_electrons_from_density(P, S) == pytest.approx(0.0)

    def test_returns_float(self):
        result = n_electrons_from_density(np.eye(2), np.eye(2))
        assert isinstance(result, float)

    def test_non_identity_overlap(self):
        # Tr[PS] = Tr[2 C_occ C_occ^T S].
        # With orthonormal C_occ and S = 2*I: result should be 2 * n_occ * 2
        C = np.eye(3)
        P = density_matrix(C, n_occ=2)   # Tr[P] = 4
        S = 2.0 * np.eye(3)
        # Tr[PS] = Tr[P * 2I] = 2 * Tr[P] = 8
        assert n_electrons_from_density(P, S) == pytest.approx(8.0)

    def test_symmetric_in_practice(self):
        # Tr[PS] == Tr[SP] since trace is cyclic
        rng = np.random.default_rng(99)
        C = _random_orthonormal(4, rng)
        P = density_matrix(C, n_occ=2)
        S = rng.standard_normal((4, 4))
        S = S @ S.T + np.eye(4)   # make SPD
        val_ps = n_electrons_from_density(P, S)
        val_sp = float(np.einsum('ij,ji->', S, P))
        assert val_ps == pytest.approx(val_sp, rel=1e-12)


# ---------------------------------------------------------------------------
# TestInvariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Properties that must hold for any valid density matrix."""

    @pytest.mark.parametrize("n_basis,n_occ", [
        (2, 1), (4, 2), (7, 3), (5, 0), (3, 3),
    ])
    def test_p_is_symmetric(self, n_basis, n_occ):
        rng = np.random.default_rng(n_basis * 100 + n_occ)
        C = rng.standard_normal((n_basis, n_basis))
        P = density_matrix(C, n_occ=n_occ)
        np.testing.assert_allclose(P, P.T, atol=1e-13)

    @pytest.mark.parametrize("n_basis,n_occ", [
        (2, 1), (4, 2), (6, 3),
    ])
    def test_trace_equals_2_n_occ_for_orthonormal_C(self, n_basis, n_occ):
        # Tr[P] = 2 * Tr[C_occ C_occ^T] = 2 * n_occ when C is orthonormal
        rng = np.random.default_rng(n_basis + n_occ)
        C = _random_orthonormal(n_basis, rng)
        P = density_matrix(C, n_occ=n_occ)
        assert np.trace(P) == pytest.approx(2.0 * n_occ, rel=1e-12)

    @pytest.mark.parametrize("n_basis,n_occ", [
        (2, 1), (4, 2), (6, 3),
    ])
    def test_tr_ps_equals_2_n_occ_for_orthonormal_C_and_identity_S(
        self, n_basis, n_occ
    ):
        rng = np.random.default_rng(n_basis * 13 + n_occ)
        C = _random_orthonormal(n_basis, rng)
        P = density_matrix(C, n_occ=n_occ)
        n_e = n_electrons_from_density(P, np.eye(n_basis))
        assert n_e == pytest.approx(2.0 * n_occ, rel=1e-12)

    @pytest.mark.parametrize("n_basis,n_occ", [
        (2, 1), (4, 2), (5, 3),
    ])
    def test_p_is_positive_semidefinite(self, n_basis, n_occ):
        # P = 2 C_occ C_occ^T is always PSD
        rng = np.random.default_rng(n_basis * 7 + n_occ)
        C = rng.standard_normal((n_basis, n_basis))
        P = density_matrix(C, n_occ=n_occ)
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues >= -1e-12), f"Negative eigenvalue: {eigenvalues.min()}"

    @pytest.mark.parametrize("n_basis,n_occ", [
        (2, 1), (4, 2), (6, 3),
    ])
    def test_rank_equals_n_occ(self, n_basis, n_occ):
        # P = 2 C_occ C_occ^T has rank exactly n_occ (for linearly independent C_occ)
        rng = np.random.default_rng(n_basis * 31 + n_occ)
        C = _random_orthonormal(n_basis, rng)
        P = density_matrix(C, n_occ=n_occ)
        rank = np.linalg.matrix_rank(P, tol=1e-10)
        assert rank == n_occ


# ---------------------------------------------------------------------------
# TestPhysical
# ---------------------------------------------------------------------------

class TestPhysical:
    """End-to-end checks using real STO-3G integrals and a real Molecule."""

    def test_h2_density_shape(self):
        S, C, n_occ = _h2_sto3g_ingredients()
        P = density_matrix(C, n_occ=n_occ)
        assert P.shape == (2, 2)

    def test_h2_density_is_symmetric(self):
        S, C, n_occ = _h2_sto3g_ingredients()
        P = density_matrix(C, n_occ=n_occ)
        np.testing.assert_allclose(P, P.T, atol=1e-14)

    def test_h2_tr_ps_equals_n_electrons(self):
        # H2 has 2 electrons: Tr[PS] must equal 2.0
        S, C, n_occ = _h2_sto3g_ingredients()
        P = density_matrix(C, n_occ=n_occ)
        n_e = n_electrons_from_density(P, S)
        assert n_e == pytest.approx(2.0, rel=1e-10)

    def test_h2_density_idempotency_with_overlap(self):
        # For a converged RHF wavefunction, P satisfies PSP = 2P.
        # Our C comes from diagonalising a valid (if not fully converged) F,
        # so this should still hold since C_occ is orthonormal w.r.t. S.
        S, C, n_occ = _h2_sto3g_ingredients()
        P = density_matrix(C, n_occ=n_occ)
        # PSP = 2P is the generalised idempotency condition for RHF
        np.testing.assert_allclose(P @ S @ P, 2.0 * P, atol=1e-12)