"""
Tests for qchem/scf/diis.py

Structure
---------
TestInit            — constructor validation
TestPush            — storage, rolling window, shape validation
TestBuildB          — B matrix correctness and symmetry
TestSolve           — coefficient properties (sum=1, known solutions)
TestExtrapolate     — single-vector passthrough, linear combination,
                      mathematical correctness
TestReset           — clear/restart behaviour
TestPhysical        — end-to-end with real H2/STO-3G matrices
TestInvariants      — properties that must hold for any valid inputs
"""

import numpy as np
import pytest

from qchem.scf.diis import DIISAccelerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_symmetric(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random symmetric n×n matrix."""
    A = rng.standard_normal((n, n))
    return 0.5 * (A + A.T)


def _random_fock_error(n: int, rng: np.random.Generator, scale: float = 1.0):
    """Return a pair (F, e) with e as a flat vector of length n²."""
    F = _random_symmetric(n, rng)
    e = rng.standard_normal(n * n) * scale
    return F, e


def _make_diis(n: int = 4, max_vec: int = 6) -> DIISAccelerator:
    return DIISAccelerator(max_vec=max_vec)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_max_vec(self):
        diis = DIISAccelerator()
        assert diis.max_vec == 8

    def test_custom_max_vec(self):
        diis = DIISAccelerator(max_vec=4)
        assert diis.max_vec == 4

    def test_invalid_max_vec_zero(self):
        with pytest.raises(ValueError, match="max_vec"):
            DIISAccelerator(max_vec=0)

    def test_invalid_max_vec_negative(self):
        with pytest.raises(ValueError, match="max_vec"):
            DIISAccelerator(max_vec=-3)

    def test_initial_len_is_zero(self):
        assert len(DIISAccelerator()) == 0

    def test_repr_contains_max_vec(self):
        diis = DIISAccelerator(max_vec=5)
        assert "5" in repr(diis)

    def test_repr_contains_stored(self):
        diis = DIISAccelerator(max_vec=5)
        assert "stored=0" in repr(diis)


# ---------------------------------------------------------------------------
# TestPush
# ---------------------------------------------------------------------------

class TestPush:
    def test_push_increases_len(self):
        rng = np.random.default_rng(0)
        diis = DIISAccelerator(max_vec=8)
        for k in range(1, 6):
            F, e = _random_fock_error(3, rng)
            diis.push(F, e)
            assert len(diis) == k

    def test_push_caps_at_max_vec(self):
        rng = np.random.default_rng(1)
        max_vec = 4
        diis = DIISAccelerator(max_vec=max_vec)
        for _ in range(max_vec + 3):
            F, e = _random_fock_error(3, rng)
            diis.push(F, e)
        assert len(diis) == max_vec

    def test_rolling_window_evicts_oldest(self):
        """The oldest Fock matrix should not be present after eviction."""
        rng = np.random.default_rng(2)
        diis = DIISAccelerator(max_vec=2)
        F0, e0 = _random_fock_error(3, rng)
        F1, e1 = _random_fock_error(3, rng)
        F2, e2 = _random_fock_error(3, rng)
        diis.push(F0, e0)
        diis.push(F1, e1)
        diis.push(F2, e2)  # should evict F0
        # F0 should no longer appear in the history
        for stored in diis._fock_history:
            assert not np.allclose(stored, F0)

    def test_push_stores_copy(self):
        """Mutating F after push must not change the stored version."""
        rng = np.random.default_rng(3)
        diis = DIISAccelerator(max_vec=4)
        F, e = _random_fock_error(3, rng)
        F_orig = F.copy()
        diis.push(F, e)
        F[:] = 0.0
        assert np.allclose(diis._fock_history[0], F_orig)

    def test_push_accepts_2d_error(self):
        """e may be passed as a 2-D matrix; it should be stored as 1-D."""
        rng = np.random.default_rng(4)
        diis = DIISAccelerator()
        n = 4
        F = _random_symmetric(n, rng)
        e = rng.standard_normal((n, n))
        diis.push(F, e)
        assert diis._error_history[0].ndim == 1
        assert diis._error_history[0].size == n * n

    def test_push_non_square_F_raises(self):
        diis = DIISAccelerator()
        F = np.eye(3, 4)
        e = np.zeros(12)
        with pytest.raises(ValueError, match="square"):
            diis.push(F, e)

    def test_push_1d_F_raises(self):
        diis = DIISAccelerator()
        with pytest.raises(ValueError, match="square"):
            diis.push(np.ones(9), np.zeros(9))

    def test_push_wrong_e_size_raises(self):
        rng = np.random.default_rng(5)
        diis = DIISAccelerator()
        n = 3
        F = _random_symmetric(n, rng)
        e_bad = np.zeros(n * n + 1)
        with pytest.raises(ValueError, match="n\\*n"):
            diis.push(F, e_bad)

    def test_push_inconsistent_F_shape_raises(self):
        rng = np.random.default_rng(6)
        diis = DIISAccelerator()
        F3, e3 = _random_fock_error(3, rng)
        F4, e4 = _random_fock_error(4, rng)
        diis.push(F3, e3)
        with pytest.raises(ValueError, match="inconsistent"):
            diis.push(F4, e4)


# ---------------------------------------------------------------------------
# TestBuildB
# ---------------------------------------------------------------------------

class TestBuildB:
    def test_b_is_symmetric(self):
        rng = np.random.default_rng(10)
        diis = DIISAccelerator()
        for _ in range(4):
            F, e = _random_fock_error(3, rng)
            diis.push(F, e)
        B = diis._build_B()
        assert np.allclose(B, B.T)

    def test_b_diagonal_equals_squared_norm(self):
        rng = np.random.default_rng(11)
        diis = DIISAccelerator()
        n = 3
        errors = []
        for _ in range(3):
            F, e = _random_fock_error(n, rng)
            diis.push(F, e)
            errors.append(diis._error_history[-1])
        B = diis._build_B()
        for i, e in enumerate(errors):
            assert pytest.approx(B[i, i], rel=1e-12) == np.dot(e, e)

    def test_b_off_diagonal_is_dot_product(self):
        rng = np.random.default_rng(12)
        diis = DIISAccelerator()
        n = 3
        errors = []
        for _ in range(3):
            F, e = _random_fock_error(n, rng)
            diis.push(F, e)
            errors.append(diis._error_history[-1])
        B = diis._build_B()
        assert pytest.approx(B[0, 1], rel=1e-12) == np.dot(errors[0], errors[1])
        assert pytest.approx(B[1, 2], rel=1e-12) == np.dot(errors[1], errors[2])

    def test_b_shape(self):
        rng = np.random.default_rng(13)
        diis = DIISAccelerator()
        k = 5
        for _ in range(k):
            diis.push(*_random_fock_error(3, rng))
        B = diis._build_B()
        assert B.shape == (k, k)


# ---------------------------------------------------------------------------
# TestSolve
# ---------------------------------------------------------------------------

class TestSolve:
    def test_coefficients_sum_to_one(self):
        """The sum constraint must hold for all valid B matrices."""
        rng = np.random.default_rng(20)
        for n_vec in range(2, 7):
            diis = DIISAccelerator()
            for _ in range(n_vec):
                diis.push(*_random_fock_error(4, rng))
            B = diis._build_B()
            c = diis._solve(B)
            assert pytest.approx(c.sum(), abs=1e-10) == 1.0

    def test_single_vector_trivial_coefficient(self):
        """With one error vector, c = [1] is forced by the constraint."""
        rng = np.random.default_rng(21)
        diis = DIISAccelerator()
        diis.push(*_random_fock_error(3, rng))
        B = diis._build_B()
        c = diis._solve(B)
        assert pytest.approx(c[0], abs=1e-12) == 1.0

    def test_two_identical_errors_give_equal_weights(self):
        """If both error vectors are identical, c = [0.5, 0.5]."""
        rng = np.random.default_rng(22)
        n = 3
        F1 = _random_symmetric(n, rng)
        e = rng.standard_normal(n * n)
        F2 = _random_symmetric(n, rng)  # different F, same e
        diis = DIISAccelerator()
        diis.push(F1, e)
        diis.push(F2, e)
        B = diis._build_B()
        c = diis._solve(B)
        assert pytest.approx(c[0], abs=1e-10) == 0.5
        assert pytest.approx(c[1], abs=1e-10) == 0.5

    def test_perfectly_converged_error_gets_weight_one(self):
        """
        If one error vector is zero (converged) and another is not, the
        zero-error iterate should receive all the weight.  We approximate
        this by making e2 tiny rather than exactly zero to avoid
        degeneracy in the B matrix.
        """
        rng = np.random.default_rng(23)
        n = 3
        F_big = _random_symmetric(n, rng)
        e_big = rng.standard_normal(n * n)          # large error
        F_tiny = _random_symmetric(n, rng)
        e_tiny = rng.standard_normal(n * n) * 1e-12  # essentially zero
        diis = DIISAccelerator()
        diis.push(F_big, e_big)
        diis.push(F_tiny, e_tiny)
        B = diis._build_B()
        c = diis._solve(B)
        # The tiny-error iterate should receive essentially all the weight
        assert c[1] > 0.99

    def test_coefficients_length_matches_history(self):
        rng = np.random.default_rng(24)
        n_vec = 5
        diis = DIISAccelerator()
        for _ in range(n_vec):
            diis.push(*_random_fock_error(3, rng))
        B = diis._build_B()
        c = diis._solve(B)
        assert len(c) == n_vec


# ---------------------------------------------------------------------------
# TestExtrapolate
# ---------------------------------------------------------------------------

class TestExtrapolate:
    def test_extrapolate_before_push_raises(self):
        diis = DIISAccelerator()
        with pytest.raises(RuntimeError, match="push"):
            diis.extrapolate()

    def test_single_push_returns_f_unchanged(self):
        rng = np.random.default_rng(30)
        n = 4
        F, e = _random_fock_error(n, rng)
        diis = DIISAccelerator()
        diis.push(F, e)
        F_star = diis.extrapolate()
        assert np.allclose(F_star, F)

    def test_extrapolated_f_is_linear_combination(self):
        """F* must lie in the affine span of the stored Fock matrices."""
        rng = np.random.default_rng(31)
        n = 4
        diis = DIISAccelerator(max_vec=8)
        n_vec = 4
        for _ in range(n_vec):
            diis.push(*_random_fock_error(n, rng))

        F_star = diis.extrapolate()
        B = diis._build_B()
        c = diis._solve(B)

        F_manual = sum(
            c[i] * diis._fock_history[i] for i in range(n_vec)
        )
        assert np.allclose(F_star, F_manual, atol=1e-12)

    def test_extrapolated_f_shape(self):
        rng = np.random.default_rng(32)
        n = 5
        diis = DIISAccelerator()
        for _ in range(3):
            diis.push(*_random_fock_error(n, rng))
        F_star = diis.extrapolate()
        assert F_star.shape == (n, n)

    def test_extrapolated_f_is_new_array(self):
        """extrapolate() must not return a reference to an internal array."""
        rng = np.random.default_rng(33)
        diis = DIISAccelerator()
        diis.push(*_random_fock_error(3, rng))
        F_star = diis.extrapolate()
        F_star[:] = 0.0
        # Internal history must be unchanged
        assert not np.all(diis._fock_history[-1] == 0.0)

    def test_extrapolate_minimises_residual_norm(self):
        """
        The DIIS extrapolation must give a smaller residual than any
        single stored iterate (for a well-conditioned history).
        """
        rng = np.random.default_rng(34)
        n = 4
        n_vec = 5
        diis = DIISAccelerator(max_vec=8)
        for k in range(n_vec):
            F, e = _random_fock_error(n, rng, scale=1.0 / (k + 1))
            diis.push(F, e)

        B = diis._build_B()
        c = diis._solve(B)

        # Residual of the DIIS combination
        E = np.array(diis._error_history)
        residual_diis = np.linalg.norm(c @ E)

        # Residual of each individual iterate (c_i = 1, rest = 0)
        for i in range(n_vec):
            c_single = np.zeros(n_vec)
            c_single[i] = 1.0
            residual_single = np.linalg.norm(c_single @ E)
            assert residual_diis <= residual_single + 1e-12


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_history(self):
        rng = np.random.default_rng(40)
        diis = DIISAccelerator()
        for _ in range(4):
            diis.push(*_random_fock_error(3, rng))
        diis.reset()
        assert len(diis) == 0

    def test_extrapolate_after_reset_raises(self):
        rng = np.random.default_rng(41)
        diis = DIISAccelerator()
        diis.push(*_random_fock_error(3, rng))
        diis.reset()
        with pytest.raises(RuntimeError):
            diis.extrapolate()

    def test_push_after_reset_accepts_different_n(self):
        """After reset, a new basis size should be accepted."""
        rng = np.random.default_rng(42)
        diis = DIISAccelerator()
        diis.push(*_random_fock_error(3, rng))
        diis.reset()
        F4, e4 = _random_fock_error(4, rng)
        diis.push(F4, e4)  # should not raise
        assert len(diis) == 1


# ---------------------------------------------------------------------------
# TestPhysical — H2 / STO-3G
# ---------------------------------------------------------------------------

class TestPhysical:
    """
    End-to-end checks using real H2/STO-3G matrices.

    We do not run a full SCF loop here (that lives in test_hartree_fock.py
    once that module exists).  Instead we verify that DIIS behaves
    correctly when fed physically meaningful Fock and overlap matrices.
    """

    @pytest.fixture(scope="class")
    def h2_matrices(self):
        """H2 / STO-3G integral matrices at R = 1.4 bohr."""
        from qchem.molecule import Molecule
        from qchem.integrals.overlap import build_overlap_matrix
        from qchem.integrals.kinetic import build_kinetic_matrix
        from qchem.integrals.nuclear import build_nuclear_matrix
        from qchem.integrals.eri import build_eri_tensor
        from qchem.scf.density import density_matrix
        from qchem.scf.fock import core_hamiltonian, fock_matrix
        from qchem import linalg

        mol   = Molecule([("H", [0., 0., -0.7]), ("H", [0., 0., 0.7])])
        basis = mol.build_basis("sto-3g")
        S     = build_overlap_matrix(basis)
        T     = build_kinetic_matrix(basis)
        V     = build_nuclear_matrix(basis, mol.atomic_numbers, mol.coords)
        ERI   = build_eri_tensor(basis)
        H     = core_hamiltonian(T, V)

        # Guess: core Hamiltonian diagonalization
        _, C0 = linalg.solve_generalized_eigenvalue(H, S)
        P0    = density_matrix(C0, n_occ=1)
        F0    = fock_matrix(H, P0, ERI)

        return dict(S=S, H=H, ERI=ERI, F0=F0, P0=P0, C0=C0, mol=mol)

    def test_push_h2_fock_matrix(self, h2_matrices):
        diis = DIISAccelerator()
        F = h2_matrices["F0"]
        P = h2_matrices["P0"]
        S = h2_matrices["S"]
        e = F @ P @ S - S @ P @ F
        diis.push(F, e)
        assert len(diis) == 1

    def test_error_vector_is_antisymmetric(self, h2_matrices):
        """
        The orbital gradient e = FPS - SPF must be antisymmetric:
        e.T = -e.  This is a necessary property of the commutator.
        """
        F = h2_matrices["F0"]
        P = h2_matrices["P0"]
        S = h2_matrices["S"]
        e = F @ P @ S - S @ P @ F
        assert np.allclose(e + e.T, 0.0, atol=1e-12)

    def test_extrapolate_single_h2_returns_f0(self, h2_matrices):
        diis = DIISAccelerator()
        F = h2_matrices["F0"]
        P = h2_matrices["P0"]
        S = h2_matrices["S"]
        e = F @ P @ S - S @ P @ F
        diis.push(F, e)
        F_star = diis.extrapolate()
        assert np.allclose(F_star, F, atol=1e-14)

    def test_diis_over_manufactured_scf_steps(self, h2_matrices):
        """
        Simulate several SCF-like iterations with real H2 matrices and
        verify that the DIIS coefficients always sum to 1 and the
        extrapolated Fock matrix is symmetric.
        """
        from qchem.scf.density import density_matrix
        from qchem.scf.fock import fock_matrix
        from qchem import linalg

        S   = h2_matrices["S"]
        H   = h2_matrices["H"]
        ERI = h2_matrices["ERI"]

        diis = DIISAccelerator(max_vec=6)
        _, C = linalg.solve_generalized_eigenvalue(H, S)
        P    = density_matrix(C, n_occ=1)

        for _ in range(6):
            F = fock_matrix(H, P, ERI)
            e = F @ P @ S - S @ P @ F
            diis.push(F, e)
            F_star = diis.extrapolate()

            # Symmetry: DIIS combination of symmetric matrices is symmetric
            assert np.allclose(F_star, F_star.T, atol=1e-13)

            # Update density for next iteration
            _, C = linalg.solve_generalized_eigenvalue(F_star, S)
            P    = density_matrix(C, n_occ=1)

        # After 6 iterations the coefficients should still sum to 1
        B = diis._build_B()
        c = diis._solve(B)
        assert pytest.approx(c.sum(), abs=1e-10) == 1.0


# ---------------------------------------------------------------------------
# TestInvariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """
    Properties that must hold for any valid inputs, tested with a range
    of sizes and random seeds.
    """

    @pytest.mark.parametrize("n,n_vec,seed", [
        (2, 2, 100), (3, 4, 101), (4, 6, 102), (5, 3, 103),
    ])
    def test_coefficients_always_sum_to_one(self, n, n_vec, seed):
        rng = np.random.default_rng(seed)
        diis = DIISAccelerator(max_vec=8)
        for _ in range(n_vec):
            diis.push(*_random_fock_error(n, rng))
        B = diis._build_B()
        c = diis._solve(B)
        assert pytest.approx(c.sum(), abs=1e-10) == 1.0

    @pytest.mark.parametrize("n,n_vec,seed", [
        (3, 3, 200), (4, 5, 201), (5, 4, 202),
    ])
    def test_extrapolated_f_is_symmetric(self, n, n_vec, seed):
        """Linear combination of symmetric matrices is symmetric."""
        rng = np.random.default_rng(seed)
        diis = DIISAccelerator(max_vec=8)
        for _ in range(n_vec):
            F = _random_symmetric(n, rng)
            e = rng.standard_normal(n * n)
            diis.push(F, e)
        F_star = diis.extrapolate()
        assert np.allclose(F_star, F_star.T, atol=1e-13)

    @pytest.mark.parametrize("n_vec,seed", [
        (3, 300), (5, 301), (8, 302),
    ])
    def test_len_matches_n_vec(self, n_vec, seed):
        rng = np.random.default_rng(seed)
        diis = DIISAccelerator(max_vec=10)
        for k in range(n_vec):
            diis.push(*_random_fock_error(3, rng))
            assert len(diis) == k + 1
