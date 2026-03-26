"""
qchem/molecule.py — molecular geometry and bookkeeping

Overview
--------
The Molecule class is the central data container for a calculation.  It
stores nuclear positions and charges, derives electron counts from charge
and multiplicity, computes the nuclear-repulsion energy, and provides the
``atoms`` sequence that basis.build_basis() and the integrals layer expect.

Typical usage::

    from qchem.molecule import Molecule

    # From a list of (symbol, coords) pairs (bohr by default)
    mol = Molecule([('O',  [0.000,  0.000,  0.000]),
                    ('H',  [0.000,  1.430,  1.107]),
                    ('H',  [0.000, -1.430,  1.107])])

    # Or from an XYZ-format string (Ångström, automatically converted)
    mol = Molecule.from_xyz(\"\"\"
    3
    water
    O  0.000  0.000  0.000
    H  0.000  0.757 -0.471
    H  0.000 -0.757 -0.471
    \"\"\", charge=0, multiplicity=1)

    print(mol.n_electrons)          # 10
    print(mol.nuclear_repulsion())  # V_nn in hartree
    basis = mol.build_basis('sto-3g')

Units
-----
All internal coordinates are in **bohr** (atomic units).  The
``angstrom=True`` keyword (or Molecule.from_xyz, which always reads Å)
applies the CODATA 2018 conversion factor 1 Å = 1.8897259886 bohr.

Electron-count conventions
--------------------------
``charge``       — net charge of the molecule (+1 means one electron removed).
``multiplicity`` — 2S + 1, where S is the total spin.  Singlet = 1, doublet =
                   2, triplet = 3, …  Must be consistent with n_electrons
                   (even/odd parity must match).

The number of electrons is derived as::

    n_electrons = sum(Z_i) - charge

Alpha and beta counts follow the ROHF convention::

    n_unpaired = multiplicity - 1
    n_beta     = (n_electrons - n_unpaired) // 2
    n_alpha    = n_electrons - n_beta

References
----------
Szabo & Ostlund, "Modern Quantum Chemistry", Dover (1996), Ch. 3.
Helgaker, Jørgensen & Olsen, "Molecular Electronic-Structure Theory",
    Wiley (2000), Ch. 1.
"""

from __future__ import annotations

import re
import numpy as np
from typing import Sequence

from .basis import ATOMIC_NUMBER, BOHR_PER_ANGSTROM, build_basis, element_symbol


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Molecule:
    """
    A molecular system: nuclear geometry + charge + spin.

    Parameters
    ----------
    atoms : sequence of (symbol, coords) pairs
        ``symbol`` is a case-insensitive element symbol ('H', 'C', 'he', …).
        ``coords`` is array-like with shape (3,), in bohr by default.
    charge : int
        Net molecular charge.  Default 0 (neutral).
    multiplicity : int
        Spin multiplicity 2S + 1.  Default 1 (singlet).
    angstrom : bool
        If True, input coordinates are in Ångström and are converted to bohr.
        Default False.

    Attributes
    ----------
    atoms : list of (str, np.ndarray)
        Normalised (CAPITALISED_SYMBOL, bohr_coords) pairs — exactly the
        format expected by basis.build_basis().
    charge : int
    multiplicity : int
    n_electrons : int
    n_alpha : int
    n_beta : int

    Raises
    ------
    ValueError
        If multiplicity is < 1, n_electrons is negative, or the
        electron count and multiplicity have mismatched parity.
    """

    def __init__(
        self,
        atoms: Sequence[tuple[str, Sequence[float]]],
        charge: int = 0,
        multiplicity: int = 1,
        angstrom: bool = False,
    ) -> None:
        if multiplicity < 1:
            raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")

        # Normalise atoms: canonical symbol + numpy array in bohr
        self.atoms: list[tuple[str, np.ndarray]] = []
        for sym, coords in atoms:
            sym = element_symbol(sym)              # validates & capitalises
            center = np.asarray(coords, dtype=float)
            if center.shape != (3,):
                raise ValueError(
                    f"Each atom coordinate must have shape (3,), got {center.shape}"
                )
            if angstrom:
                center = center * BOHR_PER_ANGSTROM
            self.atoms.append((sym, center))

        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

        # Electron bookkeeping
        n_nuclear = sum(ATOMIC_NUMBER[sym] for sym, _ in self.atoms)
        self.n_electrons: int = n_nuclear - self.charge
        if self.n_electrons < 0:
            raise ValueError(
                f"Molecule has {n_nuclear} nuclear electrons but charge "
                f"{charge} implies {self.n_electrons} electrons."
            )

        n_unpaired = self.multiplicity - 1
        if (self.n_electrons - n_unpaired) % 2 != 0:
            raise ValueError(
                f"n_electrons={self.n_electrons} and "
                f"multiplicity={multiplicity} are inconsistent "
                f"(parity mismatch)."
            )
        if n_unpaired > self.n_electrons:
            raise ValueError(
                f"multiplicity={multiplicity} requires {n_unpaired} unpaired "
                f"electrons but the molecule only has {self.n_electrons}."
            )

        self.n_beta: int = (self.n_electrons - n_unpaired) // 2
        self.n_alpha: int = self.n_electrons - self.n_beta

    # ------------------------------------------------------------------
    # Derived molecular properties
    # ------------------------------------------------------------------

    def nuclear_repulsion(self) -> float:
        """
        Classical nuclear-repulsion energy V_nn = Σ_{A<B} Z_A Z_B / |R_A - R_B|.

        Returns
        -------
        float
            Nuclear repulsion energy in hartree.
        """
        energy = 0.0
        n = len(self.atoms)
        for i in range(n):
            sym_i, ri = self.atoms[i]
            zi = ATOMIC_NUMBER[sym_i]
            for j in range(i + 1, n):
                sym_j, rj = self.atoms[j]
                zj = ATOMIC_NUMBER[sym_j]
                rij = np.linalg.norm(ri - rj)
                energy += zi * zj / rij
        return energy

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)

    @property
    def symbols(self) -> list[str]:
        """Ordered list of element symbols."""
        return [sym for sym, _ in self.atoms]

    @property
    def coords(self) -> np.ndarray:
        """
        Atomic coordinates as an (n_atoms, 3) array in bohr.
        """
        return np.array([r for _, r in self.atoms])

    @property
    def atomic_numbers(self) -> list[int]:
        """Ordered list of atomic numbers (Z values)."""
        return [ATOMIC_NUMBER[sym] for sym, _ in self.atoms]

    @property
    def is_closed_shell(self) -> bool:
        """True if n_alpha == n_beta (RHF-eligible)."""
        return self.n_alpha == self.n_beta

    # ------------------------------------------------------------------
    # Integral / basis interface
    # ------------------------------------------------------------------

    def build_basis(self, basis_name: str = 'sto-3g') -> list[dict]:
        """
        Build the AO basis for this molecule.

        Delegates directly to ``basis.build_basis()``.  Coordinates are
        already in bohr, so ``angstrom=False`` is used.

        Parameters
        ----------
        basis_name : str
            Registered basis set name (case-insensitive).  Default: 'sto-3g'.

        Returns
        -------
        list[dict]
            Ordered shell dicts as described in ``basis.build_basis()``.
        """
        return build_basis(self.atoms, basis_name=basis_name, angstrom=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_xyz(
        cls,
        xyz: str,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> "Molecule":
        """
        Construct a Molecule from an XYZ-format string.

        The standard XYZ format is::

            <n_atoms>
            <comment line (ignored)>
            <symbol>  <x>  <y>  <z>
            ...

        Coordinates are in **Ångström** (XYZ convention) and are converted
        to bohr automatically.

        The first two lines (atom count and comment) are optional: if the
        first non-blank line is not a plain integer, the parser assumes the
        header is absent and reads straight atom lines.

        Parameters
        ----------
        xyz : str
            XYZ-format geometry string.
        charge : int
            Net molecular charge.  Default 0.
        multiplicity : int
            Spin multiplicity.  Default 1 (singlet).

        Returns
        -------
        Molecule

        Examples
        --------
        >>> mol = Molecule.from_xyz(\"\"\"
        ... 3
        ... water molecule
        ... O  0.000  0.000  0.000
        ... H  0.000  0.757 -0.471
        ... H  0.000 -0.757 -0.471
        ... \"\"\")
        >>> mol.n_electrons
        10
        """
        lines = [ln.strip() for ln in xyz.strip().splitlines() if ln.strip()]
        if not lines:
            raise ValueError("Empty XYZ string.")

        # Detect and skip header (atom-count line + comment line)
        start = 0
        if re.fullmatch(r'\d+', lines[0]):
            start = 2   # skip count + comment

        atoms = []
        for ln in lines[start:]:
            parts = ln.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Expected '<symbol> x y z', got: {ln!r}"
                )
            sym = parts[0]
            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError:
                raise ValueError(
                    f"Could not parse coordinates from: {ln!r}"
                )
            atoms.append((sym, coords))

        if not atoms:
            raise ValueError("No atom lines found in XYZ string.")

        return cls(atoms, charge=charge, multiplicity=multiplicity, angstrom=True)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Molecule(n_atoms={self.n_atoms}, "
            f"n_electrons={self.n_electrons}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity})"
        )

    def __str__(self) -> str:
        """Human-readable geometry summary (coordinates in bohr)."""
        header = (
            f"Molecule: {self.n_atoms} atoms, "
            f"{self.n_electrons} electrons "
            f"(charge={self.charge}, mult={self.multiplicity})\n"
            f"  {'Atom':<6}  {'x (bohr)':>12}  {'y (bohr)':>12}  {'z (bohr)':>12}"
        )
        rows = [header]
        for sym, r in self.atoms:
            rows.append(f"  {sym:<6}  {r[0]:>12.6f}  {r[1]:>12.6f}  {r[2]:>12.6f}")
        rows.append(f"  V_nn = {self.nuclear_repulsion():.10f} hartree")
        return "\n".join(rows)
