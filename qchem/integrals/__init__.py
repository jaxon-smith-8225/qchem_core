"""
qchem.integrals — one-electron and two-electron integral matrices.

Exposes the four matrix/tensor builders that the SCF layer consumes::

    build_overlap_matrix(basis)            →  S  (n, n)
    build_kinetic_matrix(basis)            →  T  (n, n)
    build_nuclear_matrix(basis, charges, coords)  →  V  (n, n)
    build_eri_tensor(basis)                →  (μν|λσ)  (n, n, n, n)

All functions expect a ``basis`` list produced by ``basis.build_basis()``.
Coordinates and charges for the nuclear matrix come from
``Molecule.coords`` and ``Molecule.atomic_numbers`` respectively.
"""

from .overlap import build_overlap_matrix
from .kinetic import build_kinetic_matrix
from .nuclear import build_nuclear_matrix
from .eri import build_eri_tensor

__all__ = [
    "build_overlap_matrix",
    "build_kinetic_matrix",
    "build_nuclear_matrix",
    "build_eri_tensor",
]