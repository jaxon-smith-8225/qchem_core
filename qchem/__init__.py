"""
qchem — a ground-up implementation of Hartree-Fock and Kohn-Sham DFT.

Importing this package exposes the most commonly used entry points directly
under the ``qchem`` namespace so user code can write::

    from qchem import Molecule
    from qchem.basis import build_basis
    from qchem.integrals import build_overlap_matrix, build_kinetic_matrix
    from qchem.integrals import build_nuclear_matrix, build_eri_tensor

Everything else (linalg utilities, individual integral primitives, SCF
internals) is still accessible via its full dotted path.
"""

from .molecule import Molecule
from .basis import build_basis, basis_info, register_basis, load_nwchem

__all__ = [
    "Molecule",
    "build_basis",
    "basis_info",
    "register_basis",
    "load_nwchem",
]
