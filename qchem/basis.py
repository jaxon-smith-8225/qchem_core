"""
qchem/basis.py — contracted Gaussian basis set library

Overview
--------
This module stores basis-set data and converts it into the shell-dict
format consumed by every function in qchem.integrals:

    {
        'center':       np.ndarray, shape (3,)   — atomic centre in bohr
        'angular':      tuple (lx, ly, lz)       — Cartesian angular momentum
        'exponents':    list[float]               — primitive exponents
        'coefficients': list[float]               — contraction coefficients
    }

Each contracted shell (e.g. a 2p shell on carbon) expands into one dict
per Cartesian component: px, py, pz → three dicts.  The integrals layer
treats each dict as one basis function, so the list returned by
build_basis() is the full AO basis ordered atom-by-atom, shell-by-shell,
component-by-component.

Basis sets currently built in
------------------------------
sto-3g   H–Ne     Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657 (1969).

Everything else can be added via register_basis() (programmatic) or by
parsing a Basis Set Exchange NWChem-format string with load_nwchem().

Units
-----
All coordinates are in bohr (atomic units).  Input geometry is almost
always supplied in Ångström; use angstrom_to_bohr() before calling
build_basis(), or pass angstrom=True to build_basis().

Angular momentum conventions
----------------------------
Shell types and their Cartesian components:

    s  (l=0)  →  (0,0,0)
    p  (l=1)  →  (1,0,0)  (0,1,0)  (0,0,1)
    d  (l=2)  →  (2,0,0)  (0,2,0)  (0,0,2)  (1,1,0)  (1,0,1)  (0,1,1)
    f  (l=3)  →  10 components (see _ANG_COMPONENTS)

SP shells
---------
Pople-style basis sets (STO-3G for Li–Ne, 6-31G, etc.) use SP shells:
one set of primitive exponents shared between an s contraction and a p
contraction, each with its own coefficient list.  Internally these are
stored as shell_type='sp' and expand into one s function + three p
functions.
"""

from __future__ import annotations

import re
import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

BOHR_PER_ANGSTROM: float = 1.8897259886  # 1 Å in bohr (CODATA 2018)


def angstrom_to_bohr(coords: np.ndarray) -> np.ndarray:
    """Convert a coordinate array from Ångström to bohr."""
    return np.asarray(coords, dtype=float) * BOHR_PER_ANGSTROM


# ---------------------------------------------------------------------------
# Element tables
# ---------------------------------------------------------------------------

# Symbol → atomic number (Z) for Z = 1–36 (H through Kr).
# Extended on demand; only what we need for basis data is listed.
ATOMIC_NUMBER: dict[str, int] = {
    'H': 1,  'He': 2,
    'Li': 3,  'Be': 4,  'B': 5,   'C': 6,   'N': 7,   'O': 8,
    'F': 9,  'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,  'S': 16,
    'Cl': 17, 'Ar': 18,
    'K': 19,  'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,  'Cr': 24,
    'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
}

ELEMENT_SYMBOL: dict[int, str] = {v: k for k, v in ATOMIC_NUMBER.items()}


def element_symbol(z_or_name: int | str) -> str:
    """
    Normalise an element identifier to a capitalised symbol string.

    Accepts an atomic number (int) or a symbol string in any case.
    Raises KeyError for unknown elements.
    """
    if isinstance(z_or_name, int):
        return ELEMENT_SYMBOL[z_or_name]
    sym = z_or_name.strip().capitalize()
    if sym not in ATOMIC_NUMBER:
        raise KeyError(f"Unknown element: {z_or_name!r}")
    return sym


# ---------------------------------------------------------------------------
# Angular momentum component table
# ---------------------------------------------------------------------------

# Maps total angular momentum l → ordered list of (lx, ly, lz) tuples.
# Order within each shell: xx yy zz xy xz yz (matches most QC codes).
_ANG_COMPONENTS: dict[int, list[tuple[int, int, int]]] = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [
        (3, 0, 0), (0, 3, 0), (0, 0, 3),
        (2, 1, 0), (2, 0, 1), (1, 2, 0),
        (0, 2, 1), (1, 0, 2), (0, 1, 2),
        (1, 1, 1),
    ],
}

_SHELL_TYPE_TO_L: dict[str, int] = {'s': 0, 'p': 1, 'd': 2, 'f': 3}


# ---------------------------------------------------------------------------
# Raw basis-set registry
# ---------------------------------------------------------------------------
# _REGISTRY[basis_name][element_symbol] = list of shell records
#
# Each shell record is a dict:
#   {
#       'type':          's' | 'p' | 'd' | 'f' | 'sp'
#       'exponents':     list[float]
#       'coefficients':  list[float]          (s part, or only part for s/p/d/f)
#       'p_coefficients': list[float] | None  (p part; only present for 'sp')
#   }
#
# This private format is intentionally simple — one layer of indirection
# before the public shell-dict format.

_REGISTRY: dict[str, dict[str, list[dict]]] = {}


def register_basis(name: str, data: dict[str, list[dict]]) -> None:
    """
    Register a basis set programmatically.

    Parameters
    ----------
    name : str
        Basis set name, case-insensitive (stored lowercase).
    data : dict
        Maps element symbol → list of shell records in the internal format
        described in _REGISTRY's docstring above.

    Example
    -------
    register_basis('my-basis', {
        'H': [
            {'type': 's',
             'exponents':    [3.4252509, 0.6239137, 0.1688554],
             'coefficients': [0.1543290, 0.5353281, 0.4446345]},
        ],
    })
    """
    _REGISTRY[name.lower()] = {
        element_symbol(sym): shells for sym, shells in data.items()
    }


# ---------------------------------------------------------------------------
# STO-3G data (H–Ne)
# Reference: Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657 (1969).
#            Also available at https://www.basissetexchange.org
# ---------------------------------------------------------------------------
#
# Notation
# --------
# s-only shells:  ('s', exponents, s_coefficients)
# SP shells:      ('sp', exponents, s_coefficients, p_coefficients)
#
# All 2nd-row elements (Li–Ne) have two shells:
#   Shell 1 — 1s  (s-only, three primitives)
#   Shell 2 — 2sp (SP, three primitives)
#
# The SP contraction coefficients are the same for every element in this
# row; only the exponents differ.

_STO3G_SP_S = [-0.0999672,  0.3995128,  0.7001155]
_STO3G_SP_P = [ 0.1559163,  0.6076837,  0.3919574]

_STO3G_RAW: dict[str, list] = {
    # Symbol: list of (type, exponents, s_coeffs [, p_coeffs])
    'H': [
        ('s', [3.4252509, 0.6239137, 0.1688554],
              [0.1543290, 0.5353281, 0.4446345]),
    ],
    'He': [
        ('s', [6.3624214, 1.1589230, 0.3136498],
              [0.1543290, 0.5353281, 0.4446345]),
    ],
    'Li': [
        ('s',  [16.1195750,  2.9362007,  0.7946505],
               [ 0.1543290,  0.5353281,  0.4446345]),
        ('sp', [ 0.6362897,  0.1478601,  0.0480887],
               _STO3G_SP_S, _STO3G_SP_P),
    ],
    'Be': [
        ('s',  [30.1678710,  5.4951153,  1.4871927],
               [ 0.1543290,  0.5353281,  0.4446345]),
        ('sp', [ 1.3148331,  0.3055389,  0.0993707],
               _STO3G_SP_S, _STO3G_SP_P),
    ],
    'B': [
        ('s',  [48.7911130,  8.8873622,  2.4052670],
               [ 0.1543290,  0.5353281,  0.4446345]),
        ('sp', [ 2.2369561,  0.5198205,  0.1690618],
               _STO3G_SP_S, _STO3G_SP_P),
    ],
    'C': [
        ('s',  [71.6168370, 13.0450963,  3.5305122],
               [ 0.1543290,  0.5353281,  0.4446345]),
        ('sp', [ 2.9412494,  0.6834831,  0.2222899],
               _STO3G_SP_S, _STO3G_SP_P),
    ],
    'N': [
        ('s',  [99.1061690, 18.0523120,  4.8856602],
               [ 0.1543290,  0.5353281,  0.4446345]),
        ('sp', [ 3.7804559,  0.8784966,  0.2857144],
               _STO3G_SP_S, _STO3G_SP_P),
    ],
    'O': [
        ('s',  [130.7093200, 23.8088610,  6.4436083],
               [  0.1543290,  0.5353281,  0.4446345]),
        ('sp', [  5.0331513,  1.1695961,  0.3803890],
               _STO3G_SP_S, _STO3G_SP_P),
    ],
    'F': [
        ('s',  [166.6791340, 30.3608123,  8.2168207],
               [  0.1543290,  0.5353281,  0.4446345]),
        ('sp', [  6.4648032,  1.5022812,  0.4885885],
               _STO3G_SP_S, _STO3G_SP_P),
    ],
    'Ne': [
        ('s',  [207.0155630, 37.7081510, 10.2052580],
               [  0.1543290,  0.5353281,  0.4446345]),
        ('sp', [  8.2463151,  1.9162662,  0.6232293],
               _STO3G_SP_S, _STO3G_SP_P),
    ],
}


def _sto3g_raw_to_registry(raw: dict[str, list]) -> dict[str, list[dict]]:
    """Convert the compact tuple notation to the full registry dict format."""
    result: dict[str, list[dict]] = {}
    for sym, shells in raw.items():
        records = []
        for entry in shells:
            if entry[0] == 'sp':
                _, exps, s_coeff, p_coeff = entry
                records.append({
                    'type': 'sp',
                    'exponents': list(exps),
                    'coefficients': list(s_coeff),
                    'p_coefficients': list(p_coeff),
                })
            else:
                shell_type, exps, coeff = entry
                records.append({
                    'type': shell_type,
                    'exponents': list(exps),
                    'coefficients': list(coeff),
                    'p_coefficients': None,
                })
        result[sym] = records
    return result


register_basis('sto-3g', _sto3g_raw_to_registry(_STO3G_RAW))


# ---------------------------------------------------------------------------
# NWChem-format parser  (Basis Set Exchange default export format)
# ---------------------------------------------------------------------------

def load_nwchem(text: str, basis_name: str) -> None:
    """
    Parse a basis set in NWChem format and register it under basis_name.

    This lets you paste in a block from https://www.basissetexchange.org
    (choose "NWChem" as the format) and register it without touching the
    source of this file.

    Parameters
    ----------
    text       : str   Full text of the NWChem basis set block.
    basis_name : str   Name to register under (e.g. '6-31g').

    Example
    -------
    import requests
    from qchem.basis import load_nwchem

    url = ('https://www.basissetexchange.org/api/basis/6-31g/format/nwchem/'
           '?elements=1,6,7,8')
    load_nwchem(requests.get(url).text, '6-31g')
    """
    data: dict[str, list[dict]] = {}
    current_sym: str | None = None
    current_type: str | None = None
    current_exps: list[float] = []
    current_s_coeffs: list[float] = []
    current_p_coeffs: list[float] = []

    def flush():
        if current_sym is None or current_type is None or not current_exps:
            return
        record: dict = {
            'type': current_type.lower(),
            'exponents': list(current_exps),
            'coefficients': list(current_s_coeffs),
            'p_coefficients': list(current_p_coeffs) if current_p_coeffs else None,
        }
        data.setdefault(current_sym, []).append(record)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        # BASIS "..." SPHERICAL / CARTESIAN
        if re.match(r'^BASIS\b', line, re.IGNORECASE):
            continue
        if line.upper() in ('END', 'END.'):
            flush()
            current_sym = None  # prevent the post-loop flush() from double-appending
            break

        # "#BASIS_SET ..." — BSE comment line, skip
        if line.startswith('#BASIS_SET'):
            continue

        # Element + shell-type line, e.g. "H    S" or "C    SP"
        m = re.match(r'^([A-Za-z]{1,3})\s+([SPDFspdf]{1,2})\s*$', line)
        if m:
            flush()
            current_sym = element_symbol(m.group(1))
            current_type = m.group(2).lower()
            current_exps = []
            current_s_coeffs = []
            current_p_coeffs = []
            continue

        # Data line: one or two floats after the exponent
        parts = line.replace('D', 'E').replace('d', 'e').split()
        if len(parts) >= 2 and current_type is not None:
            try:
                current_exps.append(float(parts[0]))
                current_s_coeffs.append(float(parts[1]))
                if len(parts) >= 3:
                    current_p_coeffs.append(float(parts[2]))
            except ValueError:
                pass  # header lines with non-numeric content — ignore

    flush()
    register_basis(basis_name, data)


# ---------------------------------------------------------------------------
# Shell-dict builder
# ---------------------------------------------------------------------------

def _shells_for_atom(symbol: str, basis_name: str,
                     center: np.ndarray) -> list[dict]:
    """
    Return the list of shell dicts for a single atom at the given centre.

    Each shell record in the registry expands into one dict per Cartesian
    angular-momentum component; an SP record expands into (1 + 3) = 4 dicts.

    Parameters
    ----------
    symbol     : str           Capitalised element symbol, e.g. 'C'.
    basis_name : str           Lower-case basis set name, e.g. 'sto-3g'.
    center     : ndarray (3,)  Atomic position in bohr.

    Returns
    -------
    list[dict]  Shell dicts ready for the integrals layer.
    """
    key = basis_name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Basis set {basis_name!r} is not registered.  "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    sym = element_symbol(symbol)
    if sym not in _REGISTRY[key]:
        raise KeyError(
            f"Element {sym!r} is not available in basis set {basis_name!r}.  "
            "Use load_nwchem() to add it, or pick a different basis."
        )

    shells: list[dict] = []
    for record in _REGISTRY[key][sym]:
        stype = record['type']
        exps = record['exponents']
        s_coeff = record['coefficients']
        p_coeff = record['p_coefficients']

        if stype == 'sp':
            # s component
            shells.append({
                'center':       center.copy(),
                'angular':      (0, 0, 0),
                'exponents':    exps,
                'coefficients': s_coeff,
            })
            # three p components
            for ang in _ANG_COMPONENTS[1]:
                shells.append({
                    'center':       center.copy(),
                    'angular':      ang,
                    'exponents':    exps,
                    'coefficients': p_coeff,
                })
        else:
            l = _SHELL_TYPE_TO_L[stype]
            for ang in _ANG_COMPONENTS[l]:
                shells.append({
                    'center':       center.copy(),
                    'angular':      ang,
                    'exponents':    exps,
                    'coefficients': s_coeff,
                })

    return shells


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_basis(
    atoms: Sequence[tuple[str, np.ndarray]],
    basis_name: str = 'sto-3g',
    angstrom: bool = False,
) -> list[dict]:
    """
    Build the AO basis for a molecule.

    Parameters
    ----------
    atoms : sequence of (element_symbol, center) pairs
        element_symbol : str   e.g. 'C', 'H', 'O'
        center         : array-like, shape (3,)  — position in bohr
                         (or Ångström if angstrom=True)
    basis_name : str
        Name of the basis set (case-insensitive).  Must have been registered
        via register_basis() or load_nwchem().  Default: 'sto-3g'.
    angstrom : bool
        If True, input coordinates are assumed to be in Ångström and are
        converted to bohr internally.  Default: False (bohr).

    Returns
    -------
    list[dict]
        Ordered list of shell dicts, one per AO basis function, in the
        format expected by qchem.integrals.  Ordering: atom-by-atom,
        shell-by-shell (inner to outer), component-by-component (x, y, z
        for p-shells; xx, yy, zz, xy, xz, yz for d-shells).

    Raises
    ------
    KeyError
        If the basis set is not registered or an element is missing from it.

    Examples
    --------
    >>> import numpy as np
    >>> from qchem.basis import build_basis, angstrom_to_bohr
    >>>
    >>> # H₂ at the experimental bond length (0.74 Å), geometry in bohr
    >>> R = 0.74 * 1.8897259886 / 2          # half bond length
    >>> atoms = [('H', np.array([0., 0., -R])),
    ...          ('H', np.array([0., 0.,  R]))]
    >>> basis = build_basis(atoms, 'sto-3g')
    >>> len(basis)      # 1 s-function per H atom
    2
    >>>
    >>> # Water in Ångström, let build_basis convert
    >>> atoms_ang = [
    ...     ('O', np.array([0.000,  0.000,  0.117])),
    ...     ('H', np.array([0.000,  0.757, -0.471])),
    ...     ('H', np.array([0.000, -0.757, -0.471])),
    ... ]
    >>> basis = build_basis(atoms_ang, 'sto-3g', angstrom=True)
    >>> len(basis)      # O: 1s + 2sp(→1s+3p) = 5;  2×H: 1s each → 7 total
    7
    """
    key = basis_name.lower()
    shells: list[dict] = []
    for sym, raw_center in atoms:
        center = np.asarray(raw_center, dtype=float)
        if angstrom:
            center = angstrom_to_bohr(center)
        shells.extend(_shells_for_atom(sym, key, center))
    return shells


def basis_info(basis_name: str = 'sto-3g') -> dict[str, int]:
    """
    Return a summary of which elements are available in a registered basis.

    Returns
    -------
    dict mapping element symbol → number of AO basis functions (at a dummy
    centre) that the basis generates for that element.

    Useful for quick sanity checks before constructing a full molecule.
    """
    key = basis_name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Basis set {basis_name!r} is not registered.")
    dummy = np.zeros(3)
    return {
        sym: len(_shells_for_atom(sym, key, dummy))
        for sym in sorted(_REGISTRY[key].keys(), key=lambda s: ATOMIC_NUMBER[s])
    }