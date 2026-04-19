# qchem-core

![tests](https://github.com/jaxon-smith-8225/qchem_core/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/jaxon-smith-8225/qchem_core/branch/main/graph/badge.svg)](https://codecov.io/gh/your-actual-username/qchem_core)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

A ground-up implementation of Hartree–Fock and Kohn–Sham DFT in Python,
built from the integral primitives up. No external quantum-chemistry
dependencies — just NumPy and SciPy

## What's inside

- **Integrals** — Obara–Saika recursion for overlap, kinetic, nuclear-attraction,
  and two-electron repulsion integrals over contracted Cartesian Gaussians;
  Boys function via series + asymptotic expansions.
- **Basis sets** — Built-in STO-3G (H–Ne), plus a parser for Basis Set Exchange
  NWChem-format strings.
- **SCF** — Restricted Hartree–Fock with DIIS acceleration.
- **DFT** — Kohn–Sham with Becke-partitioned atom-centered grids, Lebedev
  angular quadrature, LDA (Dirac–Slater + VWN) and GGA (PBE) functionals.

## Installation

```bash
git clone https://github.com/jaxon-smith-8225/qchem_core.git
cd qchem_core
pip install -e ".[dev]"
```

## Quickstart

```python
from qchem import Molecule
from qchem.scf.hartree_fock import rhf
from qchem.dft.ks import ks

# Water molecule, geometry in Ångström
h2o = Molecule([('O',  [0.000,  0.000,  0.117]),
                ('H',  [0.000,  0.755, -0.468]),
                ('H',  [0.000, -0.755, -0.468])])

# Hartree–Fock
hf_result = rhf(h2o, basis_name="sto-3g")
print(f"HF energy:  {hf_result.homo_energy:.6f} Ha")

# Kohn–Sham DFT with PBE
ks_result = ks(h2o, basis_name="sto-3g", functional="pbe")
print(f"KS-PBE energy: {ks_result.homo_energy:.6f} Ha")
```

## Running tests

```bash
pytest                      # full suite
pytest -m "not slow"        # skip slow tests
pytest --cov=qchem          # with coverage report
```

## Project status

Pedagogical / research-grade. Correctness validated against reference values
for small molecules (see `qchem/*/tests/`). Not optimized for production use —
PySCF, Psi4, and ORCA exist for that.

## License

MIT
