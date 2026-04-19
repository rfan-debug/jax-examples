# JAX-QC: A Functional Quantum Chemistry Framework in JAX

## Implementation Plan for Coding Model — Revision 2

---

## 1. Project Overview

### 1.1 Vision

Build a JAX-native quantum chemistry framework where every computational layer is organized by its functional programming (FP) abstraction — Functor, Applicative, Monad, Foldable — making parallelism boundaries explicit and enabling JAX's XLA compiler to maximize hardware utilization.

### 1.2 Design Principles

1. **FP-first architecture**: Every module explicitly declares whether it is Applicative (parallelizable) or Monadic (sequential). This drives API design, not just comments.
2. **JAX-native**: All numerics in JAX. No NumPy fallbacks in hot paths. All core functions must be `jit`-compatible.
3. **Immutable state**: All data structures are immutable (`chex.dataclass(frozen=True)` for complex types, `NamedTuple` for simple leaf types). SCF state transitions are explicit `State -> State` functions.
4. **Separation of description and execution**: Computation graphs are built as data (like Jaxpr), then handed to an executor. This enables swapping backends, logging, profiling.
5. **Composability**: `vmap(grad(energy_fn))` must work out of the box for PES scans, frequency calculations, and forces.

### 1.3 Why `chex.dataclass(frozen=True)` over raw NamedTuple

Both enforce immutability and are JAX pytree-compatible, but `chex.dataclass` provides better ergonomics for complex, nested types:

| Feature | NamedTuple | chex.dataclass(frozen=True) |
|---------|------------|---------------------------|
| Immutable | Yes (no `__setattr__`) | Yes (frozen) |
| JAX pytree | Automatic | Automatic (chex registers) |
| `.replace()` | `._replace()` (underscore, legacy) | `.replace()` (clean API) |
| Default values | Python 3.6.1+ | Standard dataclass syntax |
| Nested types | Awkward (tuple of tuples) | Natural (dataclass of dataclass) |
| Type checking | Weak (positional) | Strong (keyword + type hints) |
| IDE support | Limited | Full autocomplete + type inference |

**Rule of thumb**: Use `NamedTuple` for simple leaf types with 2-3 fields (e.g., `Primitive`). Use `chex.dataclass` for anything with >3 fields, optional fields, or nesting (e.g., `SCFState`, `CalcConfig`, `Molecule`).

**Why not plain `@dataclass`?** Plain dataclasses are mutable by default. Even `@dataclass(frozen=True)` requires manual pytree registration (`jax.tree_util.register_dataclass`). `chex.dataclass` handles both automatically and adds JAX-specific utilities like `jax.tree_util.tree_map` support out of the box.

### 1.4 Scope

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | RHF + UHF SCF with standard basis sets | This document |
| Phase 2 | DFT (LDA/GGA), integral-direct mode | Interface contracts only |
| Phase 3 | MP2, CCSD, CCSD(T) | Interface contracts only |
| Phase 4 | Geometry optimization + frequency analysis | Interface contracts only |
| Phase 5 | Periodic systems / plane waves | Future |

---

## 2. Project Structure

```
jax_qc/
├── README.md
├── pyproject.toml
├── jax_qc/
│   ├── __init__.py
│   │
│   ├── core/                       # Pure data types, no computation
│   │   ├── __init__.py
│   │   ├── types.py                # Molecule, Shell, SCFState, etc.
│   │   └── constants.py            # Physical constants, unit conversions
│   │
│   ├── basis/                      # Basis set acquisition & parsing (Pure, Applicative)
│   │   ├── __init__.py
│   │   ├── bse_fetch.py            # Fetch from Basis Set Exchange (via basis_set_exchange pkg)
│   │   ├── parse.py                # Parse BSE dict -> internal Shell/BasisSet types
│   │   ├── normalize.py            # Normalization factors for Gaussians
│   │   ├── cache.py                # Local JSON cache for fetched basis sets
│   │   └── build.py                # Molecule + basis name -> BasisSet
│   │
│   ├── integrals/                  # Molecular integrals (Applicative)
│   │   ├── __init__.py
│   │   ├── overlap.py              # S matrix
│   │   ├── kinetic.py              # T matrix
│   │   ├── nuclear.py              # V matrix (nuclear attraction)
│   │   ├── eri.py                  # Two-electron integrals (μν|λσ)
│   │   ├── boys.py                 # Boys function F_n(t)
│   │   ├── gaussian_product.py     # Gaussian product theorem utilities
│   │   ├── obara_saika.py          # Obara-Saika recurrence for general angular momentum
│   │   ├── screening.py            # Schwarz screening (filter, CPU-side)
│   │   └── interface.py            # compute_integrals() top-level
│   │
│   ├── scf/                        # Self-consistent field (Monad)
│   │   ├── __init__.py
│   │   ├── fock.py                 # Build Fock matrix F = H + G(D) (Applicative internals)
│   │   ├── density.py              # Density matrix from MO coefficients
│   │   ├── energy.py               # Electronic + nuclear energy (Foldable)
│   │   ├── diis.py                 # DIIS convergence accelerator
│   │   ├── damping.py              # Simple mixing / level shifting
│   │   ├── guess.py                # Initial guess strategies (core, SAD, GWH)
│   │   ├── orthogonalize.py        # S^{-1/2} and canonical orthogonalization
│   │   ├── rhf.py                  # Restricted Hartree-Fock driver
│   │   ├── uhf.py                  # Unrestricted Hartree-Fock driver
│   │   └── interface.py            # run_scf() top-level
│   │
│   ├── properties/                 # Post-SCF analysis (Foldable / Applicative)
│   │   ├── __init__.py
│   │   ├── mulliken.py             # Mulliken population analysis
│   │   ├── dipole.py               # Dipole moment
│   │   └── orbital_analysis.py     # Orbital energies, HOMO-LUMO gap
│   │
│   ├── grad/                       # Analytic gradients (Adjunction: JVP ⊣ VJP)
│   │   ├── __init__.py
│   │   ├── rhf_grad.py             # RHF analytic gradient via jax.grad
│   │   ├── numerical_grad.py       # Finite difference gradient (validation)
│   │   └── interface.py            # compute_gradient() top-level
│   │
│   ├── geomopt/                    # Geometry optimization (Monad over Monad)
│   │   ├── __init__.py
│   │   ├── optimizer.py            # BFGS / L-BFGS
│   │   ├── hessian.py              # Hessian via jax.hessian or finite diff
│   │   ├── constraints.py          # Frozen atoms, fixed bonds
│   │   └── interface.py            # optimize_geometry() top-level
│   │
│   ├── profiling/                  # Stage-level profiling
│   │   ├── __init__.py
│   │   ├── timer.py                # Hierarchical timer context manager
│   │   ├── report.py               # Pretty-print + export profiling data
│   │   └── jax_profiler.py         # JAX-specific profiling (device time, compilation)
│   │
│   ├── io/                         # Input/Output (side effects, isolated)
│   │   ├── __init__.py
│   │   ├── input_parser.py         # Parse YAML/dict input specification
│   │   ├── xyz.py                  # Read/write XYZ format
│   │   ├── molden.py               # Write Molden format for visualization
│   │   └── output.py               # Formatted output / logging
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── units.py                # Bohr/Angstrom, Hartree/eV/kcal conversions
│       └── symmetry.py             # Integral symmetry index mapping
│
├── tests/
│   ├── test_integrals.py
│   ├── test_scf.py
│   ├── test_uhf.py
│   ├── test_gradient.py
│   ├── test_basis.py
│   ├── test_input.py
│   ├── test_profiling.py
│   ├── test_fp_laws.py             # Functor/Applicative law verification
│   └── conftest.py                 # Shared fixtures (molecules, basis sets, PySCF references)
│
├── benchmarks/                     # Comprehensive benchmark suite
│   ├── README.md
│   ├── generate_references.py      # Generate PySCF reference data
│   ├── run_benchmarks.py           # Run full benchmark + comparison
│   ├── molecules/                  # Benchmark molecule definitions
│   │   ├── closed_shell.py         # H₂, H₂O, NH₃, CH₄, C₂H₂, benzene
│   │   ├── open_shell.py           # O₂(triplet), NO, CH₃, OH, Li atom
│   │   ├── charged.py              # HeH⁺, NH₄⁺, OH⁻, NO⁺
│   │   ├── transition_metals.py    # (Phase 2+: ScH, TiO)
│   │   └── pathological.py         # Near-degenerate, convergence-difficult cases
│   └── reference_data/             # PySCF-generated reference values
│       ├── rhf_sto3g.json
│       ├── rhf_631g.json
│       ├── rhf_ccpvdz.json
│       ├── uhf_sto3g.json
│       └── uhf_631g.json
│
└── examples/
    ├── 01_h2_energy.py
    ├── 02_water_scf.py
    ├── 03_open_shell_o2.py
    ├── 04_pes_scan.py
    ├── 05_forces.py
    ├── 06_vmap_molecules.py
    ├── 07_geom_opt.py
    └── 08_profiling.py
```

---

## 3. Core Data Types (`jax_qc/core/types.py`)

```python
"""
Core data types. All immutable, all pytree-compatible.

Rule: chex.dataclass(frozen=True) for complex types (>3 fields, optional, nested).
      NamedTuple for simple leaf types (2-3 fields, no optionals).
      Computation lives in the module that owns the relevant FP abstraction,
      never as methods on data types.
"""

import chex
import jax.numpy as jnp
from typing import NamedTuple, Optional

# ─── Simple leaf types (NamedTuple: lightweight, auto-pytree) ───

class Primitive(NamedTuple):
    """Single primitive Gaussian: N * exp(-alpha * |r - center|^2)"""
    exponent: float
    coeff: float              # contraction coefficient * normalization
    center: jnp.ndarray       # (3,)


# ─── Complex types (chex.dataclass: rich, frozen, pytree-registered) ───

@chex.dataclass(frozen=True)
class Shell:
    """A contracted shell (e.g., one s-shell or one p-shell on an atom).
    angular_momentum: 0=s, 1=p, 2=d, 3=f
    """
    angular_momentum: int
    exponents: jnp.ndarray         # (n_primitives,)
    coefficients: jnp.ndarray      # (n_primitives,) or (n_prims, n_general)
    center: jnp.ndarray            # (3,)
    atom_index: int


@chex.dataclass(frozen=True)
class BasisSet:
    """Complete basis set for a molecule."""
    shells: list                    # List[Shell]
    n_basis: int
    shell_to_basis: list            # shell index -> [basis fn indices]
    basis_to_atom: jnp.ndarray     # (n_basis,) -> atom index


@chex.dataclass(frozen=True)
class Molecule:
    """Molecular system specification."""
    coords: jnp.ndarray            # (n_atoms, 3), Bohr
    atomic_numbers: jnp.ndarray    # (n_atoms,), int
    elements: tuple                 # ('O', 'H', 'H') — immutable
    charge: int = 0
    spin: int = 0                   # 2S (0=singlet, 1=doublet, 2=triplet)
    n_electrons: int = 0
    n_alpha: int = 0
    n_beta: int = 0


@chex.dataclass(frozen=True)
class IntegralSet:
    """All molecular integrals. Output of the Applicative integral layer.
    All fields are independent — computed in parallel, consumed by SCF.
    """
    S: jnp.ndarray                  # overlap (N, N)
    T: jnp.ndarray                  # kinetic (N, N)
    V: jnp.ndarray                  # nuclear attraction (N, N)
    H_core: jnp.ndarray            # T + V (N, N)
    ERI: jnp.ndarray               # two-electron (N, N, N, N)
    E_nuc: float                    # nuclear repulsion energy


@chex.dataclass(frozen=True)
class SCFState:
    """Immutable SCF iteration state.
    This is the 's' in State Monad: step :: s -> (s, a).
    """
    density: jnp.ndarray
    fock: jnp.ndarray
    coefficients: jnp.ndarray
    orbital_energies: jnp.ndarray
    energy: float
    iteration: int


@chex.dataclass(frozen=True)
class SCFResult:
    """Final result after SCF convergence."""
    converged: bool
    state: SCFState
    energy: float
    E_elec: float
    E_nuc: float
    n_iterations: int
    S: jnp.ndarray
    H_core: jnp.ndarray
    ERI: Optional[jnp.ndarray] = None   # None if integral-direct


@chex.dataclass(frozen=True)
class CalcConfig:
    """Calculation configuration — user-facing input."""
    method: str = 'rhf'             # 'rhf', 'uhf', 'mp2', 'ccsd', 'ccsd(t)'
    basis: str = 'sto-3g'           # any name recognized by Basis Set Exchange
    task: str = 'energy'            # 'energy', 'gradient', 'optimize', 'frequencies'
    max_scf_iter: int = 128
    scf_conv: float = 1e-10
    diis_space: int = 8
    damping: float = 0.0
    guess: str = 'core'             # 'core', 'sad', 'gwh', 'huckel'
    verbose: int = 1
    profile: bool = False           # enable stage profiling
```

---

## 4. Basis Set Integration (`jax_qc/basis/`)

### 4.1 Basis Set Exchange Integration

Use the official `basis_set_exchange` Python package (MolSSI, BSD-3 license) as the single source of truth for basis set data. No bundled basis set files — fetch from BSE, cache locally.

**File: `basis/bse_fetch.py`**

```python
"""
Fetch basis sets from Basis Set Exchange.

The BSE library ships a local copy of all basis set data (~37 MB),
so no network access is needed after install. This is NOT a web API call.

Install: pip install basis-set-exchange
"""

import basis_set_exchange as bse

def fetch_basis(basis_name: str, elements: list[int]) -> dict:
    """Fetch basis set data from BSE as a Python dict.
    
    FP: Pure function (BSE library is a local database lookup, no IO).
    
    Args:
        basis_name: e.g. 'sto-3g', '6-31G*', 'cc-pVDZ', 'aug-cc-pVTZ'
                    Case-insensitive. BSE handles name normalization.
        elements:   List of atomic numbers, e.g. [1, 8] for H and O
    
    Returns:
        BSE dict with structure:
        {
            'name': '6-31G*',
            'elements': {
                '1': {'electron_shells': [
                    {'angular_momentum': [0],
                     'exponents': ['18.7311370', ...],
                     'coefficients': [['0.03349460', ...]]}
                ]},
                '8': {'electron_shells': [...]}
            }
        }
    """
    return bse.get_basis(
        basis_name,
        elements=elements,
        uncontract_spdf=True,  # split sp-shells into separate s and p
        header=False,
    )

def list_available_bases(elements: list[int] = None) -> list[str]:
    """List all basis sets available for given elements."""
    all_bases = bse.get_all_basis_names()
    if elements is None:
        return all_bases
    return [name for name in all_bases
            if all(str(z) in bse.get_basis(name)['elements'] for z in elements)]

def get_basis_info(basis_name: str) -> dict:
    """Get metadata about a basis set (description, family, role)."""
    return {
        'family': bse.get_basis_family(basis_name),
        'description': bse.get_basis_notes(basis_name),
        'versions': list(bse.get_basis_versions(basis_name)),
    }
```

**File: `basis/parse.py`**

```python
"""
Parse BSE dictionary format into internal Shell/BasisSet types.

BSE returns basis sets as nested dicts with string exponents/coefficients.
This module converts them to JAX-compatible numeric types.
"""

def bse_dict_to_shells(bse_data: dict, mol: Molecule) -> list[Shell]:
    """Convert BSE dict -> list of Shell objects.
    
    FP: Pure function.
    
    Steps:
    1. For each atom in molecule, look up element in bse_data
    2. For each electron_shell, extract angular momentum, exponents, coefficients
    3. Parse string values to float64
    4. Apply normalization factors
    5. Create Shell objects with atom coordinates
    
    Handles:
    - General contractions (multiple coefficient vectors per shell)
    - sp-shells (should be split by uncontract_spdf=True in fetch)
    - Cartesian vs spherical harmonics (default: spherical for l >= 2)
    """
    ...

def build_basis_set(mol: Molecule, basis_name: str) -> BasisSet:
    """Top-level: molecule + basis name -> complete BasisSet.
    
    This is the main entry point for basis set construction.
    Fetches from BSE, parses, normalizes, builds index mappings.
    """
    elements = list(set(mol.atomic_numbers.tolist()))
    bse_data = fetch_basis(basis_name, elements)
    shells = bse_dict_to_shells(bse_data, mol)
    n_basis = sum(2 * s.angular_momentum + 1 for s in shells)
    # Build shell_to_basis and basis_to_atom mappings
    ...
    return BasisSet(shells=shells, n_basis=n_basis, 
                    shell_to_basis=s2b, basis_to_atom=b2a)
```

**File: `basis/cache.py`**

```python
"""
Local cache for parsed basis sets.

First call to build_basis_set('cc-pVTZ', ...) fetches and parses from BSE.
Subsequent calls with same (basis_name, elements) hit the cache.

Cache is stored as JSON in ~/.jax_qc/basis_cache/
"""

import json, hashlib, pathlib

CACHE_DIR = pathlib.Path.home() / '.jax_qc' / 'basis_cache'

def cache_key(basis_name: str, elements: list[int]) -> str:
    """Deterministic cache key from basis name + element list."""
    raw = f"{basis_name.lower()}|{sorted(elements)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def get_cached(basis_name, elements):
    """Return cached dict or None."""
    ...

def put_cache(basis_name, elements, bse_data):
    """Write parsed data to cache."""
    ...
```

### 4.2 Supported Basis Sets

Since we use BSE, the framework supports **any basis set in the BSE database** (~600 basis sets). Phase 1 testing focuses on:

| Basis | Shells | Purpose |
|-------|--------|---------|
| STO-3G | s (H-Ar), sp (Li-Ar) | Minimal, fast testing |
| 3-21G | s, sp | Budget split-valence |
| 6-31G | s, sp | Standard Pople |
| 6-31G* / 6-31G(d) | s, sp, d | Polarization |
| 6-31G** / 6-31G(d,p) | s, sp, d, p(H) | Full polarization |
| 6-311G(d,p) | s, sp, d | Triple-zeta Pople |
| cc-pVDZ | s, p, d | Correlation-consistent DZ |
| cc-pVTZ | s, p, d, f | Correlation-consistent TZ |
| aug-cc-pVDZ | s, p, d (+ diffuse) | Anions, weak interactions |
| def2-SVP | s, p, d | Ahlrichs split-valence |
| def2-TZVP | s, p, d, f | Ahlrichs triple-zeta |

---

## 5. Profiling System (`jax_qc/profiling/`)

### 5.1 Design: Hierarchical Stage Timer

The profiler tracks wall-clock time, JAX device time, and memory per computation stage. It mirrors the FP abstraction hierarchy:

```
Total Calculation
├── Basis Construction              [Pure]
├── Integral Computation            [Applicative]
│   ├── Overlap (S)
│   ├── Kinetic (T)
│   ├── Nuclear Attraction (V)
│   ├── ERI (two-electron)
│   │   ├── Schwarz screening       [Filter/CPU]
│   │   └── Integral evaluation      [Applicative/GPU]
│   └── Nuclear Repulsion
├── SCF Iteration                    [Monad]
│   ├── Initial Guess
│   ├── Iteration 1
│   │   ├── Fock Build               [Applicative]
│   │   │   ├── Coulomb (J)          [einsum]
│   │   │   └── Exchange (K)         [einsum]
│   │   ├── DIIS Extrapolation       [small linalg]
│   │   ├── Basis Transform          [Functor: matmul]
│   │   ├── Diagonalization          [eigh]
│   │   ├── Density Matrix           [einsum]
│   │   └── Energy Evaluation        [Foldable: trace]
│   ├── Iteration 2 ...
│   └── Iteration N (converged)
├── Properties                       [Foldable]
│   ├── Mulliken Population
│   ├── Dipole Moment
│   └── Orbital Analysis
└── Gradient (if requested)          [Adjunction: VJP]
```

### 5.2 Implementation

**File: `profiling/timer.py`**

```python
"""
Hierarchical timer with JAX-aware device synchronization.

Usage:
    timer = StageTimer()
    
    with timer.stage('integrals'):
        with timer.stage('overlap'):
            S = compute_overlap(basis)
        with timer.stage('eri'):
            ERI = compute_eri(basis)
    
    with timer.stage('scf'):
        for i in range(n_iter):
            with timer.stage(f'iter_{i}'):
                with timer.stage('fock_build'):
                    F = build_fock(D, H, ERI)
                ...
    
    timer.report()
"""

import time
import jax
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass
class StageRecord:
    name: str
    wall_time: float = 0.0
    call_count: int = 0
    children: dict = field(default_factory=dict)
    parent: 'StageRecord | None' = None
    fp_abstraction: str = ''         # 'applicative', 'monad', 'foldable', etc.
    
class StageTimer:
    """Hierarchical profiling timer.
    
    Features:
    - Nested stages via context manager
    - JAX device synchronization (jax.block_until_ready) for accurate GPU timing
    - Aggregation of repeated stages (e.g., SCF iterations)
    - FP abstraction annotation per stage
    - Export to dict/JSON for analysis
    """
    
    def __init__(self, sync_device=True):
        self.root = StageRecord(name='total')
        self._stack = [self.root]
        self._sync = sync_device
    
    @contextmanager
    def stage(self, name: str, fp_type: str = ''):
        """Time a named computation stage.
        
        Args:
            name: stage name (e.g., 'fock_build')
            fp_type: FP abstraction ('applicative', 'monad', 'foldable', etc.)
        """
        parent = self._stack[-1]
        if name not in parent.children:
            parent.children[name] = StageRecord(name=name, parent=parent, 
                                                 fp_abstraction=fp_type)
        record = parent.children[name]
        
        if self._sync:
            jax.effects_barrier()   # ensure previous GPU work is done
        
        t0 = time.perf_counter()
        self._stack.append(record)
        try:
            yield record
        finally:
            if self._sync:
                jax.effects_barrier()
            record.wall_time += time.perf_counter() - t0
            record.call_count += 1
            self._stack.pop()
    
    def report(self, min_percent=1.0) -> str:
        """Pretty-print hierarchical timing report."""
        ...
    
    def to_dict(self) -> dict:
        """Export as nested dict for JSON serialization."""
        ...
    
    def summary_table(self) -> str:
        """Flat summary grouped by FP abstraction type."""
        ...
```

**File: `profiling/report.py`**

```python
"""
Pretty-print profiling reports.

Example output:

┌─────────────────────────────────────────────────────────────────────────────┐
│ JAX-QC Profiling Report: H₂O / RHF / 6-31G*                              │
├──────────────────────────────┬───────────┬────────┬────────┬──────────────┤
│ Stage                        │ Wall (ms) │   %    │ Calls  │ FP Type      │
├──────────────────────────────┼───────────┼────────┼────────┼──────────────┤
│ Total                        │   1523.4  │ 100.0  │      1 │              │
│ ├─ Basis construction        │      3.2  │   0.2  │      1 │ Pure         │
│ ├─ Integral computation      │    892.1  │  58.6  │      1 │ Applicative  │
│ │  ├─ Overlap (S)            │      1.2  │   0.1  │      1 │ Applicative  │
│ │  ├─ Kinetic (T)            │      1.4  │   0.1  │      1 │ Applicative  │
│ │  ├─ Nuclear attraction (V) │      3.8  │   0.2  │      1 │ Applicative  │
│ │  └─ ERI                    │    885.0  │  58.1  │      1 │ Applicative  │
│ ├─ SCF iteration             │    624.7  │  41.0  │      1 │ Monad        │
│ │  ├─ Fock build (total)     │    412.3  │  27.1  │     12 │ Applicative  │
│ │  ├─ DIIS                   │      8.4  │   0.6  │     12 │ Writer+State │
│ │  ├─ Diagonalization        │     18.2  │   1.2  │     12 │ Pure         │
│ │  ├─ Density matrix         │      3.1  │   0.2  │     12 │ Functor      │
│ │  └─ Energy evaluation      │      2.8  │   0.2  │     12 │ Foldable     │
│ └─ Properties                │      3.4  │   0.2  │      1 │ Foldable     │
├──────────────────────────────┼───────────┼────────┼────────┼──────────────┤
│ FP Type Summary:                                                          │
│   Applicative (parallelizable):  1300.2 ms (85.3%)                        │
│   Monad (sequential overhead):    204.3 ms (13.4%)                        │
│   Foldable / Pure:                 18.9 ms ( 1.2%)                        │
│                                                                           │
│   Implication: 85% of compute is Applicative → GPU acceleration viable    │
└─────────────────────────────────────────────────────────────────────────────┘
"""
```

**File: `profiling/jax_profiler.py`**

```python
"""
JAX-specific profiling: JIT compilation time, XLA device time, memory.

Uses jax.profiler and jax.make_jaxpr for deeper analysis.
"""

def profile_jit_compilation(fn, *args):
    """Measure JIT compilation time vs execution time.
    
    Returns:
        {'compile_time_ms': ..., 'first_run_ms': ..., 'subsequent_run_ms': ...,
         'jaxpr_nodes': ..., 'xla_hlo_ops': ...}
    """
    ...

def memory_profile(fn, *args):
    """Track peak memory usage during computation.
    
    Returns:
        {'peak_host_mb': ..., 'peak_device_mb': ..., 
         'eri_size_mb': ..., 'density_size_mb': ...}
    """
    ...

def flop_estimate(integrals: IntegralSet, n_scf_steps: int) -> dict:
    """Estimate FLOPs for each computation stage.
    
    Returns:
        {'eri_flops': N^4 * K, 'fock_build_flops': N^4 * n_steps, 
         'diag_flops': N^3 * n_steps, 'arithmetic_intensity': ...}
    """
    ...
```

### 5.3 Integration into SCF Driver

```python
def run_rhf(integrals, mol, basis, config):
    timer = StageTimer(sync_device=True) if config.profile else None
    
    with optional_stage(timer, 'orthogonalization', 'pure'):
        X = symmetric_orthogonalization(integrals.S)
    
    with optional_stage(timer, 'initial_guess', 'pure'):
        D = initial_guess(integrals, config.guess, ...)
    
    with optional_stage(timer, 'scf_iteration', 'monad'):
        for iteration in range(config.max_scf_iter):
            with optional_stage(timer, f'fock_build', 'applicative'):
                F = build_fock(D, integrals.H_core, integrals.ERI)
            with optional_stage(timer, f'diis', 'writer+state'):
                F = diis_extrapolate(...)
            with optional_stage(timer, f'diagonalization', 'pure'):
                eps, C_prime = jnp.linalg.eigh(X.T @ F @ X)
            ...
    
    if timer:
        print(timer.report())
    
    return SCFResult(...)

# Helper to make profiling optional without cluttering code
@contextmanager
def optional_stage(timer, name, fp_type=''):
    if timer is not None:
        with timer.stage(name, fp_type):
            yield
    else:
        yield
```

---

## 6. Benchmark Suite (`benchmarks/`)

### 6.1 Benchmark Molecules — Comprehensive Coverage

**Tier 1: Minimal validation (s-type only, Phase 1a)**

| Molecule | Atoms | Electrons | Basis | Spin | Charge | Tests |
|----------|-------|-----------|-------|------|--------|-------|
| H₂ | 2 | 2 | STO-3G | 0 | 0 | Basic RHF, PES scan |
| HeH⁺ | 2 | 2 | STO-3G | 0 | +1 | Charged, heteronuclear |
| He | 1 | 2 | STO-3G | 0 | 0 | Single atom |
| H₃⁺ | 3 | 2 | STO-3G | 0 | +1 | 3-center system |

**Tier 2: Standard closed-shell (s+p, Phase 1b)**

| Molecule | Atoms | Electrons | Basis | Spin | Charge | Tests |
|----------|-------|-----------|-------|------|--------|-------|
| H₂O | 3 | 10 | STO-3G, 6-31G* | 0 | 0 | Standard benchmark |
| NH₃ | 4 | 10 | STO-3G, 6-31G* | 0 | 0 | Pyramidal geometry |
| CH₄ | 5 | 10 | STO-3G, 6-31G* | 0 | 0 | Tetrahedral symmetry |
| HF | 2 | 10 | STO-3G, cc-pVDZ | 0 | 0 | Polar, strong electronegativity |
| CO | 2 | 14 | STO-3G, 6-31G | 0 | 0 | Triple bond |
| N₂ | 2 | 14 | STO-3G, cc-pVDZ | 0 | 0 | Strong triple bond |
| C₂H₂ | 4 | 14 | STO-3G, 6-31G | 0 | 0 | Linear molecule |
| H₂CO | 4 | 16 | STO-3G, 6-31G* | 0 | 0 | Aldehyde, planar |
| C₂H₄ | 6 | 16 | STO-3G, 6-31G | 0 | 0 | Ethylene, π bond |

**Tier 3: Open-shell UHF (tests spin-unrestricted code)**

| Molecule | Atoms | Electrons | Basis | 2S | Charge | Notes |
|----------|-------|-----------|-------|-----|--------|-------|
| H atom | 1 | 1 | STO-3G, cc-pVDZ | 1 | 0 | Simplest open-shell |
| Li atom | 1 | 3 | STO-3G, 6-31G | 1 | 0 | 1s²2s¹ |
| B atom | 1 | 5 | STO-3G | 1 | 0 | 2p¹ |
| O atom | 1 | 8 | STO-3G, 6-31G | 2 | 0 | Triplet ground state |
| CH₃ (methyl) | 4 | 9 | STO-3G, 6-31G | 1 | 0 | Planar radical |
| OH | 2 | 9 | STO-3G, 6-31G | 1 | 0 | Hydroxyl radical |
| NO | 2 | 15 | STO-3G, 6-31G | 1 | 0 | Odd-electron, 2Π |
| O₂ | 2 | 16 | STO-3G, 6-31G | 2 | 0 | Triplet, classic UHF test |
| NO₂ | 3 | 23 | STO-3G | 1 | 0 | Bent radical |

**Tier 4: Charged species**

| Molecule | Atoms | Electrons | Basis | Spin | Charge | Notes |
|----------|-------|-----------|-------|------|--------|-------|
| HeH⁺ | 2 | 2 | STO-3G | 0 | +1 | Simplest cation |
| Li⁺ | 1 | 2 | STO-3G | 0 | +1 | Closed-shell cation |
| NH₄⁺ | 5 | 10 | STO-3G, 6-31G | 0 | +1 | Tetrahedral cation |
| OH⁻ | 2 | 10 | STO-3G, 6-31G | 0 | -1 | Anion |
| F⁻ | 1 | 10 | STO-3G, aug-cc-pVDZ | 0 | -1 | Anion, needs diffuse |
| NO⁺ | 2 | 14 | STO-3G, 6-31G | 0 | +1 | Isoelectronic with N₂ |
| CN⁻ | 2 | 14 | STO-3G, 6-31G | 0 | -1 | Anion, triple bond |

**Tier 5: SCF convergence stress tests**

| Molecule | Challenge | Notes |
|----------|-----------|-------|
| Cr atom (Phase 2+) | Near-degeneracy | d-electron correlation |
| Fe(CO)₅ (Phase 2+) | Transition metal, many electrons | Tests ERI performance |
| Ozone O₃ | Multireference character | RHF may struggle |
| Stretched H₂ (R=5 bohr) | Near-degeneracy at dissociation | Tests DIIS robustness |
| Twisted C₂H₄ | Breaking π bond | UHF vs RHF instability |

### 6.2 Reference Data Generation

```python
# benchmarks/generate_references.py
"""
Generate reference values for all benchmark molecules using PySCF.

For each (molecule, basis, method) combination, stores:
- Total energy (to 1e-12 Hartree)
- Electronic energy
- Nuclear repulsion
- Orbital energies (all)
- Mulliken charges
- Dipole moment (x, y, z)
- Number of SCF iterations
- S, T, V matrix checksums (for integral validation)
- Individual integral matrix elements (first 10 diagonal elements of S, T, V)
"""

from pyscf import gto, scf
import json

def generate_rhf_reference(atoms_str, basis_name, charge=0, spin=0):
    mol = gto.M(atom=atoms_str, basis=basis_name, charge=charge, spin=spin, unit='bohr')
    
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    
    energy = mf.kernel()
    
    return {
        'atoms': atoms_str,
        'basis': basis_name,
        'charge': charge,
        'spin': spin,
        'method': 'rhf' if spin == 0 else 'uhf',
        'energy_total': float(energy),
        'energy_elec': float(mf.energy_elec()[0]),
        'energy_nuc': float(mf.energy_nuc()),
        'orbital_energies': mf.mo_energy.tolist() if isinstance(mf.mo_energy, np.ndarray) 
                           else [e.tolist() for e in mf.mo_energy],
        'converged': bool(mf.converged),
        'n_scf_iter': mf.iterations if hasattr(mf, 'iterations') else -1,
        'n_basis': mol.nao,
        'n_electrons': mol.nelectron,
        'mulliken_charges': mf.mulliken_pop(verbose=0)[1].tolist(),
        'dipole': mf.dip_moment(verbose=0).tolist(),
        # Integral validation (spot checks)
        'overlap_diag': mol.intor('int1e_ovlp').diagonal()[:10].tolist(),
        'kinetic_diag': mol.intor('int1e_kin').diagonal()[:10].tolist(),
    }

# Generate for all benchmark molecules
BENCHMARKS = [
    # Tier 1
    ('H 0 0 0; H 0 0 1.4', 'sto-3g', 0, 0),
    # Tier 2
    ('O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469', 'sto-3g', 0, 0),
    ('O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469', '6-31g*', 0, 0),
    # Tier 3 - open shell
    ('O 0 0 0; O 0 0 2.282', 'sto-3g', 0, 2),  # O₂ triplet
    ('N 0 0 0; O 0 0 2.175', 'sto-3g', 0, 1),   # NO doublet
    # ...
]
```

### 6.3 Benchmark Runner

```python
# benchmarks/run_benchmarks.py
"""
Run all benchmarks, compare against PySCF references, report accuracy.

Output:
┌──────────────────────────────────────────────────────────────────────┐
│ JAX-QC Benchmark Results                                            │
├───────────────┬──────────┬──────────┬────────────┬─────────┬────────┤
│ Molecule      │ Basis    │ Method   │ ΔE (μHa)   │ SCF Its │ Status │
├───────────────┼──────────┼──────────┼────────────┼─────────┼────────┤
│ H₂            │ STO-3G   │ RHF      │     0.001  │       2 │ PASS   │
│ H₂O           │ STO-3G   │ RHF      │     0.003  │      11 │ PASS   │
│ H₂O           │ 6-31G*   │ RHF      │     0.002  │      13 │ PASS   │
│ O₂ (triplet)  │ STO-3G   │ UHF      │     0.005  │      18 │ PASS   │
│ O₂ (triplet)  │ 6-31G    │ UHF      │     0.004  │      21 │ PASS   │
│ NO (doublet)  │ STO-3G   │ UHF      │     0.008  │      25 │ PASS   │
│ CH₃ (doublet) │ 6-31G    │ UHF      │     0.003  │      15 │ PASS   │
│ OH⁻           │ 6-31G    │ RHF      │     0.002  │      12 │ PASS   │
│ stretched H₂  │ STO-3G   │ UHF      │     0.012  │      35 │ PASS   │
├───────────────┴──────────┴──────────┴────────────┴─────────┴────────┤
│ Tolerance: 1.0 μHartree | 42/42 passed | 0 failed                   │
└──────────────────────────────────────────────────────────────────────┘
"""
```

---

## 7. Input Interface (`jax_qc/io/input_parser.py`)

### 7.1 User-facing API

```python
import jax_qc

# ─── Style 1: Minimal Python API ───
mol = jax_qc.Molecule(
    atoms=['O', 'H', 'H'],
    coords=[[0.000, 0.000, 0.117],
            [0.000, 0.757, -0.469],
            [0.000, -0.757, -0.469]],
    unit='angstrom',
)

result = jax_qc.energy(mol, method='rhf', basis='cc-pVDZ')
print(result.energy)

# ─── Style 2: Dictionary/YAML ───
result = jax_qc.run({
    'molecule': {
        'atoms': ['O', 'H', 'H'],
        'coords': [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
        'unit': 'angstrom',
    },
    'method': 'rhf',
    'basis': '6-31g*',        # any BSE name works
    'task': 'energy',
    'profile': True,           # enable stage profiling
})

# ─── Style 3: XYZ file ───
result = jax_qc.run_xyz('water.xyz', method='rhf', basis='cc-pvdz')

# ─── Style 4: Open-shell ───
result = jax_qc.energy(
    jax_qc.Molecule(atoms=['O', 'O'], coords=[[0,0,0], [0,0,2.282]],
                     unit='bohr', spin=2),   # triplet
    method='uhf', basis='6-31g',
)

# ─── Composable low-level API ───
mol = jax_qc.build_molecule(atoms=['H', 'H'], coords=[[0,0,0], [0,0,1.4]], unit='bohr')
basis = jax_qc.build_basis(mol, 'aug-cc-pVDZ')    # fetches from BSE
integrals = jax_qc.compute_integrals(mol, basis)   # Applicative
scf_result = jax_qc.run_scf(integrals, mol, basis) # Monad
forces = jax_qc.compute_gradient(scf_result, mol)  # Adjunction: VJP
```

---

## 8. Module Specifications

### 8.1 Integrals Module — APPLICATIVE

Specifications unchanged from v1 except:

- **Boys function**: Must support `F_n(t)` for n=0..8 (needed for (dd|dd) integrals).
- **Obara-Saika recurrence**: Required from Phase 1b for p-type and d-type orbitals. Implement for overlap, kinetic, nuclear. For ERI, implement Head-Gordon-Pople (HGP) variant.
- **All integral routines must accept `StageTimer`** as optional kwarg and report to it.

### 8.2 SCF Module — MONAD

Specifications unchanged from v1 except:

- **UHF**: Required from Phase 1 (not deferred). Needed for open-shell benchmarks.
- **Fock build UHF**: Two Fock matrices F_alpha, F_beta. Coulomb uses total density, Exchange uses per-spin density.
- **All SCF internals must report to `StageTimer`** if profiling is enabled.

### 8.3 Gradient Module — ADJUNCTION

Specifications unchanged from v1.

### 8.4 Geometry Optimization — MONAD over MONAD

Specifications unchanged from v1.

---

## 9. Implementation Order (Revised)

### Step 1: Core types + BSE integration (Day 1)

```
- [ ] core/types.py — chex.dataclass types
- [ ] core/constants.py — physical constants
- [ ] basis/bse_fetch.py — fetch from basis_set_exchange
- [ ] basis/parse.py — BSE dict -> Shell/BasisSet
- [ ] basis/normalize.py — Gaussian normalization
- [ ] basis/cache.py — local caching
- [ ] basis/build.py — build_basis() top-level
- [ ] io/xyz.py — read XYZ files
- [ ] profiling/timer.py — StageTimer
- [ ] profiling/report.py — pretty-print
- [ ] tests/test_basis.py
```

### Step 2: s-type integrals + profiling (Day 2)

```
- [ ] integrals/boys.py — F_0(t)
- [ ] integrals/gaussian_product.py
- [ ] integrals/overlap.py — s-type
- [ ] integrals/kinetic.py — s-type
- [ ] integrals/nuclear.py — s-type
- [ ] integrals/eri.py — s-type with 8-fold symmetry
- [ ] integrals/interface.py — compute_integrals() with profiling hooks
- [ ] tests/test_integrals.py — compare against PySCF for H₂, He, HeH⁺
```

### Step 3: RHF SCF + benchmark Tier 1 (Day 3-4)

```
- [ ] scf/orthogonalize.py
- [ ] scf/guess.py — core Hamiltonian guess
- [ ] scf/fock.py — build_fock() 
- [ ] scf/density.py
- [ ] scf/energy.py
- [ ] scf/damping.py
- [ ] scf/diis.py
- [ ] scf/rhf.py — with profiling integration
- [ ] scf/interface.py
- [ ] benchmarks/generate_references.py — Tier 1 molecules
- [ ] tests/test_scf.py — Tier 1 benchmarks pass
```

### Step 4: p/d-type integrals + Tier 2 benchmarks (Day 5-6)

```
- [ ] integrals/obara_saika.py — recurrence for overlap, kinetic, nuclear
- [ ] integrals/eri.py — extend to (sp|sp), (spd|spd)
- [ ] integrals/screening.py — Schwarz bounds
- [ ] basis sets: 6-31G, 6-31G*, cc-pVDZ tested via BSE
- [ ] benchmarks: Tier 2 molecules (H₂O, NH₃, CH₄, HF, CO, N₂)
- [ ] tests: all Tier 2 pass within 1 μHartree of PySCF
```

### Step 5: UHF + Tier 3 benchmarks (Day 7-8)

```
- [ ] scf/uhf.py
- [ ] scf/fock.py — build_fock_uhf()
- [ ] benchmarks: Tier 3 molecules (H atom, Li, O₂, NO, CH₃, OH)
- [ ] benchmarks: Tier 4 charged species
- [ ] benchmarks: Tier 5 stress tests (stretched H₂ UHF)
- [ ] tests: all Tier 3-4 pass
```

### Step 6: Input interface + properties + examples (Day 9)

```
- [ ] io/input_parser.py
- [ ] properties/mulliken.py, dipole.py, orbital_analysis.py
- [ ] __init__.py — public API
- [ ] examples/ — all 8 examples
- [ ] profiling/jax_profiler.py — JIT/memory profiling
```

### Step 7: Gradients + PES scan (Day 10)

```
- [ ] grad/rhf_grad.py
- [ ] grad/numerical_grad.py
- [ ] examples/04_pes_scan.py, 05_forces.py
- [ ] tests/test_gradient.py — analytic vs finite diff
```

### Step 8: Geometry optimization (Day 11-12)

```
- [ ] geomopt/optimizer.py — BFGS
- [ ] geomopt/hessian.py
- [ ] examples/07_geom_opt.py
- [ ] tests/test_geomopt.py — H₂O optimization to equilibrium
```

---

## 10. FP Abstraction Map (unchanged)

```
Component              FP Abstraction    Parallelism       Hardware
─────────────────────  ────────────────  ────────────────  ────────
Integral computation   Applicative       Fully parallel    GPU
  S, T, V matrices     Functor(fmap)    vmap over pairs   GPU
  ERI tensor           Applicative       vmap over quads   GPU
  Schwarz screening    Filter/Pred       CPU-side filter   CPU
  Contraction          Foldable          fold over prims   GPU if large

SCF iteration          State Monad       Sequential steps  CPU control
  Fock build           Applicative       J,K independent   GPU
  Diagonalization      Pure function     Single call       CPU or GPU
  DIIS                 Writer+State      Small linalg      CPU
  Convergence check    Predicate         Scalar compare    CPU
  Density matrix       Functor           einsum            GPU

Energy computation     Foldable          Trace = fold      GPU
  E_elec = Tr(D(H+F)) Catamorphism      Single reduction  GPU
  E_nuc = Σ ZZ/R      Foldable          fold over pairs   CPU

Basis transformation   Functor           Matrix multiply   GPU
  AO → MO: C^T O C    Covariant         matmul            GPU
  MO → AO: C D C^T    Contravariant     matmul            GPU

Gradient               Adjunction        One backward pass GPU
  Forces = -dE/dR     VJP (right adj)   jax.grad          GPU

PES scan               Applicative       Fully parallel    GPU×N
Geom optimization      Monad             Sequential        CPU control
```

---

## 11. Dependencies

```toml
[project]
name = "jax-qc"
requires-python = ">=3.10"
dependencies = [
    "jax[cuda12]>=0.4.30",
    "jaxlib>=0.4.30",
    "chex>=0.1.8",
    "basis-set-exchange>=0.9",
    "pyyaml>=6.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pyscf>=2.4",
]
```

---

## 12. Code Quality Standards

1. **Every function must have a docstring** stating: (a) what FP abstraction it is, (b) whether it is `jit`-compatible, (c) its pure/effectful status.

2. **No global mutable state.** All state flows through function arguments and return values.

3. **Type hints on all public functions.** Use `jnp.ndarray` for JAX arrays, `chex.dataclass` types for structured data.

4. **Every `@jit` function must be tested** with concrete inputs before use in a pipeline.

5. **Numerical precision**: All computations in float64. Energy agreement to 1 μHartree vs PySCF. Gradient agreement to 1e-6 vs finite difference.

6. **No Python loops over basis indices in jitted code.** Use einsum/vmap. Python loops acceptable outside `@jit`.

7. **Profiling**: All major computation stages wrapped in `StageTimer` context managers. Profiling is off by default (`config.profile = False`) and adds zero overhead when disabled.
