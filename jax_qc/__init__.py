"""JAX-QC: A functional quantum chemistry framework in JAX.

Steps 1-5 provide RHF and UHF SCF on closed- and open-shell molecules.
Step 6 adds post-SCF properties (Mulliken, dipole, orbital analysis)
and a high-level convenience API (``energy()``, ``run()``, ``run_xyz()``).
"""

import jax

jax.config.update("jax_enable_x64", True)

# --- Core types ---
from jax_qc.core.types import (
    Primitive,
    Shell,
    BasisSet,
    Molecule,
    IntegralSet,
    SCFState,
    SCFResult,
    CalcConfig,
)

# --- Low-level pipeline ---
from jax_qc.basis.build import build_basis_set
from jax_qc.integrals.interface import compute_integrals
from jax_qc.io.xyz import read_xyz, write_xyz
from jax_qc.profiling.timer import StageTimer
from jax_qc.profiling.report import format_report
from jax_qc.scf.interface import run_scf
from jax_qc.scf.rhf import run_rhf
from jax_qc.scf.uhf import run_uhf

# --- Properties ---
from jax_qc.properties.mulliken import mulliken_charges, mulliken_analysis
from jax_qc.properties.dipole import dipole_moment, dipole_moment_debye
from jax_qc.properties.orbital_analysis import orbital_analysis, OrbitalInfo

# --- High-level convenience API ---
from jax_qc.io.input_parser import build_molecule, energy, run, run_xyz

__all__ = [
    # Types
    "Primitive",
    "Shell",
    "BasisSet",
    "Molecule",
    "IntegralSet",
    "SCFState",
    "SCFResult",
    "CalcConfig",
    "OrbitalInfo",
    # Low-level pipeline
    "build_basis_set",
    "compute_integrals",
    "read_xyz",
    "write_xyz",
    "StageTimer",
    "format_report",
    "run_scf",
    "run_rhf",
    "run_uhf",
    # Properties
    "mulliken_charges",
    "mulliken_analysis",
    "dipole_moment",
    "dipole_moment_debye",
    "orbital_analysis",
    # Convenience API
    "build_molecule",
    "energy",
    "run",
    "run_xyz",
]

__version__ = "0.1.0"
