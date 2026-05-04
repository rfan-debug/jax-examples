"""JAX-QC: A functional quantum chemistry framework in JAX.

Steps 1-4 provide RHF on closed-shell molecules with general angular
momentum.  Step 5 adds UHF for open-shell systems.
"""

import jax

jax.config.update("jax_enable_x64", True)

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
from jax_qc.basis.build import build_basis_set
from jax_qc.integrals.interface import compute_integrals
from jax_qc.io.xyz import read_xyz, write_xyz
from jax_qc.profiling.timer import StageTimer
from jax_qc.profiling.report import format_report
from jax_qc.scf.interface import run_scf
from jax_qc.scf.rhf import run_rhf
from jax_qc.scf.uhf import run_uhf

__all__ = [
    "Primitive",
    "Shell",
    "BasisSet",
    "Molecule",
    "IntegralSet",
    "SCFState",
    "SCFResult",
    "CalcConfig",
    "build_basis_set",
    "compute_integrals",
    "read_xyz",
    "write_xyz",
    "StageTimer",
    "format_report",
    "run_scf",
    "run_rhf",
    "run_uhf",
]

__version__ = "0.1.0"
