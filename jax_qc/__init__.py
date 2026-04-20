"""JAX-QC: A functional quantum chemistry framework in JAX.

Step 1 provides the foundation: core data types, basis set integration via
Basis Set Exchange, XYZ file IO, and a hierarchical profiling timer.
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
]

__version__ = "0.1.0"
