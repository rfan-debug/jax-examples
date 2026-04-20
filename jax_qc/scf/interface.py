"""Top-level SCF dispatcher.

``run_scf(mol, ints, config)`` picks the right driver based on
``config.method``. Step 3 only ships RHF; UHF lands in Step 5.
"""

from __future__ import annotations

from typing import Optional

from jax_qc.core.types import CalcConfig, IntegralSet, Molecule, SCFResult
from jax_qc.profiling.timer import StageTimer
from jax_qc.scf.rhf import run_rhf


def run_scf(
    mol: Molecule,
    ints: IntegralSet,
    config: CalcConfig,
    timer: Optional[StageTimer] = None,
) -> SCFResult:
    """Dispatch on ``config.method`` and run the appropriate SCF driver."""
    method = config.method.lower()
    if method == "rhf":
        return run_rhf(mol, ints, config, timer=timer)
    raise NotImplementedError(
        f"SCF method '{config.method}' is not implemented yet; Step 3 ships RHF only."
    )
