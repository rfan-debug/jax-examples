"""Top-level geometry optimization dispatcher.

``optimize_geometry(mol, basis_name, ...)`` runs the BFGS optimizer
and returns the optimized geometry with convergence information.

FP: Monad over Monad — the optimization loop is the outer Monad;
each gradient evaluation runs the inner SCF Monad.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from jax_qc.core.types import Molecule
from jax_qc.geomopt.optimizer import OptResult, optimize_bfgs


def optimize_geometry(
    mol: Molecule,
    basis_name: str = "sto-3g",
    method: str = "rhf",
    max_steps: int = 50,
    grad_tol: float = 1e-4,
    step_tol: float = 1e-6,
    scf_conv: float = 1e-10,
    max_step_size: float = 0.3,
    verbose: int = 1,
) -> OptResult:
    """Optimize the molecular geometry to minimize the total energy.

    Uses the BFGS quasi-Newton method with backtracking line search.

    Args:
        mol:            initial Molecule with starting coordinates.
        basis_name:     basis set name (any BSE name).
        method:         SCF method ('rhf' or 'uhf').
        max_steps:      maximum number of optimization steps.
        grad_tol:       convergence on max |gradient| (Ha/Bohr).
        step_tol:       convergence on max |step| (Bohr).
        scf_conv:       SCF convergence threshold per step.
        max_step_size:  maximum allowed step magnitude (Bohr).
        verbose:        0=silent, 1=per-step summary.

    Returns:
        OptResult with the optimized Molecule, energy, gradient,
        convergence status, and full trajectory.
    """
    return optimize_bfgs(
        mol,
        basis_name=basis_name,
        method=method,
        max_steps=max_steps,
        grad_tol=grad_tol,
        step_tol=step_tol,
        scf_conv=scf_conv,
        max_step_size=max_step_size,
        verbose=verbose,
    )
