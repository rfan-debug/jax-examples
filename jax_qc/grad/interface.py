"""Top-level gradient dispatcher.

``compute_gradient(mol, result, basis_name)`` picks the right gradient
method and returns dE/dR as an (n_atoms, 3) array.

FP: Adjunction — the gradient is the right adjoint of the energy.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from jax_qc.core.types import CalcConfig, Molecule, SCFResult
from jax_qc.grad.numerical_grad import numerical_gradient
from jax_qc.grad.rhf_grad import rhf_gradient


def compute_gradient(
    mol: Molecule,
    result: SCFResult,
    basis_name: str = "sto-3g",
    method: str = "analytic",
    step_size: float = 1e-4,
) -> jnp.ndarray:
    """Compute the nuclear gradient dE/dR.

    Args:
        mol:        the Molecule at the reference geometry.
        result:     converged SCF result.
        basis_name: basis set name (must match the SCF calculation).
        method:     'analytic' (default) uses the Hellmann-Feynman + Pulay
                    expression with integral finite differences.
                    'numerical' uses full SCF finite differences (slow but
                    useful for validation).
        step_size:  displacement in Bohr for finite differences.

    Returns:
        (n_atoms, 3) gradient in Hartree/Bohr.
    """
    if method == "analytic":
        return rhf_gradient(mol, result, basis_name=basis_name, step_size=step_size)
    elif method == "numerical":
        config_method = "rhf" if mol.spin == 0 else "uhf"
        return numerical_gradient(
            mol,
            basis_name=basis_name,
            method=config_method,
            step_size=step_size,
        )
    else:
        raise ValueError(
            f"Unknown gradient method '{method}'; expected 'analytic' or 'numerical'."
        )
