"""Finite-difference nuclear gradient.

Computes dE/dR by central finite differences:

    dE/dR_{A,k} = [E(R + h*e_{A,k}) - E(R - h*e_{A,k})] / (2h)

This is the reference implementation for validating the analytic
gradient. It is slow (requires 6*N_atoms SCF evaluations) but
always correct.

FP: Effectful — triggers 6*N_atoms full SCF calculations.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from jax_qc.basis.build import build_basis_set
from jax_qc.core.types import CalcConfig, Molecule, SCFResult, make_molecule
from jax_qc.integrals.interface import compute_integrals
from jax_qc.scf.interface import run_scf


def _energy_at_coords(
    coords: np.ndarray,
    mol: Molecule,
    basis_name: str,
    config: CalcConfig,
) -> float:
    """Compute total SCF energy at displaced coordinates.

    Builds a new Molecule with the given coordinates, reconstructs
    the basis set and integrals, and runs SCF.
    """
    displaced = make_molecule(
        elements=mol.elements,
        coords=jnp.asarray(coords),
        atomic_numbers=mol.atomic_numbers,
        charge=mol.charge,
        spin=mol.spin,
    )
    basis = build_basis_set(displaced, basis_name)
    ints = compute_integrals(displaced, basis)
    result = run_scf(displaced, ints, config)
    return float(result.energy)


def numerical_gradient(
    mol: Molecule,
    basis_name: str = "sto-3g",
    method: str = "rhf",
    step_size: float = 1e-4,
    scf_conv: float = 1e-10,
) -> jnp.ndarray:
    """Compute the nuclear gradient by central finite differences.

    FP: Effectful — runs 6*N_atoms SCF calculations.

    Args:
        mol:        the Molecule.
        basis_name: basis set name.
        method:     'rhf' or 'uhf'.
        step_size:  displacement in Bohr for the finite difference.
        scf_conv:   SCF convergence threshold (tighter than default
                    for gradient accuracy).

    Returns:
        (n_atoms, 3) gradient dE/dR in Hartree/Bohr.
    """
    config = CalcConfig(
        method=method,
        basis=basis_name,
        scf_conv=scf_conv,
        max_scf_iter=200,
    )
    coords = np.asarray(mol.coords, dtype=np.float64)
    n_atoms = coords.shape[0]
    grad = np.zeros_like(coords)

    for a in range(n_atoms):
        for k in range(3):
            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[a, k] += step_size
            coords_minus[a, k] -= step_size

            e_plus = _energy_at_coords(coords_plus, mol, basis_name, config)
            e_minus = _energy_at_coords(coords_minus, mol, basis_name, config)
            grad[a, k] = (e_plus - e_minus) / (2.0 * step_size)

    return jnp.asarray(grad)
