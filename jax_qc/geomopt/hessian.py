"""Numerical Hessian via finite differences of the gradient.

The Hessian d^2E/dR_i dR_j is approximated by central finite
differences of the analytic gradient:

    H_{ij} = [g_i(R + h*e_j) - g_i(R - h*e_j)] / (2h)

where g is the gradient vector (3*N_atoms) and e_j is the j-th
Cartesian unit displacement.

FP: Effectful — requires 6*N_atoms gradient evaluations.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_qc.basis.build import build_basis_set
from jax_qc.core.types import CalcConfig, Molecule, make_molecule
from jax_qc.grad.rhf_grad import rhf_gradient
from jax_qc.integrals.interface import compute_integrals
from jax_qc.scf.interface import run_scf


def numerical_hessian(
    mol: Molecule,
    basis_name: str = "sto-3g",
    method: str = "rhf",
    step_size: float = 1e-3,
    scf_conv: float = 1e-10,
) -> jnp.ndarray:
    """Compute the Hessian by finite differences of the analytic gradient.

    FP: Effectful — runs 6*N_atoms SCF + gradient evaluations.

    Args:
        mol:        the Molecule at the reference geometry.
        basis_name: basis set name.
        method:     SCF method ('rhf' or 'uhf').
        step_size:  displacement in Bohr.
        scf_conv:   SCF convergence threshold.

    Returns:
        (3*n_atoms, 3*n_atoms) Hessian in Hartree/Bohr^2.
    """
    coords = np.asarray(mol.coords, dtype=np.float64)
    n_atoms = coords.shape[0]
    n_dof = 3 * n_atoms
    config = CalcConfig(
        method=method, basis=basis_name, scf_conv=scf_conv, max_scf_iter=200
    )

    hessian = np.zeros((n_dof, n_dof), dtype=np.float64)

    for j_atom in range(n_atoms):
        for j_cart in range(3):
            j = j_atom * 3 + j_cart

            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[j_atom, j_cart] += step_size
            coords_minus[j_atom, j_cart] -= step_size

            # Gradient at +h
            mol_plus = make_molecule(
                elements=mol.elements,
                coords=jnp.asarray(coords_plus),
                atomic_numbers=mol.atomic_numbers,
                charge=mol.charge,
                spin=mol.spin,
            )
            basis_p = build_basis_set(mol_plus, basis_name)
            ints_p = compute_integrals(mol_plus, basis_p)
            result_p = run_scf(mol_plus, ints_p, config)
            grad_plus = np.asarray(
                rhf_gradient(mol_plus, result_p, basis_name)
            ).flatten()

            # Gradient at -h
            mol_minus = make_molecule(
                elements=mol.elements,
                coords=jnp.asarray(coords_minus),
                atomic_numbers=mol.atomic_numbers,
                charge=mol.charge,
                spin=mol.spin,
            )
            basis_m = build_basis_set(mol_minus, basis_name)
            ints_m = compute_integrals(mol_minus, basis_m)
            result_m = run_scf(mol_minus, ints_m, config)
            grad_minus = np.asarray(
                rhf_gradient(mol_minus, result_m, basis_name)
            ).flatten()

            hessian[:, j] = (grad_plus - grad_minus) / (2.0 * step_size)

    # Symmetrize (central differences are symmetric in exact arithmetic)
    hessian = 0.5 * (hessian + hessian.T)
    return jnp.asarray(hessian)
