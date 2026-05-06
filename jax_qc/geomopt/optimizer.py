"""BFGS geometry optimizer.

Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton
method for minimizing the total energy with respect to nuclear
coordinates.

The BFGS update builds an approximate inverse Hessian from successive
gradient differences, enabling near-quadratic convergence without
computing the full Hessian at each step.

FP: Monad over Monad — the outer optimization loop threads the
geometry through gradient evaluations (each of which runs a full
Monadic SCF loop inside). The optimizer state is immutable.

Design notes:
- Initial Hessian: identity matrix (scaled by a reasonable step size)
- Line search: backtracking Armijo to ensure sufficient decrease
- Convergence: max |gradient| < threshold
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np

from jax_qc.basis.build import build_basis_set
from jax_qc.core.types import CalcConfig, Molecule, SCFResult, make_molecule
from jax_qc.grad.rhf_grad import rhf_gradient
from jax_qc.integrals.interface import compute_integrals
from jax_qc.scf.interface import run_scf


class OptResult(NamedTuple):
    """Result of a geometry optimization.

    Fields:
        converged:    True if the gradient norm is below threshold.
        molecule:     the optimized Molecule.
        energy:       total energy at the optimized geometry.
        gradient:     (n_atoms, 3) gradient at the final geometry.
        n_steps:      number of optimization steps taken.
        trajectory:   list of (coords, energy) at each step.
    """

    converged: bool
    molecule: Molecule
    energy: float
    gradient: jnp.ndarray
    n_steps: int
    trajectory: list


def _energy_and_gradient(
    coords: np.ndarray,
    mol: Molecule,
    basis_name: str,
    config: CalcConfig,
) -> tuple:
    """Compute energy and gradient at given coordinates.

    Returns (energy, gradient, scf_result).
    """
    new_mol = make_molecule(
        elements=mol.elements,
        coords=jnp.asarray(coords),
        atomic_numbers=mol.atomic_numbers,
        charge=mol.charge,
        spin=mol.spin,
    )
    basis = build_basis_set(new_mol, basis_name)
    ints = compute_integrals(new_mol, basis)
    result = run_scf(new_mol, ints, config)
    grad = np.asarray(rhf_gradient(new_mol, result, basis_name=basis_name))
    return float(result.energy), grad, new_mol


def optimize_bfgs(
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
    """Optimize molecular geometry using BFGS.

    FP: Monadic — threads geometry through gradient evaluations.

    Args:
        mol:            initial Molecule.
        basis_name:     basis set name.
        method:         'rhf' or 'uhf'.
        max_steps:      maximum optimization steps.
        grad_tol:       convergence threshold on max |gradient| (Ha/Bohr).
        step_tol:       convergence threshold on max |step| (Bohr).
        scf_conv:       SCF convergence threshold.
        max_step_size:  maximum allowed step magnitude (Bohr).
        verbose:        0=silent, 1=summary per step.

    Returns:
        OptResult with the optimized geometry and trajectory.
    """
    config = CalcConfig(
        method=method, basis=basis_name, scf_conv=scf_conv, max_scf_iter=200
    )
    coords = np.asarray(mol.coords, dtype=np.float64)
    n_atoms = coords.shape[0]
    n_dof = 3 * n_atoms

    # Initial energy and gradient
    energy, grad, current_mol = _energy_and_gradient(coords, mol, basis_name, config)
    grad_flat = grad.flatten()

    # Initial inverse Hessian approximation: identity
    H_inv = np.eye(n_dof, dtype=np.float64)

    trajectory = [(coords.copy(), energy)]
    converged = False

    if verbose >= 1:
        print(
            f"  Step   0: E = {energy:16.10f}  |g|_max = {np.max(np.abs(grad_flat)):.2e}"
        )

    for step in range(1, max_steps + 1):
        # Check convergence
        g_max = float(np.max(np.abs(grad_flat)))
        if g_max < grad_tol:
            converged = True
            if verbose >= 1:
                print(f"  Converged: max |grad| = {g_max:.2e} < {grad_tol:.2e}")
            break

        # BFGS search direction: p = -H_inv @ g
        direction = -H_inv @ grad_flat

        # Trust region: clip step to max_step_size
        step_norm = float(np.linalg.norm(direction))
        if step_norm > max_step_size:
            direction = direction * (max_step_size / step_norm)

        # Backtracking line search (Armijo condition)
        alpha = 1.0
        c1 = 1e-4  # sufficient decrease parameter
        energy_old = energy
        grad_old = grad_flat.copy()

        for _ in range(10):
            new_coords = coords + alpha * direction.reshape(n_atoms, 3)
            energy_new, grad_new, new_mol = _energy_and_gradient(
                new_coords, mol, basis_name, config
            )
            if energy_new <= energy_old + c1 * alpha * np.dot(grad_old, direction):
                break
            alpha *= 0.5
        else:
            # Accept the step even if line search didn't converge
            pass

        # Compute BFGS update vectors
        s = (alpha * direction).reshape(-1)  # step
        coords = new_coords
        energy = energy_new
        grad_flat_new = grad_new.flatten()
        y = grad_flat_new - grad_old  # gradient change

        # Check step convergence
        s_max = float(np.max(np.abs(s)))
        if s_max < step_tol and g_max < grad_tol * 10:
            converged = True
            if verbose >= 1:
                print(f"  Converged: max |step| = {s_max:.2e} < {step_tol:.2e}")
            grad_flat = grad_flat_new
            current_mol = new_mol
            trajectory.append((coords.copy(), energy))
            break

        # BFGS inverse Hessian update
        sy = float(np.dot(s, y))
        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(n_dof)
            V = I - rho * np.outer(s, y)
            H_inv = V @ H_inv @ V.T + rho * np.outer(s, s)

        grad_flat = grad_flat_new
        current_mol = new_mol
        trajectory.append((coords.copy(), energy))

        if verbose >= 1:
            print(
                f"  Step {step:3d}: E = {energy:16.10f}"
                f"  |g|_max = {np.max(np.abs(grad_flat)):.2e}"
                f"  |s|_max = {s_max:.2e}"
            )

    return OptResult(
        converged=converged,
        molecule=current_mol,
        energy=energy,
        gradient=jnp.asarray(grad_flat.reshape(n_atoms, 3)),
        n_steps=len(trajectory) - 1,
        trajectory=trajectory,
    )
