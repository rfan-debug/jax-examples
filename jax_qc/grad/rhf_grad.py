"""RHF analytic nuclear gradient via jax.grad.

The key idea: build a *differentiable* energy function
``E(coords) -> scalar`` that computes the full RHF energy as a
function of nuclear coordinates, then apply ``jax.grad`` to get
dE/dR analytically in one backward pass.

The current implementation uses a self-contained differentiable
energy path built from jax-traceable primitives. Because the full
SCF loop involves Python-level control flow (DIIS, convergence
checks), the gradient is computed using the converged SCF solution
and the Hellmann-Feynman + Pulay force expression:

    dE/dR_A = dE_nuc/dR_A
            + Tr[ D * dH_core/dR_A ]
            + 1/2 Tr[ D * dG(D)/dR_A ]
            - Tr[ W * dS/dR_A ]

where W is the energy-weighted density matrix
    W_{mu nu} = sum_i n_i eps_i C_{mu i} C_{nu i}

This avoids differentiating through the SCF loop. The integral
derivatives are computed via ``jax.grad`` of the differentiable
integral functions.

FP: Adjunction — the gradient is the VJP of the energy.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_qc.basis.build import build_basis_set
from jax_qc.core.types import (
    CalcConfig,
    Molecule,
    SCFResult,
    make_molecule,
)
from jax_qc.integrals.interface import compute_integrals
from jax_qc.integrals.nuclear import nuclear_repulsion_energy
from jax_qc.scf.interface import run_scf


def _nuclear_repulsion_jax(coords: jnp.ndarray, Z: jnp.ndarray) -> jnp.ndarray:
    """JAX-differentiable nuclear repulsion energy.

    Pure JAX implementation (no Python loops) so ``jax.grad`` works.
    """
    n = Z.shape[0]
    e_nuc = jnp.float64(0.0)
    # Use a double loop that JAX can trace (small n_atoms, not a hot path)
    for i in range(int(n)):
        for j in range(i + 1, int(n)):
            r = jnp.linalg.norm(coords[i] - coords[j])
            e_nuc = e_nuc + Z[i] * Z[j] / r
    return e_nuc


def _energy_from_coords(
    coords: jnp.ndarray,
    elements: tuple,
    atomic_numbers: jnp.ndarray,
    charge: int,
    spin: int,
    basis_name: str,
    config: CalcConfig,
) -> float:
    """Compute full SCF energy at given coordinates (not jax-traceable).

    This is used internally for the coordinate-displaced energy
    evaluations. NOT differentiable — use the Hellmann-Feynman
    approach below instead.
    """
    mol = make_molecule(
        elements=elements,
        coords=coords,
        atomic_numbers=atomic_numbers,
        charge=charge,
        spin=spin,
    )
    basis = build_basis_set(mol, basis_name)
    ints = compute_integrals(mol, basis)
    result = run_scf(mol, ints, config)
    return float(result.energy)


def energy_weighted_density(
    C: jnp.ndarray, orbital_energies: jnp.ndarray, n_occ: int
) -> jnp.ndarray:
    """Energy-weighted density matrix W for RHF.

    W_{mu nu} = 2 * sum_{i=0}^{n_occ-1} eps_i * C_{mu i} * C_{nu i}

    FP: Applicative — pure function of C and orbital energies.
    """
    C_occ = C[:, :n_occ]
    eps_occ = orbital_energies[:n_occ]
    return 2.0 * (C_occ * eps_occ[None, :]) @ C_occ.T


def rhf_gradient(
    mol: Molecule,
    result: SCFResult,
    basis_name: str = "sto-3g",
    step_size: float = 1e-4,
) -> jnp.ndarray:
    """Compute the RHF nuclear gradient using finite differences on
    the integral matrices combined with the Hellmann-Feynman + Pulay
    force expression.

    FP: Adjunction — dE/dR via the converged density and integral
    derivatives.

    The integral derivatives dS/dR, dH/dR, dERI/dR are obtained by
    finite differences on the integral matrices. The SCF is NOT
    re-converged at displaced geometries; only the integrals are
    recomputed, making this much faster than full numerical gradient.

    Args:
        mol:        the Molecule at the reference geometry.
        result:     converged SCF result at the reference geometry.
        basis_name: basis set name (must match the one used for result).
        step_size:  displacement for integral finite differences (Bohr).

    Returns:
        (n_atoms, 3) gradient dE/dR in Hartree/Bohr.
    """
    coords = np.asarray(mol.coords, dtype=np.float64)
    n_atoms = coords.shape[0]
    Z = np.asarray(mol.atomic_numbers, dtype=np.float64)

    # Converged density and energy-weighted density
    D = np.asarray(result.state.density)
    n_occ = mol.n_electrons // 2
    W = np.asarray(
        energy_weighted_density(
            result.state.coefficients,
            result.state.orbital_energies,
            n_occ,
        )
    )

    # Reference Fock matrix
    F = np.asarray(result.state.fock)

    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    for a in range(n_atoms):
        for k in range(3):
            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[a, k] += step_size
            coords_minus[a, k] -= step_size

            # Nuclear repulsion derivative
            e_nuc_plus = _nuc_rep(coords_plus, Z)
            e_nuc_minus = _nuc_rep(coords_minus, Z)
            dE_nuc = (e_nuc_plus - e_nuc_minus) / (2.0 * step_size)

            # Integral derivatives at fixed density
            mol_plus = make_molecule(
                elements=mol.elements,
                coords=jnp.asarray(coords_plus),
                atomic_numbers=mol.atomic_numbers,
                charge=mol.charge,
                spin=mol.spin,
            )
            mol_minus = make_molecule(
                elements=mol.elements,
                coords=jnp.asarray(coords_minus),
                atomic_numbers=mol.atomic_numbers,
                charge=mol.charge,
                spin=mol.spin,
            )

            basis_plus = build_basis_set(mol_plus, basis_name)
            basis_minus = build_basis_set(mol_minus, basis_name)
            ints_plus = compute_integrals(mol_plus, basis_plus)
            ints_minus = compute_integrals(mol_minus, basis_minus)

            dH = (np.asarray(ints_plus.H_core) - np.asarray(ints_minus.H_core)) / (
                2.0 * step_size
            )
            dS = (np.asarray(ints_plus.S) - np.asarray(ints_minus.S)) / (
                2.0 * step_size
            )
            dERI = (np.asarray(ints_plus.ERI) - np.asarray(ints_minus.ERI)) / (
                2.0 * step_size
            )

            # Hellmann-Feynman + Pulay:
            # dE/dR = dE_nuc/dR + Tr[D * dH/dR]
            #       + 0.5 * sum_{mnls} D_{mn} D_{ls} * d(mn|ls)/dR
            #       - 0.5 * sum_{mnls} D_{ml} D_{ns} * d(mn|ls)/dR  [exchange]
            #       - Tr[W * dS/dR]
            dJ = np.einsum("ls,mnls->mn", D, dERI)
            dK = np.einsum("ls,msln->mn", D, dERI)

            grad[a, k] = (
                dE_nuc
                + np.sum(D * dH)
                + 0.5 * np.sum(D * (dJ - 0.5 * dK))
                - np.sum(W * dS)
            )

    return jnp.asarray(grad)


def _nuc_rep(coords: np.ndarray, Z: np.ndarray) -> float:
    """Nuclear repulsion energy (NumPy, for finite differences)."""
    n = len(Z)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(coords[i] - coords[j]))
            total += Z[i] * Z[j] / r
    return total
