"""Mulliken population analysis.

Mulliken gross atomic populations partition the total electron count
among atoms using the overlap-weighted density:

    P_{mu} = (D S)_{mu mu}               per-basis-function population
    q_A    = Z_A - sum_{mu in A} P_{mu}   Mulliken charge on atom A

FP: Foldable — a reduction of D and S into per-atom scalars.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from jax_qc.core.types import BasisSet, Molecule, SCFResult


def mulliken_populations(
    D: jnp.ndarray, S: jnp.ndarray, basis_to_atom: jnp.ndarray, n_atoms: int
) -> jnp.ndarray:
    """Compute gross Mulliken populations per atom.

    FP: Foldable — pure function of D, S, and the mapping arrays.

    Args:
        D:             (n_basis, n_basis) total density matrix.
        S:             (n_basis, n_basis) overlap matrix.
        basis_to_atom: (n_basis,) int array mapping each basis function
                       to its parent atom index.
        n_atoms:       number of atoms in the molecule.

    Returns:
        (n_atoms,) array of gross electron populations per atom.
    """
    PS = jnp.diag(D @ S)  # (n_basis,) per-AO populations
    # Sum populations by atom
    pops = jnp.zeros(n_atoms)
    for mu in range(len(PS)):
        atom_idx = int(basis_to_atom[mu])
        pops = pops.at[atom_idx].add(PS[mu])
    return pops


def mulliken_charges(result: SCFResult, mol: Molecule, basis: BasisSet) -> jnp.ndarray:
    """Compute Mulliken partial atomic charges.

    FP: Foldable — pure function of SCF result and molecule data.

    Args:
        result: converged SCF result.
        mol:    the Molecule.
        basis:  the BasisSet (provides basis_to_atom mapping).

    Returns:
        (n_atoms,) array of Mulliken charges (nuclear charge - population).
    """
    D = result.state.density
    S = result.S
    n_atoms = len(mol.elements)
    pops = mulliken_populations(D, S, basis.basis_to_atom, n_atoms)
    Z = jnp.asarray(mol.atomic_numbers, dtype=jnp.float64)
    return Z - pops


def mulliken_analysis(
    result: SCFResult, mol: Molecule, basis: BasisSet
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Full Mulliken analysis: returns (populations, charges).

    FP: Foldable.

    Returns:
        (populations, charges) tuple of (n_atoms,) arrays.
    """
    D = result.state.density
    S = result.S
    n_atoms = len(mol.elements)
    pops = mulliken_populations(D, S, basis.basis_to_atom, n_atoms)
    Z = jnp.asarray(mol.atomic_numbers, dtype=jnp.float64)
    charges = Z - pops
    return pops, charges
