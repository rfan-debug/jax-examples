"""Nuclear attraction integrals.

V_{mu nu} = - sum_C Z_C * <chi_mu | 1 / |r - R_C| | chi_nu>

For arbitrary angular momentum we use the McMurchie-Davidson primitive
in :mod:`jax_qc.integrals.obara_saika`. The nuclear-attraction primitive
already loops over nuclei internally and contracts with shell coefficients
above; here we just tile the resulting Cartesian shell blocks into the
matrix and apply the optional spherical projection.

Also provides ``nuclear_repulsion_energy`` (Foldable over atom pairs).
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import BasisSet, Molecule, Shell
from jax_qc.integrals.boys import boys_f0
from jax_qc.integrals.gaussian_product import (
    distance_squared,
    gaussian_product_center,
)
from jax_qc.integrals.obara_saika import (
    contracted_nuclear_block,
    n_cartesian,
)
from jax_qc.integrals.overlap import (
    _shell_block_spherical,
    shell_offsets,
)


def nuclear_primitive_ss(alpha, A, beta, B, C, Z):
    """Analytic primitive s|V|s helper (kept for legacy / tests)."""
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = distance_squared(A, B)
    P = gaussian_product_center(alpha, A, beta, B)
    PC2 = distance_squared(P, C)
    return -(2.0 * jnp.pi / p) * Z * jnp.exp(-mu * AB2) * boys_f0(p * PC2)


def nuclear_shell_pair_ss(
    shell_a: Shell,
    shell_b: Shell,
    nuc_coords: jnp.ndarray,
    nuc_charges: jnp.ndarray,
) -> jnp.ndarray:
    block = contracted_nuclear_block(
        shell_a,
        shell_b,
        np.asarray(nuc_coords, dtype=np.float64),
        np.asarray(nuc_charges, dtype=np.float64),
    )
    return jnp.asarray(block[0, 0])


def compute_nuclear_matrix(basis: BasisSet, mol: Molecule) -> jnp.ndarray:
    spherical = bool(basis.spherical)
    shells = basis.shells
    n = basis.n_basis
    V = np.zeros((n, n), dtype=np.float64)
    offsets = shell_offsets(shells, spherical)
    nuc_coords = np.asarray(mol.coords, dtype=np.float64)
    nuc_charges = np.asarray(mol.atomic_numbers, dtype=np.float64)
    for i, sa in enumerate(shells):
        ia0, ia1 = offsets[i], offsets[i + 1]
        for j, sb in enumerate(shells):
            jb0, jb1 = offsets[j], offsets[j + 1]
            if j < i:
                V[ia0:ia1, jb0:jb1] = V[jb0:jb1, ia0:ia1].T
                continue
            block = contracted_nuclear_block(sa, sb, nuc_coords, nuc_charges)
            if spherical:
                block = _shell_block_spherical(block, sa, sb)
            V[ia0:ia1, jb0:jb1] = block
    return jnp.asarray(V)


def nuclear_repulsion_energy(mol: Molecule) -> float:
    """E_nuc = sum_{A<B} Z_A Z_B / R_AB.  FP: Foldable."""
    coords = np.asarray(mol.coords, dtype=np.float64)
    Z = np.asarray(mol.atomic_numbers, dtype=np.float64)
    n = len(Z)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(coords[i] - coords[j]))
            if r == 0.0:
                raise ValueError(
                    f"Atoms {i} and {j} are coincident; cannot compute E_nuc."
                )
            total += float(Z[i] * Z[j]) / r
    return total
