"""Nuclear attraction integrals.

V_{mu nu} = - sum_C Z_C * <chi_mu | 1 / |r - R_C| | chi_nu>

For a primitive s|V|s integral to a nucleus at C with charge Z:

    V_AB^C = -2 pi / p * Z * exp(-mu |A-B|^2) * F_0(p * |P-C|^2)

where p = alpha + beta, mu = alpha beta / p, and
P = (alpha A + beta B) / p is the Gaussian-product center.

Also provides the nuclear repulsion energy:

    E_nuc = sum_{A < B} Z_A Z_B / R_AB

FP: Applicative over (mu, nu) and Foldable over the nucleus index.
"""

from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import BasisSet, Molecule, Shell
from jax_qc.integrals.boys import boys_f0
from jax_qc.integrals.gaussian_product import (
    distance_squared,
    gaussian_product_center,
)
from jax_qc.integrals.overlap import _assert_all_s


def nuclear_primitive_ss(alpha, A, beta, B, C, Z):
    """Primitive attraction of shell a-b to a single nucleus at C with charge Z."""
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
    """Contracted s|V|s integral, summed over all nuclei.

    Vectorized over primitive pairs and nuclei simultaneously.
    """
    alpha = shell_a.exponents[:, None, None]  # (Na, 1, 1)
    beta = shell_b.exponents[None, :, None]  # (1, Nb, 1)
    ca = shell_a.coefficients[:, None, None]
    cb = shell_b.coefficients[None, :, None]
    # Broadcast nuclei on the last axis.
    C = nuc_coords[None, None, :, :]  # (1, 1, Natom, 3)
    Z = nuc_charges[None, None, :].astype(jnp.float64)

    p = alpha + beta
    mu = alpha * beta / p
    AB2 = distance_squared(shell_a.center, shell_b.center)

    # Gaussian product center per primitive pair, shape (Na, Nb, 1, 3).
    P_vec = (
        alpha[..., None] * shell_a.center[None, None, None, :]
        + beta[..., None] * shell_b.center[None, None, None, :]
    ) / p[..., None]
    PC2 = jnp.sum((P_vec - C) ** 2, axis=-1)  # (Na, Nb, Natom)

    prim = -(2.0 * jnp.pi / p) * Z * jnp.exp(-mu * AB2) * boys_f0(p * PC2)
    return jnp.sum(ca * cb * prim)


def compute_nuclear_matrix(basis: BasisSet, mol: Molecule) -> jnp.ndarray:
    """Build the nuclear-attraction matrix V (n_basis x n_basis)."""
    _assert_all_s(basis.shells)
    nuc_coords = jnp.asarray(mol.coords, dtype=jnp.float64)
    nuc_charges = jnp.asarray(mol.atomic_numbers, dtype=jnp.float64)
    n = basis.n_basis
    rows: List[list] = [[None] * n for _ in range(n)]
    for i, sa in enumerate(basis.shells):
        for j, sb in enumerate(basis.shells):
            if j < i:
                rows[i][j] = rows[j][i]
                continue
            rows[i][j] = nuclear_shell_pair_ss(sa, sb, nuc_coords, nuc_charges)
    return jnp.asarray(np.array([[float(x) for x in row] for row in rows]))


def nuclear_repulsion_energy(mol: Molecule) -> float:
    """E_nuc = sum_{A<B} Z_A Z_B / R_AB.

    FP: Foldable — single reduction over atom pairs.
    """
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
