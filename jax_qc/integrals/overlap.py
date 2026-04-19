"""Overlap integrals S_{mu nu} = <chi_mu | chi_nu>.

Step 2: s-type (l = 0) shells only. The s|s primitive overlap is

    <g_A | g_B> = (pi / p)^(3/2) * exp(-mu * |A - B|^2)

where p = alpha + beta and mu = alpha beta / p. For contracted shells
with pre-normalized coefficients ``c_i``, ``d_j`` the overlap is simply

    S_{AB} = sum_{i,j} c_i d_j (pi/p_ij)^(3/2) exp(-mu_ij |A-B|^2).

FP: Applicative — every (mu, nu) matrix element is independent.
"""

from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import BasisSet, Shell
from jax_qc.integrals.gaussian_product import distance_squared


def overlap_primitive_ss(alpha, A, beta, B):
    """Primitive s|s overlap. Scalar output; inputs broadcast."""
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = distance_squared(A, B)
    return (jnp.pi / p) ** 1.5 * jnp.exp(-mu * AB2)


def overlap_shell_pair_ss(shell_a: Shell, shell_b: Shell) -> jnp.ndarray:
    """Contracted s|s shell-pair overlap.

    Returns a 0-d JAX array. Vectorizes over all primitive pairs via
    broadcasting.
    """
    alpha = shell_a.exponents[:, None]  # (Na, 1)
    beta = shell_b.exponents[None, :]  # (1, Nb)
    ca = shell_a.coefficients[:, None]
    cb = shell_b.coefficients[None, :]
    prim = overlap_primitive_ss(alpha, shell_a.center, beta, shell_b.center)
    return jnp.sum(ca * cb * prim)


def _assert_all_s(shells: Sequence[Shell]) -> None:
    for i, sh in enumerate(shells):
        if int(sh.angular_momentum) != 0:
            raise NotImplementedError(
                "Step 2 only supports s-type shells; "
                f"shell {i} has l={int(sh.angular_momentum)}."
            )


def compute_overlap_matrix(basis: BasisSet) -> jnp.ndarray:
    """Build the full overlap matrix S (n_basis x n_basis) for an s-only basis.

    FP: Applicative at the matrix level — we compute all unique shell
    pairs and write both (i, j) and (j, i) positions for symmetry.
    """
    _assert_all_s(basis.shells)
    n = basis.n_basis
    rows: List[list] = [[None] * n for _ in range(n)]
    for i, sa in enumerate(basis.shells):
        for j, sb in enumerate(basis.shells):
            if j < i:
                rows[i][j] = rows[j][i]
                continue
            rows[i][j] = overlap_shell_pair_ss(sa, sb)
    return jnp.asarray(np.array([[float(x) for x in row] for row in rows]))
