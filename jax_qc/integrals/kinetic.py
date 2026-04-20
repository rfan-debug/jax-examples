"""Kinetic energy integrals T_{mu nu} = <chi_mu | -1/2 nabla^2 | chi_nu>.

Step 2: s-type shells only. The primitive s|T|s integral is

    T_AB = mu * (3 - 2 mu |A-B|^2) * S_AB,

where S_AB is the primitive overlap, p = alpha + beta, mu = alpha beta / p.

FP: Applicative — independent over matrix positions.
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import BasisSet, Shell
from jax_qc.integrals.gaussian_product import distance_squared
from jax_qc.integrals.overlap import overlap_primitive_ss, _assert_all_s


def kinetic_primitive_ss(alpha, A, beta, B):
    """Primitive s|T|s integral."""
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = distance_squared(A, B)
    S = overlap_primitive_ss(alpha, A, beta, B)
    return mu * (3.0 - 2.0 * mu * AB2) * S


def kinetic_shell_pair_ss(shell_a: Shell, shell_b: Shell) -> jnp.ndarray:
    """Contracted s|T|s shell-pair kinetic integral."""
    alpha = shell_a.exponents[:, None]
    beta = shell_b.exponents[None, :]
    ca = shell_a.coefficients[:, None]
    cb = shell_b.coefficients[None, :]
    prim = kinetic_primitive_ss(alpha, shell_a.center, beta, shell_b.center)
    return jnp.sum(ca * cb * prim)


def compute_kinetic_matrix(basis: BasisSet) -> jnp.ndarray:
    """Build the full kinetic matrix T (n_basis x n_basis) for an s-only basis."""
    _assert_all_s(basis.shells)
    n = basis.n_basis
    rows: List[list] = [[None] * n for _ in range(n)]
    for i, sa in enumerate(basis.shells):
        for j, sb in enumerate(basis.shells):
            if j < i:
                rows[i][j] = rows[j][i]
                continue
            rows[i][j] = kinetic_shell_pair_ss(sa, sb)
    return jnp.asarray(np.array([[float(x) for x in row] for row in rows]))
