"""Kinetic energy integrals T_{mu nu} = <chi_mu | -1/2 nabla^2 | chi_nu>.

General-l implementation backed by the McMurchie-Davidson primitives
in :mod:`jax_qc.integrals.obara_saika`. Spherical projection mirrors
:mod:`jax_qc.integrals.overlap`.
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import BasisSet, Shell
from jax_qc.integrals.gaussian_product import distance_squared
from jax_qc.integrals.obara_saika import (
    contracted_kinetic_block,
    n_cartesian,
)
from jax_qc.integrals.overlap import (
    _shell_block_spherical,
    shell_offsets,
)
from jax_qc.integrals.overlap import overlap_primitive_ss


def kinetic_primitive_ss(alpha, A, beta, B):
    """Analytic primitive s|T|s integral (Step 2 helper, kept for tests)."""
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = distance_squared(A, B)
    S = overlap_primitive_ss(alpha, A, beta, B)
    return mu * (3.0 - 2.0 * mu * AB2) * S


def kinetic_shell_pair_ss(shell_a: Shell, shell_b: Shell) -> jnp.ndarray:
    block = contracted_kinetic_block(shell_a, shell_b)
    return jnp.asarray(block[0, 0])


def compute_kinetic_matrix(basis: BasisSet) -> jnp.ndarray:
    spherical = bool(basis.spherical)
    shells = basis.shells
    n = basis.n_basis
    T = np.zeros((n, n), dtype=np.float64)
    offsets = shell_offsets(shells, spherical)
    for i, sa in enumerate(shells):
        ia0, ia1 = offsets[i], offsets[i + 1]
        for j, sb in enumerate(shells):
            jb0, jb1 = offsets[j], offsets[j + 1]
            if j < i:
                T[ia0:ia1, jb0:jb1] = T[jb0:jb1, ia0:ia1].T
                continue
            block = contracted_kinetic_block(sa, sb)
            if spherical:
                block = _shell_block_spherical(block, sa, sb)
            T[ia0:ia1, jb0:jb1] = block
    return jnp.asarray(T)
