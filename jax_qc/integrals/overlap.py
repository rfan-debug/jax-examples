"""Overlap integrals S_{mu nu} = <chi_mu | chi_nu>.

For arbitrary angular momentum we delegate the per-primitive Cartesian
block to :mod:`jax_qc.integrals.obara_saika`. Each shell pair (a, b)
yields a dense Cartesian block of shape ``(nc_a, nc_b)``; for a spherical
basis we transform each shell dimension through ``cart_to_spherical(l)``
to land on the ``(2 l + 1)`` real solid harmonics used by libcint / PySCF.

FP: Applicative — every (i, j) shell-pair block is independent.
"""

from __future__ import annotations

from typing import List, Sequence

import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import BasisSet, Shell
from jax_qc.integrals.gaussian_product import distance_squared
from jax_qc.integrals.obara_saika import (
    contracted_overlap_block,
    n_cartesian,
)
from jax_qc.integrals.spherical import cart_to_spherical


def overlap_primitive_ss(alpha, A, beta, B):
    """Primitive s|s overlap (analytic). Convenience for tests / Step 2 callers."""
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = distance_squared(A, B)
    return (jnp.pi / p) ** 1.5 * jnp.exp(-mu * AB2)


def overlap_shell_pair_ss(shell_a: Shell, shell_b: Shell) -> jnp.ndarray:
    """Contracted s|s shell-pair overlap. Convenience for legacy callers."""
    block = contracted_overlap_block(shell_a, shell_b)
    return jnp.asarray(block[0, 0])


def _assert_all_s(shells: Sequence[Shell]) -> None:
    """Legacy guard kept for the few s-only call sites that still use it."""
    for i, sh in enumerate(shells):
        if int(sh.angular_momentum) != 0:
            raise NotImplementedError(
                f"Caller restricted to s-type shells; shell {i} has "
                f"l={int(sh.angular_momentum)}."
            )


def _shell_block_spherical(block_cart: np.ndarray, sa: Shell, sb: Shell) -> np.ndarray:
    Ca = cart_to_spherical(int(sa.angular_momentum))
    Cb = cart_to_spherical(int(sb.angular_momentum))
    return Ca @ block_cart @ Cb.T


def shell_offsets(shells: Sequence[Shell], spherical: bool) -> List[int]:
    """Cumulative basis-function offsets per shell, for the chosen layout."""
    offsets = [0]
    for sh in shells:
        l = int(sh.angular_momentum)
        offsets.append(offsets[-1] + ((2 * l + 1) if spherical else n_cartesian(l)))
    return offsets


def compute_overlap_matrix(basis: BasisSet) -> jnp.ndarray:
    """Build the overlap matrix S (n_basis, n_basis) for a general basis."""
    spherical = bool(basis.spherical)
    shells = basis.shells
    n = basis.n_basis
    S = np.zeros((n, n), dtype=np.float64)
    offsets = shell_offsets(shells, spherical)
    for i, sa in enumerate(shells):
        ia0, ia1 = offsets[i], offsets[i + 1]
        for j, sb in enumerate(shells):
            jb0, jb1 = offsets[j], offsets[j + 1]
            if j < i:
                S[ia0:ia1, jb0:jb1] = S[jb0:jb1, ia0:ia1].T
                continue
            block = contracted_overlap_block(sa, sb)
            if spherical:
                block = _shell_block_spherical(block, sa, sb)
            S[ia0:ia1, jb0:jb1] = block
    return jnp.asarray(S)
