"""Two-electron repulsion integrals (mu nu | lambda sigma).

For arbitrary angular momentum we evaluate the primitive Cartesian
(ab|cd) block via :mod:`jax_qc.integrals.obara_saika` (McMurchie-Davidson
expansion in Hermite Gaussians plus the auxiliary R_{tuv}^n recurrence).
We exploit 8-fold permutational symmetry on the *shell* level: only
canonical quartets (i<=j, k<=l, compound(i,j) >= compound(k,l)) are
computed and the resulting block is scattered to all 8 symmetric
positions in the final tensor.
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import BasisSet, Shell
from jax_qc.integrals.boys import boys_f0
from jax_qc.integrals.gaussian_product import distance_squared
from jax_qc.integrals.obara_saika import (
    contracted_eri_block,
    n_cartesian,
)
from jax_qc.integrals.overlap import _assert_all_s, shell_offsets
from jax_qc.integrals.spherical import cart_to_spherical


_PREFACTOR = 2.0 * jnp.pi ** 2.5  # 2 pi^(5/2)


def eri_primitive_ssss(alpha, A, beta, B, gamma, C, delta, D):
    """Analytic primitive (ss|ss). Convenience for tests."""
    p = alpha + beta
    q = gamma + delta
    mu_p = alpha * beta / p
    mu_q = gamma * delta / q
    AB2 = distance_squared(A, B)
    CD2 = distance_squared(C, D)
    P = (alpha[..., None] * A + beta[..., None] * B) / p[..., None]
    Q = (gamma[..., None] * C + delta[..., None] * D) / q[..., None]
    PQ2 = jnp.sum((P - Q) ** 2, axis=-1)
    rho = p * q / (p + q)
    return (
        _PREFACTOR
        / (p * q * jnp.sqrt(p + q))
        * jnp.exp(-mu_p * AB2 - mu_q * CD2)
        * boys_f0(rho * PQ2)
    )


def eri_shell_quartet_ssss(
    sa: Shell, sb: Shell, sc: Shell, sd: Shell
) -> jnp.ndarray:
    """Convenience wrapper for the s-only quartet."""
    block = contracted_eri_block(sa, sb, sc, sd)
    return jnp.asarray(block[0, 0, 0, 0])


def _spherical_eri_block(block, sa, sb, sc, sd):
    Ca = cart_to_spherical(int(sa.angular_momentum))
    Cb = cart_to_spherical(int(sb.angular_momentum))
    Cc = cart_to_spherical(int(sc.angular_momentum))
    Cd = cart_to_spherical(int(sd.angular_momentum))
    return np.einsum("Ii,Jj,Kk,Ll,ijkl->IJKL", Ca, Cb, Cc, Cd, block)


def compute_eri_tensor(basis: BasisSet) -> jnp.ndarray:
    """Build the 4-index ERI tensor for a general basis.

    Iterates over canonical shell quartets (8-fold redundancy removed),
    computes one Cartesian block per canonical quartet, optionally
    transforms to spherical, then scatters into all 8 permutations of
    the final tensor.
    """
    spherical = bool(basis.spherical)
    shells = basis.shells
    n = basis.n_basis
    eri = np.zeros((n, n, n, n), dtype=np.float64)
    offsets = shell_offsets(shells, spherical)
    n_shells = len(shells)

    def _shell_pair_idx(i: int, j: int) -> int:
        if i >= j:
            return i * (i + 1) // 2 + j
        return j * (j + 1) // 2 + i

    for i in range(n_shells):
        for j in range(i + 1):
            ij = _shell_pair_idx(i, j)
            for k in range(n_shells):
                for l in range(k + 1):
                    kl = _shell_pair_idx(k, l)
                    if ij < kl:
                        continue
                    block = contracted_eri_block(
                        shells[i], shells[j], shells[k], shells[l]
                    )
                    if spherical:
                        block = _spherical_eri_block(
                            block, shells[i], shells[j], shells[k], shells[l]
                        )
                    _scatter_8fold(eri, block, offsets, i, j, k, l)
    return jnp.asarray(eri)


def _scatter_8fold(
    eri: np.ndarray,
    block: np.ndarray,
    offsets: List[int],
    i: int,
    j: int,
    k: int,
    l: int,
) -> None:
    """Place ``block`` at all 8 permutations of (i, j, k, l) in ``eri``."""
    si = slice(offsets[i], offsets[i + 1])
    sj = slice(offsets[j], offsets[j + 1])
    sk = slice(offsets[k], offsets[k + 1])
    sl = slice(offsets[l], offsets[l + 1])

    eri[si, sj, sk, sl] = block
    if i != j:
        eri[sj, si, sk, sl] = block.transpose(1, 0, 2, 3)
    if k != l:
        eri[si, sj, sl, sk] = block.transpose(0, 1, 3, 2)
    if i != j and k != l:
        eri[sj, si, sl, sk] = block.transpose(1, 0, 3, 2)

    if (i, j) != (k, l):
        eri[sk, sl, si, sj] = block.transpose(2, 3, 0, 1)
        if i != j:
            eri[sk, sl, sj, si] = block.transpose(2, 3, 1, 0)
        if k != l:
            eri[sl, sk, si, sj] = block.transpose(3, 2, 0, 1)
        if i != j and k != l:
            eri[sl, sk, sj, si] = block.transpose(3, 2, 1, 0)
