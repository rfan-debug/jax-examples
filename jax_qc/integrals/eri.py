"""Two-electron repulsion integrals.

(mu nu | lambda sigma) = <chi_mu(1) chi_nu(1) | 1/r_12 | chi_lambda(2) chi_sigma(2)>

Step 2: s-type (l = 0) shells only. Primitive (ss|ss) integral:

    (ab|cd) = 2 pi^(5/2) / (p q sqrt(p + q))
              * exp(-mu_p |A-B|^2) * exp(-mu_q |C-D|^2)
              * F_0(rho |P-Q|^2)

with
    p     = alpha + beta
    q     = gamma + delta
    mu_p  = alpha beta / p
    mu_q  = gamma delta / q
    rho   = p q / (p + q)
    P     = (alpha A + beta B) / p
    Q     = (gamma C + delta D) / q

8-fold permutational symmetry (real basis):
    (mu nu | la si) = (nu mu | la si) = (mu nu | si la) = (nu mu | si la)
                    = (la si | mu nu) = (si la | mu nu) = (la si | nu mu)
                    = (si la | nu mu)
The tensor we return respects this automatically because the underlying
formula is symmetric under swaps of (mu, nu), (lambda, sigma), and bra/ket.

FP: Applicative — every quartet is independent.
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import BasisSet, Shell
from jax_qc.integrals.boys import boys_f0
from jax_qc.integrals.gaussian_product import distance_squared
from jax_qc.integrals.overlap import _assert_all_s


_PREFACTOR = 2.0 * jnp.pi ** 2.5  # 2 pi^(5/2)


def eri_primitive_ssss(
    alpha, A, beta, B, gamma, C, delta, D
):
    """Primitive (ss|ss) ERI. All inputs broadcast."""
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
    """Contracted (ss|ss) ERI for one shell quartet.

    Vectorized over the 4-d grid of primitive tuples. This scales as
    Na*Nb*Nc*Nd memory-wise; for STO-3G / 6-31G this is <= 81 entries.
    """
    # Shape (Na, Nb, Nc, Nd) for every broadcasted tensor.
    alpha = sa.exponents[:, None, None, None]
    beta = sb.exponents[None, :, None, None]
    gamma = sc.exponents[None, None, :, None]
    delta = sd.exponents[None, None, None, :]

    p = alpha + beta
    q = gamma + delta
    mu_p = alpha * beta / p
    mu_q = gamma * delta / q

    A = sa.center
    B = sb.center
    C = sc.center
    D = sd.center

    AB2 = distance_squared(A, B)
    CD2 = distance_squared(C, D)

    # Broadcast centers to (Na, Nb, Nc, Nd, 3).
    P_vec = (alpha[..., None] * A + beta[..., None] * B) / p[..., None]
    Q_vec = (gamma[..., None] * C + delta[..., None] * D) / q[..., None]
    PQ2 = jnp.sum((P_vec - Q_vec) ** 2, axis=-1)

    rho = p * q / (p + q)
    prim = (
        _PREFACTOR
        / (p * q * jnp.sqrt(p + q))
        * jnp.exp(-mu_p * AB2 - mu_q * CD2)
        * boys_f0(rho * PQ2)
    )
    ca = sa.coefficients[:, None, None, None]
    cb = sb.coefficients[None, :, None, None]
    cc = sc.coefficients[None, None, :, None]
    cd = sd.coefficients[None, None, None, :]
    return jnp.sum(ca * cb * cc * cd * prim)


def compute_eri_tensor(basis: BasisSet) -> jnp.ndarray:
    """Build the 4-index ERI tensor (n_basis,)*4 for an s-only basis.

    Uses the 8-fold symmetry (mu,nu,lambda,sigma): compute the unique
    canonical quartets with mu <= nu, lambda <= sigma, (mu,nu) <=
    (lambda,sigma) compound index, then scatter to all symmetric slots.
    """
    _assert_all_s(basis.shells)
    n = basis.n_basis
    eri = np.zeros((n, n, n, n), dtype=np.float64)
    shells = basis.shells
    for mu in range(n):
        for nu in range(mu + 1):
            for la in range(n):
                for si in range(la + 1):
                    if _compound(mu, nu) < _compound(la, si):
                        continue
                    val = float(
                        eri_shell_quartet_ssss(
                            shells[mu], shells[nu], shells[la], shells[si]
                        )
                    )
                    _assign_8fold(eri, mu, nu, la, si, val)
    return jnp.asarray(eri)


def _compound(i: int, j: int) -> int:
    """Compound index used for ordering in the 8-fold unique scan."""
    if i >= j:
        return i * (i + 1) // 2 + j
    return j * (j + 1) // 2 + i


def _assign_8fold(eri, mu, nu, la, si, val) -> None:
    for (a, b, c, d) in (
        (mu, nu, la, si),
        (nu, mu, la, si),
        (mu, nu, si, la),
        (nu, mu, si, la),
        (la, si, mu, nu),
        (si, la, mu, nu),
        (la, si, nu, mu),
        (si, la, nu, mu),
    ):
        eri[a, b, c, d] = val
