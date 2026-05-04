"""Fock matrix assembly for RHF and UHF.

Given the density D and the two-electron integrals (mu nu | lambda sigma)
stored in chemist's notation, the Coulomb and exchange contributions are

    J_{mu nu} = sum_{lambda sigma} D_{lambda sigma} (mu nu | lambda sigma)
    K_{mu nu} = sum_{lambda sigma} D_{lambda sigma} (mu sigma | lambda nu)

**RHF** (closed-shell): F = H_core + J(D) - 1/2 K(D), where D is total.

**UHF** (open-shell): two Fock matrices built from total and per-spin
densities:

    D_total = D_alpha + D_beta
    F_alpha = H_core + J(D_total) - K(D_alpha)
    F_beta  = H_core + J(D_total) - K(D_beta)

Note: UHF exchange uses the *full* K (no 1/2 factor) because each
spin channel's density already counts each electron once.

FP: Applicative — independent ``einsum`` calls over index quadruples.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def _coulomb(D: jnp.ndarray, ERI: jnp.ndarray) -> jnp.ndarray:
    """Coulomb matrix J_{mn} = sum_{ls} D_{ls} (mn|ls)."""
    return jnp.einsum("ls,mnls->mn", D, ERI)


def _exchange(D: jnp.ndarray, ERI: jnp.ndarray) -> jnp.ndarray:
    """Exchange matrix K_{mn} = sum_{ls} D_{ls} (ms|ln)."""
    return jnp.einsum("ls,msln->mn", D, ERI)


def build_fock_rhf(
    H_core: jnp.ndarray, D: jnp.ndarray, ERI: jnp.ndarray
) -> jnp.ndarray:
    """Assemble the closed-shell Fock matrix F = H + J - 1/2 K."""
    J = _coulomb(D, ERI)
    K = _exchange(D, ERI)
    return H_core + J - 0.5 * K


def build_fock_uhf(
    H_core: jnp.ndarray,
    D_alpha: jnp.ndarray,
    D_beta: jnp.ndarray,
    ERI: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Assemble the UHF Fock matrices (F_alpha, F_beta).

    FP: Applicative — pure function of densities and integrals.

    The Coulomb term uses the total density D_total = D_alpha + D_beta.
    The exchange term is spin-specific: K_alpha from D_alpha, K_beta
    from D_beta, each with a full (no 1/2) prefactor.

    Args:
        H_core:  (n, n) core Hamiltonian.
        D_alpha: (n, n) alpha spin density matrix.
        D_beta:  (n, n) beta spin density matrix.
        ERI:     (n, n, n, n) two-electron integrals in chemist's notation.

    Returns:
        (F_alpha, F_beta) tuple of (n, n) Fock matrices.
    """
    D_total = D_alpha + D_beta
    J = _coulomb(D_total, ERI)
    K_a = _exchange(D_alpha, ERI)
    K_b = _exchange(D_beta, ERI)
    F_alpha = H_core + J - K_a
    F_beta = H_core + J - K_b
    return F_alpha, F_beta
