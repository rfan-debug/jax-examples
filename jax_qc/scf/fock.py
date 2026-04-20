"""RHF Fock matrix assembly.

Given the density D and the two-electron integrals (mu nu | lambda sigma)
stored in chemist's notation, the Coulomb and exchange contributions are

    J_{mu nu} = sum_{lambda sigma} D_{lambda sigma} (mu nu | lambda sigma)
    K_{mu nu} = sum_{lambda sigma} D_{lambda sigma} (mu sigma | lambda nu)

and the closed-shell Fock matrix is

    F = H_core + J - 1/2 K.

FP: Applicative — a single ``einsum`` over independent index quadruples.
"""

from __future__ import annotations

import jax.numpy as jnp


def build_fock_rhf(
    H_core: jnp.ndarray, D: jnp.ndarray, ERI: jnp.ndarray
) -> jnp.ndarray:
    """Assemble the closed-shell Fock matrix F = H + J - 1/2 K."""
    J = jnp.einsum("ls,mnls->mn", D, ERI)
    K = jnp.einsum("ls,msln->mn", D, ERI)
    return H_core + J - 0.5 * K
