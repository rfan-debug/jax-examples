"""Electronic energy for RHF and UHF.

**RHF**: E_elec = 1/2 Tr[D (H_core + F)].

**UHF**: E_elec = 1/2 Tr[D_alpha (H_core + F_alpha)]
               + 1/2 Tr[D_beta  (H_core + F_beta)].

FP: Foldable — reductions over matrix elements.
"""

from __future__ import annotations

import jax.numpy as jnp


def electronic_energy_rhf(
    D: jnp.ndarray, H_core: jnp.ndarray, F: jnp.ndarray
) -> jnp.ndarray:
    """Return the RHF electronic energy (0-d JAX array)."""
    return 0.5 * jnp.sum(D * (H_core + F))


def electronic_energy_uhf(
    D_alpha: jnp.ndarray,
    D_beta: jnp.ndarray,
    H_core: jnp.ndarray,
    F_alpha: jnp.ndarray,
    F_beta: jnp.ndarray,
) -> jnp.ndarray:
    """Return the UHF electronic energy (0-d JAX array).

    FP: Foldable — independent trace reductions per spin channel.

    E_elec = 1/2 [ Tr(D_a (H + F_a)) + Tr(D_b (H + F_b)) ]
    """
    E_a = 0.5 * jnp.sum(D_alpha * (H_core + F_alpha))
    E_b = 0.5 * jnp.sum(D_beta * (H_core + F_beta))
    return E_a + E_b
