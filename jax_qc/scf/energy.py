"""RHF electronic energy.

For closed-shell RHF the electronic energy reduces to

    E_elec = 1/2 Tr[D (H_core + F)].

FP: Foldable — a reduction over matrix elements.
"""

from __future__ import annotations

import jax.numpy as jnp


def electronic_energy_rhf(
    D: jnp.ndarray, H_core: jnp.ndarray, F: jnp.ndarray
) -> jnp.ndarray:
    """Return the RHF electronic energy (0-d JAX array)."""
    return 0.5 * jnp.sum(D * (H_core + F))
