"""Fock-matrix damping (simple linear mixing).

Damping stabilizes SCF when the density oscillates. We blend the new
Fock with the previous one:

    F_mixed = (1 - alpha) F_new + alpha F_old.

alpha in [0, 1); alpha = 0 disables damping.
"""

from __future__ import annotations

import jax.numpy as jnp


def damp(
    F_new: jnp.ndarray, F_old: jnp.ndarray, alpha: float
) -> jnp.ndarray:
    """Return (1 - alpha) F_new + alpha F_old."""
    return (1.0 - alpha) * F_new + alpha * F_old
