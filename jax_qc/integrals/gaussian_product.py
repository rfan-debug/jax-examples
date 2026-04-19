"""Gaussian product theorem utilities.

The product of two spherical Gaussians centered on A and B reduces to a
single Gaussian on the centroid P plus a scalar pre-factor:

    exp(-alpha |r - A|^2) * exp(-beta |r - B|^2)
        = exp(-mu |A - B|^2) * exp(-p |r - P|^2)

where

    p  = alpha + beta          (product exponent)
    mu = alpha * beta / p      (reduced exponent)
    P  = (alpha * A + beta * B) / p

Everything here is a pure, jit-compatible JAX operation.
"""

from __future__ import annotations

import jax.numpy as jnp


def distance_squared(A, B):
    """Return |A - B|^2. Works on arrays of shape (..., 3)."""
    diff = A - B
    return jnp.sum(diff * diff, axis=-1)


def gaussian_product_exponent(alpha, beta):
    """Return the combined exponent p = alpha + beta."""
    return alpha + beta


def gaussian_product_center(alpha, A, beta, B):
    """Return the centroid P = (alpha A + beta B) / (alpha + beta).

    ``alpha`` and ``beta`` may be scalars or broadcastable to the leading
    dimensions of ``A`` / ``B``; ``A`` and ``B`` have shape (..., 3).
    """
    p = alpha + beta
    # Add a trailing axis on the exponents so shape broadcasts with coord.
    alpha_b = jnp.asarray(alpha)[..., None]
    beta_b = jnp.asarray(beta)[..., None]
    p_b = jnp.asarray(p)[..., None]
    return (alpha_b * A + beta_b * B) / p_b
