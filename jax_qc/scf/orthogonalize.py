"""Basis orthogonalization transforms.

For Roothaan-Hall we need a matrix X such that X^T S X = I; the
eigenvectors of the Fock matrix in the X basis then give the MO
coefficients directly.

Two variants:

* Symmetric (Lowdin): X = S^{-1/2}.
* Canonical: X = U s^{-1/2} where S = U s U^T, optionally dropping
  eigenvalues smaller than ``eps`` (useful when the basis has near
  linear dependencies).

FP: Applicative — both are pure functions of S.
"""

from __future__ import annotations

import jax.numpy as jnp


def symmetric_orthogonalization(S: jnp.ndarray) -> jnp.ndarray:
    """Return X = S^{-1/2} (Lowdin)."""
    evals, evecs = jnp.linalg.eigh(S)
    inv_sqrt = jnp.diag(1.0 / jnp.sqrt(evals))
    return evecs @ inv_sqrt @ evecs.T


def canonical_orthogonalization(
    S: jnp.ndarray, eps: float = 1e-7
) -> jnp.ndarray:
    """Return X = U s^{-1/2} with eigenvalues below ``eps`` dropped.

    The output may be rectangular (n_basis, n_kept) when the basis is
    linearly dependent.
    """
    evals, evecs = jnp.linalg.eigh(S)
    mask = evals > eps
    kept_vals = evals[mask]
    kept_vecs = evecs[:, mask]
    return kept_vecs / jnp.sqrt(kept_vals)[None, :]
