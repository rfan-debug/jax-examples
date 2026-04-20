"""Density matrix for closed-shell RHF.

For a closed-shell system with ``n_occ`` doubly-occupied orbitals:

    D_{mu nu} = 2 * sum_{i=0}^{n_occ-1} C_{mu i} C_{nu i}.

FP: Applicative — pure function of C.
"""

from __future__ import annotations

import jax.numpy as jnp


def density_rhf(C: jnp.ndarray, n_occ: int) -> jnp.ndarray:
    """Build the RHF density matrix from MO coefficients.

    Args:
        C:     (n_basis, n_mo) MO coefficient matrix.
        n_occ: number of doubly-occupied orbitals (= n_electrons // 2).

    Returns:
        (n_basis, n_basis) density matrix with trace = n_electrons.
    """
    C_occ = C[:, :n_occ]
    return 2.0 * C_occ @ C_occ.T
