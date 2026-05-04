"""Density matrices for RHF and UHF.

**RHF**: For a closed-shell system with ``n_occ`` doubly-occupied orbitals:

    D_{mu nu} = 2 * sum_{i=0}^{n_occ-1} C_{mu i} C_{nu i}.

**UHF**: Each spin channel has its own density matrix built from its
own occupied MO coefficients:

    D_alpha_{mu nu} = sum_{i=0}^{n_alpha-1} C_alpha_{mu i} C_alpha_{nu i}
    D_beta_{mu nu}  = sum_{i=0}^{n_beta-1}  C_beta_{mu i}  C_beta_{nu i}

FP: Applicative — pure functions of MO coefficients.
"""

from __future__ import annotations

from typing import Tuple

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


def density_uhf(
    C_alpha: jnp.ndarray, C_beta: jnp.ndarray, n_alpha: int, n_beta: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build UHF per-spin density matrices.

    FP: Applicative — pure function of C_alpha, C_beta.

    Args:
        C_alpha: (n_basis, n_mo) alpha MO coefficient matrix.
        C_beta:  (n_basis, n_mo) beta MO coefficient matrix.
        n_alpha: number of alpha electrons.
        n_beta:  number of beta electrons.

    Returns:
        (D_alpha, D_beta): per-spin density matrices.
        Tr(D_alpha @ S) = n_alpha, Tr(D_beta @ S) = n_beta.
    """
    C_occ_a = C_alpha[:, :n_alpha]
    C_occ_b = C_beta[:, :n_beta]
    D_alpha = C_occ_a @ C_occ_a.T
    D_beta = C_occ_b @ C_occ_b.T
    return D_alpha, D_beta
