"""Initial-guess density matrices for SCF.

Only the core Hamiltonian guess is implemented here — diagonalize
X^T H_core X to get initial MOs and build the density from the lowest
``n_occ`` orbitals. Good enough for Tier 1 molecules; for hard cases we
will add SAD / minimal-basis projection in later steps.
"""

from __future__ import annotations

import jax.numpy as jnp

from jax_qc.scf.density import density_rhf


def core_guess(
    H_core: jnp.ndarray, X: jnp.ndarray, n_occ: int
) -> jnp.ndarray:
    """Core-Hamiltonian initial density for RHF.

    Args:
        H_core: core Hamiltonian (n_basis, n_basis).
        X:      orthogonalizer such that X^T S X = I.
        n_occ:  number of doubly-occupied orbitals.

    Returns:
        (n_basis, n_basis) initial density matrix.
    """
    F_prime = X.T @ H_core @ X
    _, C_prime = jnp.linalg.eigh(F_prime)
    C = X @ C_prime
    return density_rhf(C, n_occ)
