"""Initial-guess density matrices for SCF.

The core Hamiltonian guess diagonalizes X^T H_core X to get initial MOs
and builds the density from the lowest occupied orbitals. Good enough
for most molecules; for hard cases we will add SAD / minimal-basis
projection in later steps.

For UHF the symmetry breaking strategy is a "one-step alpha repulsion"
approach: the alpha density is built from the standard core Hamiltonian
guess, then the beta MOs are computed from a Fock-like operator that
includes the Coulomb repulsion from the alpha electrons. This naturally
breaks all spatial degeneracies that would trap the UHF at the
restricted solution.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from jax_qc.scf.density import density_rhf, density_uhf


def core_guess(H_core: jnp.ndarray, X: jnp.ndarray, n_occ: int) -> jnp.ndarray:
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


def core_guess_uhf(
    H_core: jnp.ndarray,
    X: jnp.ndarray,
    n_alpha: int,
    n_beta: int,
    ERI: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Core-Hamiltonian initial guess for UHF with symmetry breaking.

    FP: Applicative — pure function of H_core, X, and ERI.

    For open-shell systems (n_alpha != n_beta), the beta MOs are
    computed from a Fock matrix that includes the Coulomb + exchange
    repulsion from the alpha electrons. This physically motivated
    approach breaks spatial degeneracies: the beta electrons "see"
    the alpha distribution, giving them different preferred orbitals.

    For closed-shell systems (n_alpha == n_beta), both spin channels
    share the same core Hamiltonian MOs.

    Args:
        H_core:  (n, n) core Hamiltonian.
        X:       (n, n) orthogonalizer such that X^T S X = I.
        n_alpha: number of alpha electrons.
        n_beta:  number of beta electrons.
        ERI:     (n, n, n, n) two-electron integrals.

    Returns:
        (D_alpha, D_beta, C_alpha): initial per-spin densities and
        alpha MO coefficient matrix.
    """
    # Alpha: standard core Hamiltonian diagonalization
    F_prime = X.T @ H_core @ X
    _, C_prime = jnp.linalg.eigh(F_prime)
    C_alpha = X @ C_prime

    if n_alpha != n_beta:
        # Build alpha density from core guess
        C_occ_a = C_alpha[:, :n_alpha]
        D_alpha_init = C_occ_a @ C_occ_a.T

        # Build a one-electron effective potential for beta that includes
        # Coulomb + exchange repulsion from alpha electrons. This makes
        # beta "see" the alpha distribution, naturally breaking most
        # spatial degeneracies.
        J_alpha = jnp.einsum("ls,mnls->mn", D_alpha_init, ERI)
        K_alpha = jnp.einsum("ls,msln->mn", D_alpha_init, ERI)
        F_beta_eff = H_core + J_alpha - K_alpha

        # Add a deterministic perturbation to break any residual
        # degeneracies (e.g. pi_g pair in O2 where the alpha density
        # is symmetric and J/K don't break it). The perturbation is
        # scaled relative to the spread of the effective Fock eigenvalues
        # to adapt to different molecules.
        n = H_core.shape[0]
        F_prime_tmp = X.T @ F_beta_eff @ X
        evals_tmp = jnp.linalg.eigvalsh(F_prime_tmp)
        spread = jnp.max(evals_tmp) - jnp.min(evals_tmp)
        # Use ~1% of the eigenvalue spread as perturbation magnitude
        noise_scale = 0.01 * float(spread) + 0.1
        idx = jnp.arange(n, dtype=jnp.float64)
        noise = noise_scale * jnp.outer(
            jnp.sin(idx * 1.23 + 0.45), jnp.cos(idx * 0.67 + 0.89)
        )
        F_beta_eff = F_beta_eff + (noise + noise.T) / 2.0

        F_prime_b = X.T @ F_beta_eff @ X
        _, C_prime_b = jnp.linalg.eigh(F_prime_b)
        C_beta = X @ C_prime_b
    else:
        C_beta = C_alpha

    D_alpha, D_beta = density_uhf(C_alpha, C_beta, n_alpha, n_beta)
    return D_alpha, D_beta, C_alpha
