"""Pulay's Direct Inversion in the Iterative Subspace (DIIS).

DIIS accelerates SCF convergence by extrapolating a new Fock matrix as a
linear combination of the previous ``m`` Fock matrices, chosen to
minimize the norm of a commutator error vector. For RHF the standard
error vector is the orthogonalized commutator

    e = X^T (F D S - S D F) X.

``DIISHistory`` holds the ring of (F, e) pairs. ``diis_extrapolate``
returns the extrapolated Fock matrix and the updated history.

FP: The history is an immutable value; ``diis_extrapolate`` is a pure
function history -> (F_ext, history').
"""

from __future__ import annotations

from typing import Optional, Tuple

import chex
import jax.numpy as jnp
import numpy as np


@chex.dataclass(frozen=True)
class DIISHistory:
    """Rolling window of DIIS Fock/error pairs.

    fock_list:   (m, n, n) stack of the last m Fock matrices.
    error_list:  (m, n, n) stack of the corresponding error matrices.
    size:        current number of valid entries in the window (<= max_size).
    max_size:    ring-buffer capacity (``diis_space`` from CalcConfig).
    """

    fock_list: jnp.ndarray
    error_list: jnp.ndarray
    size: int
    max_size: int


def diis_history_init(n_basis: int, max_size: int) -> DIISHistory:
    """Create an empty DIIS history sized for an (n_basis x n_basis) Fock."""
    shape = (max_size, n_basis, n_basis)
    return DIISHistory(
        fock_list=jnp.zeros(shape),
        error_list=jnp.zeros(shape),
        size=0,
        max_size=max_size,
    )


def diis_error(
    F: jnp.ndarray, D: jnp.ndarray, S: jnp.ndarray, X: jnp.ndarray
) -> jnp.ndarray:
    """Pulay commutator error e = X^T (F D S - S D F) X."""
    FDS = F @ D @ S
    SDF = S @ D @ F
    return X.T @ (FDS - SDF) @ X


def _push(history: DIISHistory, F: jnp.ndarray, e: jnp.ndarray) -> DIISHistory:
    """Append (F, e) into the ring buffer.

    When the buffer is full the oldest entry is evicted (shift left).
    """
    if history.size < history.max_size:
        fock_list = history.fock_list.at[history.size].set(F)
        error_list = history.error_list.at[history.size].set(e)
        return history.replace(
            fock_list=fock_list,
            error_list=error_list,
            size=history.size + 1,
        )
    fock_list = jnp.concatenate([history.fock_list[1:], F[None]], axis=0)
    error_list = jnp.concatenate([history.error_list[1:], e[None]], axis=0)
    return history.replace(fock_list=fock_list, error_list=error_list)


def _solve_diis(errors: jnp.ndarray) -> Optional[jnp.ndarray]:
    """Solve the DIIS linear system in NumPy (not jit'd; size <= diis_space+1).

    Returns the extrapolation coefficients or ``None`` if the system is
    singular (caller should fall back to the latest Fock).
    """
    e = np.asarray(errors)
    m = e.shape[0]
    B = np.empty((m + 1, m + 1), dtype=e.dtype)
    for i in range(m):
        for j in range(m):
            B[i, j] = float(np.sum(e[i] * e[j]))
    B[-1, :-1] = -1.0
    B[:-1, -1] = -1.0
    B[-1, -1] = 0.0
    rhs = np.zeros(m + 1, dtype=e.dtype)
    rhs[-1] = -1.0
    try:
        sol = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        return None
    return sol[:-1]


def diis_extrapolate(
    history: DIISHistory,
    F: jnp.ndarray,
    D: jnp.ndarray,
    S: jnp.ndarray,
    X: jnp.ndarray,
) -> Tuple[jnp.ndarray, DIISHistory, jnp.ndarray]:
    """Push (F, error) and return an extrapolated Fock.

    Returns:
        F_ext:   extrapolated Fock matrix (same shape as F).
        history: updated history with the new pair appended.
        error:   the error matrix that was just stored (used for
                 convergence diagnostics; ``max(abs(error))`` is a common
                 SCF convergence criterion).
    """
    e = diis_error(F, D, S, X)
    history = _push(history, F, e)
    if history.size < 2:
        return F, history, e
    errors = history.error_list[: history.size]
    focks = history.fock_list[: history.size]
    coeffs = _solve_diis(errors)
    if coeffs is None:
        return F, history, e
    F_ext = jnp.einsum("i,imn->mn", jnp.asarray(coeffs), focks)
    return F_ext, history, e
