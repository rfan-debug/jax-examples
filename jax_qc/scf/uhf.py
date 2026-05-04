"""Unrestricted Hartree-Fock SCF driver.

FP classification:

* ``scf_step_uhf`` — pure State transition ``UHFState -> UHFState``.
  Each call rebuilds both alpha and beta Fock matrices, applies optional
  damping and combined DIIS extrapolation, diagonalizes, and returns
  the new per-spin densities / coefficients / energy.  This is a
  single step of the State Monad.

* ``run_uhf`` — the Monadic outer loop.  It threads the UHFState through
  ``scf_step_uhf`` until the energy change and DIIS error fall below
  tolerance or ``max_iter`` is reached.

Sub-computations (``build_fock_uhf``, ``density_uhf``,
``electronic_energy_uhf``) stay Applicative and live in their own modules.

UHF maintains two independent spin channels (alpha and beta) each with
its own Fock matrix, MO coefficients, density matrix, and orbital
energies.  The Coulomb interaction couples them through the total
density D_total = D_alpha + D_beta.

DIIS uses a combined approach: both spin channels are stacked into a
single "super-Fock" for the DIIS solve, ensuring consistent
extrapolation across alpha and beta.
"""

from __future__ import annotations

from typing import Optional, Tuple

import chex
import jax.numpy as jnp
import numpy as np

from jax_qc.core.types import (
    CalcConfig,
    IntegralSet,
    Molecule,
    SCFResult,
    SCFState,
)
from jax_qc.profiling.timer import StageTimer, optional_stage
from jax_qc.scf.damping import damp
from jax_qc.scf.density import density_uhf
from jax_qc.scf.diis import (
    DIISHistory,
    diis_extrapolate_uhf,
    diis_history_init_uhf,
)
from jax_qc.scf.energy import electronic_energy_uhf
from jax_qc.scf.fock import build_fock_uhf
from jax_qc.scf.guess import core_guess_uhf
from jax_qc.scf.orthogonalize import symmetric_orthogonalization


# ---------------------------------------------------------------------------
#  UHF-specific immutable state
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class UHFState:
    """Immutable UHF SCF iteration state.

    This is the 's' in State Monad: step :: s -> (s, a).
    Each spin channel has its own density, Fock matrix, MO coefficients,
    and orbital energies.
    """

    density_alpha: jnp.ndarray
    density_beta: jnp.ndarray
    fock_alpha: jnp.ndarray
    fock_beta: jnp.ndarray
    coefficients_alpha: jnp.ndarray
    coefficients_beta: jnp.ndarray
    orbital_energies_alpha: jnp.ndarray
    orbital_energies_beta: jnp.ndarray
    energy: float
    iteration: int


# ---------------------------------------------------------------------------
#  Helper: Fock diagonalization (same as RHF, used for each spin channel)
# ---------------------------------------------------------------------------


def _diagonalize_fock(
    F: jnp.ndarray, X: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transform F -> F' = X^T F X, diagonalize, back-transform MOs."""
    F_prime = X.T @ F @ X
    evals, C_prime = jnp.linalg.eigh(F_prime)
    C = X @ C_prime
    return evals, C


# ---------------------------------------------------------------------------
#  Single SCF step
# ---------------------------------------------------------------------------


def scf_step_uhf(
    state: UHFState,
    H_core: jnp.ndarray,
    ERI: jnp.ndarray,
    S: jnp.ndarray,
    X: jnp.ndarray,
    n_alpha: int,
    n_beta: int,
    diis_history: DIISHistory,
    damping: float,
    timer: Optional[StageTimer] = None,
) -> Tuple[UHFState, DIISHistory, jnp.ndarray]:
    """One UHF SCF iteration. Pure State -> State (ignoring the timer).

    Uses combined DIIS: both spin channels share a single DIIS history
    and solve, preventing the independent-channel oscillation that
    plagues naive dual-DIIS implementations.

    Returns:
        new_state:    UHFState after this iteration.
        diis_history: updated combined DIIS ring buffer.
        error:        combined DIIS error (max of alpha and beta).
    """
    # --- Fock build (Applicative) ---
    with optional_stage(timer, "fock_build", "applicative"):
        F_a_new, F_b_new = build_fock_uhf(
            H_core, state.density_alpha, state.density_beta, ERI
        )

    # --- Damping ---
    with optional_stage(timer, "damp", "applicative"):
        if damping > 0.0:
            F_a_new = damp(F_a_new, state.fock_alpha, damping)
            F_b_new = damp(F_b_new, state.fock_beta, damping)

    # --- Combined DIIS ---
    with optional_stage(timer, "diis", "applicative"):
        F_a_ext, F_b_ext, diis_history, error_a, error_b = diis_extrapolate_uhf(
            diis_history,
            F_a_new,
            F_b_new,
            state.density_alpha,
            state.density_beta,
            S,
            X,
        )

    # --- Diagonalize each spin channel ---
    with optional_stage(timer, "diag", "applicative"):
        orb_e_a, C_a = _diagonalize_fock(F_a_ext, X)
        orb_e_b, C_b = _diagonalize_fock(F_b_ext, X)

    # --- Build per-spin density matrices ---
    with optional_stage(timer, "density", "applicative"):
        D_a, D_b = density_uhf(C_a, C_b, n_alpha, n_beta)

    # --- Electronic energy ---
    with optional_stage(timer, "energy", "foldable"):
        E_elec = electronic_energy_uhf(D_a, D_b, H_core, F_a_ext, F_b_ext)

    # --- Combined error for convergence ---
    error = jnp.maximum(jnp.max(jnp.abs(error_a)), jnp.max(jnp.abs(error_b)))
    error_combined = jnp.array([[error]])

    new_state = UHFState(
        density_alpha=D_a,
        density_beta=D_b,
        fock_alpha=F_a_ext,
        fock_beta=F_b_ext,
        coefficients_alpha=C_a,
        coefficients_beta=C_b,
        orbital_energies_alpha=orb_e_a,
        orbital_energies_beta=orb_e_b,
        energy=float(E_elec),
        iteration=state.iteration + 1,
    )
    return new_state, diis_history, error_combined


# ---------------------------------------------------------------------------
#  UHF main driver
# ---------------------------------------------------------------------------


def run_uhf(
    mol: Molecule,
    ints: IntegralSet,
    config: CalcConfig,
    timer: Optional[StageTimer] = None,
) -> SCFResult:
    """Run UHF to self-consistency.

    FP: Monadic loop over ``scf_step_uhf``. The loop is written in
    Python rather than ``lax.while_loop`` so it can hold the mutable
    timer and the NumPy DIIS solve. Inside each step, every
    sub-computation is Applicative and fully jit-able.

    The result is packaged into an ``SCFResult`` for consistency with
    ``run_rhf``. The ``state`` field contains the *total* density
    (D_alpha + D_beta) and the alpha MO coefficients / energies.
    """
    n_alpha = mol.n_alpha
    n_beta = mol.n_beta

    with optional_stage(timer, "scf", "monad"):
        # --- Orthogonalization ---
        with optional_stage(timer, "orthogonalize", "applicative"):
            X = symmetric_orthogonalization(ints.S)

        # --- Initial guess (alpha-repulsion symmetry breaking) ---
        with optional_stage(timer, "initial_guess", "applicative"):
            D_a0, D_b0, C0 = core_guess_uhf(ints.H_core, X, n_alpha, n_beta, ints.ERI)
            F_a0, F_b0 = build_fock_uhf(ints.H_core, D_a0, D_b0, ints.ERI)
            orb_e_a0, C_a0 = _diagonalize_fock(F_a0, X)
            orb_e_b0, C_b0 = _diagonalize_fock(F_b0, X)
            D_a0, D_b0 = density_uhf(C_a0, C_b0, n_alpha, n_beta)
            E0 = float(electronic_energy_uhf(D_a0, D_b0, ints.H_core, F_a0, F_b0))

        state = UHFState(
            density_alpha=D_a0,
            density_beta=D_b0,
            fock_alpha=F_a0,
            fock_beta=F_b0,
            coefficients_alpha=C_a0,
            coefficients_beta=C_b0,
            orbital_energies_alpha=orb_e_a0,
            orbital_energies_beta=orb_e_b0,
            energy=E0,
            iteration=0,
        )
        diis_history = diis_history_init_uhf(ints.S.shape[0], config.diis_space)

        converged = False
        last_energy = state.energy
        # Use damping for the first few iterations to stabilize the
        # SCF before DIIS has enough vectors to extrapolate reliably.
        # After _DAMP_ITERS, switch to user-specified damping (default 0).
        _DAMP_ITERS = 5
        _INIT_DAMP = 0.3 if n_alpha != n_beta else 0.0
        for it in range(config.max_scf_iter):
            damp_val = _INIT_DAMP if it < _DAMP_ITERS else config.damping
            state, diis_history, error = scf_step_uhf(
                state,
                ints.H_core,
                ints.ERI,
                ints.S,
                X,
                n_alpha,
                n_beta,
                diis_history,
                damping=damp_val,
                timer=timer,
            )
            dE = abs(state.energy - last_energy)
            err_max = float(jnp.max(jnp.abs(error)))
            last_energy = state.energy
            if dE < config.scf_conv and err_max < max(config.scf_conv, 1e-6):
                converged = True
                break

    # --- Package result ---
    total_density = state.density_alpha + state.density_beta
    total = state.energy + float(ints.E_nuc)

    # Wrap into SCFState (total density, alpha coefficients) for compatibility.
    scf_state = SCFState(
        density=total_density,
        fock=state.fock_alpha,
        coefficients=state.coefficients_alpha,
        orbital_energies=state.orbital_energies_alpha,
        energy=state.energy,
        iteration=state.iteration,
    )

    return SCFResult(
        converged=converged,
        state=scf_state,
        energy=total,
        E_elec=state.energy,
        E_nuc=float(ints.E_nuc),
        n_iterations=state.iteration,
        S=ints.S,
        H_core=ints.H_core,
        ERI=ints.ERI,
    )
