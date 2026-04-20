"""Restricted Hartree-Fock SCF driver.

FP classification:

* ``scf_step_rhf`` — pure State transition ``SCFState -> SCFState``.
  Each call rebuilds the Fock matrix, applies optional damping and
  DIIS extrapolation, diagonalizes, and returns the new density /
  coefficients / energy. This is a single step of the State Monad.

* ``run_rhf`` — the Monadic outer loop. It threads the SCFState through
  ``scf_step_rhf`` until the energy change and DIIS error fall below
  tolerance or ``max_iter`` is reached.

Sub-computations (``build_fock_rhf``, ``density_rhf``,
``electronic_energy_rhf``) stay Applicative and live in their own modules.
"""

from __future__ import annotations

from typing import Optional, Tuple

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
from jax_qc.scf.density import density_rhf
from jax_qc.scf.diis import DIISHistory, diis_extrapolate, diis_history_init
from jax_qc.scf.energy import electronic_energy_rhf
from jax_qc.scf.fock import build_fock_rhf
from jax_qc.scf.guess import core_guess
from jax_qc.scf.orthogonalize import symmetric_orthogonalization


def _diagonalize_fock(
    F: jnp.ndarray, X: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transform F -> F' = X^T F X, diagonalize, back-transform MOs."""
    F_prime = X.T @ F @ X
    evals, C_prime = jnp.linalg.eigh(F_prime)
    C = X @ C_prime
    return evals, C


def scf_step_rhf(
    state: SCFState,
    H_core: jnp.ndarray,
    ERI: jnp.ndarray,
    S: jnp.ndarray,
    X: jnp.ndarray,
    n_occ: int,
    diis_history: DIISHistory,
    damping: float,
    timer: Optional[StageTimer] = None,
) -> Tuple[SCFState, DIISHistory, jnp.ndarray]:
    """One RHF SCF iteration. Pure State -> State (ignoring the timer).

    Returns:
        new_state:    SCFState after this iteration.
        diis_history: updated DIIS ring buffer.
        error:        DIIS error matrix (for convergence diagnostics).
    """
    with optional_stage(timer, "fock_build", "applicative"):
        F_new = build_fock_rhf(H_core, state.density, ERI)
    with optional_stage(timer, "damp", "applicative"):
        F_damped = damp(F_new, state.fock, damping) if damping > 0.0 else F_new
    with optional_stage(timer, "diis", "applicative"):
        F_ext, diis_history, error = diis_extrapolate(
            diis_history, F_damped, state.density, S, X
        )
    with optional_stage(timer, "diag", "applicative"):
        orb_e, C = _diagonalize_fock(F_ext, X)
    with optional_stage(timer, "density", "applicative"):
        D_new = density_rhf(C, n_occ)
    with optional_stage(timer, "energy", "foldable"):
        E_elec = electronic_energy_rhf(D_new, H_core, F_ext)
    new_state = SCFState(
        density=D_new,
        fock=F_ext,
        coefficients=C,
        orbital_energies=orb_e,
        energy=float(E_elec),
        iteration=state.iteration + 1,
    )
    return new_state, diis_history, error


def run_rhf(
    mol: Molecule,
    ints: IntegralSet,
    config: CalcConfig,
    timer: Optional[StageTimer] = None,
) -> SCFResult:
    """Run RHF to self-consistency.

    FP: Monadic loop over ``scf_step_rhf``. The loop is written in Python
    rather than ``lax.while_loop`` so it can hold the mutable timer and
    the NumPy DIIS solve. Inside each step, every sub-computation is
    Applicative and fully jit-able.
    """
    if mol.spin != 0:
        raise ValueError(
            f"run_rhf requires a closed-shell molecule (spin=0); got spin={mol.spin}."
        )
    if mol.n_electrons % 2 != 0:
        raise ValueError(
            "run_rhf requires an even number of electrons; got "
            f"{mol.n_electrons}."
        )
    n_occ = mol.n_electrons // 2

    with optional_stage(timer, "scf", "monad"):
        with optional_stage(timer, "orthogonalize", "applicative"):
            X = symmetric_orthogonalization(ints.S)
        with optional_stage(timer, "initial_guess", "applicative"):
            D0 = core_guess(ints.H_core, X, n_occ)
            F0 = build_fock_rhf(ints.H_core, D0, ints.ERI)
            orb_e0, C0 = _diagonalize_fock(F0, X)
            E0 = float(electronic_energy_rhf(D0, ints.H_core, F0))
        state = SCFState(
            density=D0,
            fock=F0,
            coefficients=C0,
            orbital_energies=orb_e0,
            energy=E0,
            iteration=0,
        )
        diis_history = diis_history_init(ints.S.shape[0], config.diis_space)

        converged = False
        last_energy = state.energy
        for _ in range(config.max_scf_iter):
            state, diis_history, error = scf_step_rhf(
                state,
                ints.H_core,
                ints.ERI,
                ints.S,
                X,
                n_occ,
                diis_history,
                damping=config.damping,
                timer=timer,
            )
            dE = abs(state.energy - last_energy)
            err_max = float(jnp.max(jnp.abs(error)))
            last_energy = state.energy
            if dE < config.scf_conv and err_max < max(config.scf_conv, 1e-6):
                converged = True
                break

    total = state.energy + float(ints.E_nuc)
    return SCFResult(
        converged=converged,
        state=state,
        energy=total,
        E_elec=state.energy,
        E_nuc=float(ints.E_nuc),
        n_iterations=state.iteration,
        S=ints.S,
        H_core=ints.H_core,
        ERI=ints.ERI,
    )
