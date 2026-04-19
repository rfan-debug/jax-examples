"""Top-level entry point for the Applicative integral layer.

``compute_integrals(molecule, basis, timer=None)`` builds every matrix
required by an SCF: S, T, V, H_core = T + V, ERI, and E_nuc. Each stage
is wrapped in an optional StageTimer context so the caller can profile
the integral build without cluttering the call site.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from jax_qc.core.types import BasisSet, IntegralSet, Molecule
from jax_qc.integrals.eri import compute_eri_tensor
from jax_qc.integrals.kinetic import compute_kinetic_matrix
from jax_qc.integrals.nuclear import (
    compute_nuclear_matrix,
    nuclear_repulsion_energy,
)
from jax_qc.integrals.overlap import compute_overlap_matrix
from jax_qc.profiling.timer import StageTimer, optional_stage


def compute_integrals(
    mol: Molecule,
    basis: BasisSet,
    timer: Optional[StageTimer] = None,
) -> IntegralSet:
    """Build a complete IntegralSet for the given molecule and basis.

    FP: The entire layer is Applicative — S, T, V, ERI, and E_nuc are
    independent and could run in parallel. We evaluate them sequentially
    here to keep the profiling readable; XLA will still fuse within each
    matrix.

    Args:
        mol:    parsed Molecule (coordinates in Bohr).
        basis:  BasisSet produced by ``jax_qc.build_basis_set``.
        timer:  optional StageTimer; if provided, each sub-stage is
                recorded under the 'integrals' parent.

    Returns:
        IntegralSet with S, T, V, H_core, ERI, and E_nuc.
    """
    with optional_stage(timer, "integrals", "applicative"):
        with optional_stage(timer, "overlap", "applicative"):
            S = compute_overlap_matrix(basis)
        with optional_stage(timer, "kinetic", "applicative"):
            T = compute_kinetic_matrix(basis)
        with optional_stage(timer, "nuclear", "applicative"):
            V = compute_nuclear_matrix(basis, mol)
        with optional_stage(timer, "h_core", "applicative"):
            H_core = T + V
        with optional_stage(timer, "eri", "applicative"):
            ERI = compute_eri_tensor(basis)
        with optional_stage(timer, "nuclear_repulsion", "foldable"):
            E_nuc = nuclear_repulsion_energy(mol)
    return IntegralSet(
        S=jnp.asarray(S),
        T=jnp.asarray(T),
        V=jnp.asarray(V),
        H_core=jnp.asarray(H_core),
        ERI=jnp.asarray(ERI),
        E_nuc=float(E_nuc),
    )
