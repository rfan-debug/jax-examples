"""Orbital energy analysis.

Extracts physically meaningful quantities from the converged MO
energies: HOMO/LUMO identification, gap, and occupation summary.

FP: Foldable — pure functions over the orbital_energies array.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp

from jax_qc.core.types import Molecule, SCFResult


class OrbitalInfo(NamedTuple):
    """Summary of orbital energies and occupations.

    Fields:
        orbital_energies: (n_basis,) sorted eigenvalues in Hartree.
        occupations:      (n_basis,) occupation numbers (2.0 or 0.0 for RHF;
                          alpha count for UHF stored in state).
        homo_energy:      energy of the highest occupied MO.
        lumo_energy:      energy of the lowest unoccupied MO (None if all occupied).
        homo_lumo_gap:    LUMO - HOMO in Hartree (None if all occupied).
        homo_index:       index of the HOMO in the orbital array.
        lumo_index:       index of the LUMO (None if all occupied).
    """

    orbital_energies: jnp.ndarray
    occupations: jnp.ndarray
    homo_energy: float
    lumo_energy: Optional[float]
    homo_lumo_gap: Optional[float]
    homo_index: int
    lumo_index: Optional[int]


def orbital_analysis(result: SCFResult, mol: Molecule) -> OrbitalInfo:
    """Analyze orbital energies from a converged SCF result.

    FP: Foldable — pure function of the SCFResult and Molecule.

    For RHF (spin=0), HOMO is the highest doubly-occupied orbital and
    LUMO is the first virtual. For UHF, the analysis uses the alpha
    orbital energies stored in ``result.state``.

    Args:
        result: converged SCF result.
        mol:    the Molecule (provides electron count).

    Returns:
        OrbitalInfo named tuple with HOMO/LUMO energies and gap.
    """
    orb_e = result.state.orbital_energies
    n_basis = len(orb_e)

    if mol.spin == 0:
        n_occ = mol.n_electrons // 2
    else:
        # UHF: alpha orbital energies, alpha occupation
        n_occ = mol.n_alpha

    # Build occupation array
    occ = jnp.zeros(n_basis)
    if mol.spin == 0:
        occ = occ.at[:n_occ].set(2.0)  # RHF: doubly occupied
    else:
        occ = occ.at[:n_occ].set(1.0)  # UHF alpha channel

    homo_idx = n_occ - 1
    homo_e = float(orb_e[homo_idx])

    if n_occ < n_basis:
        lumo_idx = n_occ
        lumo_e = float(orb_e[lumo_idx])
        gap = lumo_e - homo_e
    else:
        lumo_idx = None
        lumo_e = None
        gap = None

    return OrbitalInfo(
        orbital_energies=orb_e,
        occupations=occ,
        homo_energy=homo_e,
        lumo_energy=lumo_e,
        homo_lumo_gap=gap,
        homo_index=homo_idx,
        lumo_index=lumo_idx,
    )
