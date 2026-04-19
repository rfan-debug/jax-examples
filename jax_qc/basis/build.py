"""Top-level basis set construction.

Entry point: ``build_basis_set(molecule, basis_name) -> BasisSet``.

Uses the disk cache to avoid re-fetching across calls.

FP: Pure apart from the disk cache side effect, which is confined to
``jax_qc.basis.cache``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import jax.numpy as jnp
import numpy as np

from jax_qc.basis.bse_fetch import fetch_basis
from jax_qc.basis.cache import get_cached, put_cache
from jax_qc.basis.parse import bse_dict_to_shells, build_shell_indices
from jax_qc.core.types import BasisSet, Molecule


def build_basis_set(
    mol: Molecule,
    basis_name: str,
    use_cache: bool = True,
    spherical: bool = True,
) -> BasisSet:
    """Build a ``BasisSet`` for ``mol`` using the named basis.

    Args:
        mol:         parsed Molecule.
        basis_name:  any name recognized by Basis Set Exchange, e.g. 'sto-3g',
                     '6-31G*', 'cc-pVDZ'.
        use_cache:   whether to read/write the on-disk BSE cache.
        spherical:   True (default) uses 2 l + 1 spherical-harmonic basis
                     functions per shell. False uses Cartesian.

    Returns:
        BasisSet with normalized contracted shells.
    """
    elements = _unique_elements(mol.atomic_numbers)
    bse_data: Optional[dict] = None
    if use_cache:
        bse_data = get_cached(basis_name, elements)
    if bse_data is None:
        bse_data = fetch_basis(basis_name, elements)
        if use_cache:
            put_cache(basis_name, elements, bse_data)
    shells = bse_dict_to_shells(bse_data, mol)
    shell_to_basis, basis_to_atom, n_basis = build_shell_indices(
        shells, spherical=spherical
    )
    return BasisSet(
        shells=tuple(shells),
        n_basis=n_basis,
        shell_to_basis=shell_to_basis,
        basis_to_atom=jnp.asarray(basis_to_atom, dtype=jnp.int32),
        name=basis_name,
        spherical=spherical,
    )


def _unique_elements(atomic_numbers) -> Iterable[int]:
    arr = np.asarray(atomic_numbers).tolist()
    return sorted({int(z) for z in arr})
