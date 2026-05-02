"""Parse BSE dictionary format into internal Shell/BasisSet types.

BSE returns basis sets as nested dicts with string-encoded exponents and
coefficients. This module converts them to JAX-compatible numeric arrays
and wraps them in Shell objects.

FP: Pure functions.
"""

from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from jax_qc.basis.normalize import normalize_shell
from jax_qc.core.types import Molecule, Shell


def _to_float_array(string_list) -> np.ndarray:
    """BSE stores numbers as strings; convert to float64 numpy array."""
    return np.array([float(s) for s in string_list], dtype=np.float64)


def parse_electron_shell(
    shell_dict: dict,
    center: np.ndarray,
    atom_index: int,
) -> List[Shell]:
    """Parse one BSE ``electron_shells`` entry into one or more Shell objects.

    A single BSE entry may carry multiple angular momenta (combined sp/spd
    shells) and multiple contraction coefficient vectors (general
    contraction). We always emit one Shell per (angular momentum,
    contraction vector) pair so downstream code can assume a single l and
    a single coefficient vector per Shell.

    Args:
        shell_dict: BSE electron_shell dict with keys
                    'angular_momentum', 'exponents', 'coefficients'.
        center:     (3,) atom coordinate in Bohr.
        atom_index: index of the atom in the parent Molecule.

    Returns:
        list of Shell objects.
    """
    angular_momenta = list(shell_dict["angular_momentum"])
    exponents = _to_float_array(shell_dict["exponents"])
    coeff_rows = shell_dict["coefficients"]

    # BSE pairs each contraction vector with one angular momentum. For
    # uncontract_spdf=True, len(angular_momenta) == 1, so every coeff row
    # shares that l. For general contractions (multiple rows, one l), we
    # emit one Shell per row.
    shells: List[Shell] = []
    if len(angular_momenta) == len(coeff_rows):
        pairs = list(zip(angular_momenta, coeff_rows))
    elif len(angular_momenta) == 1:
        l = angular_momenta[0]
        pairs = [(l, row) for row in coeff_rows]
    else:
        raise ValueError(
            "Unexpected BSE shell layout: "
            f"{len(angular_momenta)} angular momenta, {len(coeff_rows)} coefficient rows."
        )

    for l, coeff_strings in pairs:
        coeffs = _to_float_array(coeff_strings)
        # BSE sometimes zero-pads coefficients when different l's share an
        # exponent list in a general contraction. Drop primitives whose
        # coefficient is exactly zero for this l.
        mask = coeffs != 0.0
        if not mask.all():
            alphas = exponents[mask]
            coeffs = coeffs[mask]
        else:
            alphas = exponents
        normed = normalize_shell(alphas, coeffs, int(l))
        shells.append(
            Shell(
                angular_momentum=int(l),
                exponents=jnp.asarray(alphas, dtype=jnp.float64),
                coefficients=jnp.asarray(normed, dtype=jnp.float64),
                center=jnp.asarray(center, dtype=jnp.float64),
                atom_index=int(atom_index),
            )
        )
    return shells


def bse_dict_to_shells(bse_data: dict, mol: Molecule) -> List[Shell]:
    """Convert a BSE dict into a flat list of Shell objects for ``mol``.

    For each atom in ``mol`` (in order), look up its element in
    ``bse_data``, parse every electron shell, and attach the atom's
    coordinate. Within each atom we sort shells by ``(angular_momentum
    ascending, max_exponent descending)`` to match the convention used by
    PySCF / libcint (innermost-first within each l block); BSE itself
    leaves them in input-file order, which is usually the opposite.
    """
    shells: List[Shell] = []
    elements_data = bse_data.get("elements", {})
    atomic_numbers = np.asarray(mol.atomic_numbers).tolist()
    coords = np.asarray(mol.coords, dtype=np.float64)

    for atom_idx, z in enumerate(atomic_numbers):
        key = str(int(z))
        if key not in elements_data:
            raise KeyError(
                f"Basis set does not define element Z={z} "
                f"(atom index {atom_idx})."
            )
        atom_center = coords[atom_idx]
        atom_shells: List[Shell] = []
        for shell_dict in elements_data[key].get("electron_shells", []):
            atom_shells.extend(
                parse_electron_shell(shell_dict, atom_center, atom_idx)
            )
        atom_shells.sort(
            key=lambda sh: (
                int(sh.angular_momentum),
                -float(np.max(np.asarray(sh.exponents))),
            )
        )
        shells.extend(atom_shells)
    return shells


def build_shell_indices(
    shells: List[Shell], spherical: bool = True
) -> Tuple[Tuple[Tuple[int, ...], ...], np.ndarray, int]:
    """Build ``(shell_to_basis, basis_to_atom, n_basis)`` index tables.

    For spherical harmonics, a shell with angular momentum l contributes
    2 l + 1 basis functions. For Cartesian, it contributes (l+1)(l+2)/2.
    """
    shell_to_basis: List[Tuple[int, ...]] = []
    basis_to_atom: List[int] = []
    offset = 0
    for sh in shells:
        l = int(sh.angular_momentum)
        n = (2 * l + 1) if spherical else ((l + 1) * (l + 2) // 2)
        indices = tuple(range(offset, offset + n))
        shell_to_basis.append(indices)
        basis_to_atom.extend([int(sh.atom_index)] * n)
        offset += n
    return (
        tuple(shell_to_basis),
        np.asarray(basis_to_atom, dtype=np.int32),
        offset,
    )
