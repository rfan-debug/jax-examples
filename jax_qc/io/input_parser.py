"""High-level convenience API for jax_qc.

Provides user-friendly entry points that accept dictionaries, keyword
arguments, or XYZ files and wire together the low-level modules
(molecule construction, basis set, integrals, SCF).

Styles supported:

    # Style 1 — dict / YAML
    result = jax_qc.run({
        'molecule': {'atoms': ['O','H','H'], 'coords': [...], 'unit': 'angstrom'},
        'method': 'rhf',
        'basis': '6-31g*',
    })

    # Style 2 — keyword API
    result = jax_qc.energy(mol, method='rhf', basis='cc-pVDZ')

    # Style 3 — XYZ file
    result = jax_qc.run_xyz('water.xyz', method='rhf', basis='sto-3g')

FP: These are the effectful edge of the framework (they trigger IO and
side effects). Internally they compose pure functions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax.numpy as jnp
import numpy as np

from jax_qc.basis.build import build_basis_set
from jax_qc.core.constants import ANGSTROM_TO_BOHR, symbol_to_z
from jax_qc.core.types import (
    BasisSet,
    CalcConfig,
    IntegralSet,
    Molecule,
    SCFResult,
    make_molecule,
)
from jax_qc.integrals.interface import compute_integrals
from jax_qc.io.xyz import read_xyz
from jax_qc.profiling.timer import StageTimer
from jax_qc.scf.interface import run_scf


def build_molecule(
    atoms: list[str],
    coords: list[list[float]],
    unit: str = "angstrom",
    charge: int = 0,
    spin: int = 0,
) -> Molecule:
    """Build a Molecule from element symbols and coordinates.

    FP: Pure (aside from the list -> array conversion).

    Args:
        atoms:  list of element symbols, e.g. ['O', 'H', 'H'].
        coords: (n_atoms, 3) list of coordinates.
        unit:   'angstrom' (default) or 'bohr'.
        charge: total molecular charge.
        spin:   2S (0=singlet, 1=doublet, ...).

    Returns:
        A Molecule with coordinates in Bohr.
    """
    coord_arr = np.array(coords, dtype=np.float64)
    if unit.lower() == "angstrom":
        coord_arr = coord_arr * ANGSTROM_TO_BOHR
    elif unit.lower() != "bohr":
        raise ValueError(f"Unknown unit {unit!r}; expected 'angstrom' or 'bohr'.")

    elements = [a.strip().title() for a in atoms]
    atomic_numbers = np.array([symbol_to_z(a) for a in elements], dtype=np.int32)
    return make_molecule(
        elements=tuple(elements),
        coords=jnp.asarray(coord_arr),
        atomic_numbers=jnp.asarray(atomic_numbers),
        charge=charge,
        spin=spin,
    )


def _parse_molecule_dict(mol_dict: Dict[str, Any]) -> Molecule:
    """Parse a molecule specification from a dict."""
    atoms = mol_dict["atoms"]
    coords = mol_dict["coords"]
    unit = mol_dict.get("unit", "angstrom")
    charge = mol_dict.get("charge", 0)
    spin = mol_dict.get("spin", 0)
    return build_molecule(atoms, coords, unit=unit, charge=charge, spin=spin)


def energy(
    mol: Molecule,
    method: str = "rhf",
    basis: str = "sto-3g",
    **kwargs: Any,
) -> SCFResult:
    """Compute the SCF energy for a Molecule.

    This is the main convenience function. It builds the basis set,
    computes integrals, and runs the SCF in one call.

    Args:
        mol:    a Molecule object.
        method: 'rhf' or 'uhf'.
        basis:  any basis set name supported by BSE.
        **kwargs: additional CalcConfig overrides (e.g. max_scf_iter,
                  scf_conv, profile).

    Returns:
        An SCFResult with the converged energy and wavefunction.
    """
    config = CalcConfig(method=method, basis=basis, **kwargs)
    timer = StageTimer(sync_device=False) if config.profile else None
    bs = build_basis_set(mol, basis)
    ints = compute_integrals(mol, bs, timer=timer)
    return run_scf(mol, ints, config, timer=timer)


def run(input_dict: Dict[str, Any]) -> SCFResult:
    """Run a calculation from a dictionary specification.

    Expected keys:
        'molecule': dict with 'atoms', 'coords', and optional 'unit',
                    'charge', 'spin'.
        'method':   'rhf' or 'uhf' (default: 'rhf').
        'basis':    basis set name (default: 'sto-3g').
        'task':     'energy' (default; only option for now).
        'profile':  bool (default: False).
        ... plus any CalcConfig fields.

    Returns:
        An SCFResult.
    """
    mol = _parse_molecule_dict(input_dict["molecule"])
    method = input_dict.get("method", "rhf")
    basis = input_dict.get("basis", "sto-3g")
    config_keys = {
        "method",
        "basis",
        "task",
        "max_scf_iter",
        "scf_conv",
        "diis_space",
        "damping",
        "guess",
        "verbose",
        "profile",
    }
    kwargs = {k: v for k, v in input_dict.items() if k in config_keys}
    kwargs.setdefault("method", method)
    kwargs.setdefault("basis", basis)
    return energy(mol, **kwargs)


def run_xyz(
    path: str,
    method: str = "rhf",
    basis: str = "sto-3g",
    unit: str = "angstrom",
    charge: int = 0,
    spin: int = 0,
    **kwargs: Any,
) -> SCFResult:
    """Run a calculation from an XYZ file.

    Args:
        path:   path to the XYZ file.
        method: 'rhf' or 'uhf'.
        basis:  basis set name.
        unit:   coordinate unit in the XYZ file ('angstrom' or 'bohr').
        charge: molecular charge.
        spin:   2S.
        **kwargs: additional CalcConfig overrides.

    Returns:
        An SCFResult.
    """
    mol = read_xyz(path, unit=unit, charge=charge, spin=spin)
    return energy(mol, method=method, basis=basis, **kwargs)
