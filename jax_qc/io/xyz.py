"""Read and write XYZ coordinate files.

Standard XYZ format:

    <n_atoms>
    <comment line>
    <symbol> <x> <y> <z>
    ...

Coordinates in XYZ files are conventionally in Angstrom. We convert to
Bohr on read and convert back on write. Override with ``unit='bohr'``
to skip conversion.

FP: File IO is a side effect, isolated here. The returned ``Molecule``
object is pure data.
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Union

import jax.numpy as jnp
import numpy as np

from jax_qc.core.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, symbol_to_z
from jax_qc.core.types import Molecule, make_molecule

PathLike = Union[str, pathlib.Path]


def parse_xyz_string(
    text: str,
    unit: str = "angstrom",
    charge: int = 0,
    spin: int = 0,
) -> Molecule:
    """Parse the contents of an XYZ file into a ``Molecule``.

    Args:
        text:   entire file contents as a string.
        unit:   'angstrom' (default) or 'bohr'. Controls whether coordinates
                are converted to Bohr for the internal representation.
        charge: total molecular charge.
        spin:   2S (0=singlet, 1=doublet, 2=triplet, ...).
    """
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ input must contain at least a count line and a comment line.")
    try:
        n_atoms = int(lines[0].strip().split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid atom count on first line: {lines[0]!r}") from exc
    body = [ln for ln in lines[2:] if ln.strip()]
    if len(body) < n_atoms:
        raise ValueError(
            f"XYZ declares {n_atoms} atoms but only {len(body)} coordinate lines were found."
        )
    elements: list[str] = []
    coords: list[list[float]] = []
    for i in range(n_atoms):
        parts = body[i].split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ coordinate line: {body[i]!r}")
        sym, x, y, z = parts[0], parts[1], parts[2], parts[3]
        elements.append(sym.strip().title())
        coords.append([float(x), float(y), float(z)])
    coord_arr = np.array(coords, dtype=np.float64)
    if unit.lower() == "angstrom":
        coord_arr = coord_arr * ANGSTROM_TO_BOHR
    elif unit.lower() == "bohr":
        pass
    else:
        raise ValueError(f"Unknown unit {unit!r}; expected 'angstrom' or 'bohr'.")
    atomic_numbers = np.array([symbol_to_z(sym) for sym in elements], dtype=np.int32)
    return make_molecule(
        elements=tuple(elements),
        coords=jnp.asarray(coord_arr),
        atomic_numbers=jnp.asarray(atomic_numbers),
        charge=charge,
        spin=spin,
    )


def read_xyz(
    path: PathLike,
    unit: str = "angstrom",
    charge: int = 0,
    spin: int = 0,
) -> Molecule:
    """Read an XYZ file and return a ``Molecule`` with coordinates in Bohr."""
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as f:
        return parse_xyz_string(f.read(), unit=unit, charge=charge, spin=spin)


def write_xyz(
    path: PathLike,
    molecule: Molecule,
    unit: str = "angstrom",
    comment: str = "",
) -> None:
    """Write a Molecule to an XYZ file.

    Internal coordinates are Bohr; the default output unit is Angstrom to
    match the XYZ convention.
    """
    coords = np.asarray(molecule.coords, dtype=np.float64)
    if unit.lower() == "angstrom":
        coords = coords * BOHR_TO_ANGSTROM
    elif unit.lower() == "bohr":
        pass
    else:
        raise ValueError(f"Unknown unit {unit!r}; expected 'angstrom' or 'bohr'.")
    elements: Iterable[str] = molecule.elements
    lines = [f"{len(molecule.elements)}", comment]
    for sym, (x, y, z) in zip(elements, coords):
        lines.append(f"{sym:<3s} {x: .10f} {y: .10f} {z: .10f}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
