"""XYZ input/output: parse, inspect, write back.

Reads a small water XYZ string, prints the derived Molecule, writes it
to disk in Bohr units, reads it back, and confirms the round-trip.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.io.xyz import parse_xyz_string
from examples._colors import banner, header, label, ok, value


WATER_XYZ = """\
3
water at equilibrium, angstrom
O   0.000000   0.000000   0.117176
H   0.000000   0.757087  -0.468706
H   0.000000  -0.757087  -0.468706
"""


def main() -> None:
    print(banner("XYZ Round-Trip (Angstrom <-> Bohr)"))

    mol = parse_xyz_string(WATER_XYZ)  # default unit: Angstrom on disk
    print(header("Parsed water:"))
    print(f"  {label('elements')}      = {mol.elements}")
    print(f"  {label('n_electrons')}   = {value(str(mol.n_electrons))}")
    print(f"  {label('coords (Bohr)')} =\n{np.asarray(mol.coords)}")

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "water.xyz"
        # Write in Bohr so we can verify the internal-coord round-trip.
        jax_qc.write_xyz(path, mol, unit="bohr", comment="water (bohr)")
        print()
        print(header(f"Wrote {path}:", color="bright_green"))
        print(path.read_text())

        mol2 = jax_qc.read_xyz(path, unit="bohr")
        delta = float(np.max(np.abs(np.asarray(mol.coords) - np.asarray(mol2.coords))))
        tag = ok("PASS") if delta < 1e-9 else ok("OK") if delta < 1e-6 else ""
        print(
            f"{label('Max coordinate difference after round-trip')}: "
            f"{value(f'{delta:.2e} Bohr')} [{tag}]"
        )


if __name__ == "__main__":
    main()
