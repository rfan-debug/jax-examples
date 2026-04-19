"""Tests for XYZ I/O."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_qc.core.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from jax_qc.core.types import make_molecule
from jax_qc.io.xyz import parse_xyz_string, read_xyz, write_xyz


WATER_XYZ = """3
water, Angstrom
O   0.000000   0.000000   0.117176
H   0.000000   0.757087  -0.468706
H   0.000000  -0.757087  -0.468706
"""


def test_parse_xyz_basic():
    mol = parse_xyz_string(WATER_XYZ)
    assert mol.elements == ("O", "H", "H")
    assert np.asarray(mol.atomic_numbers).tolist() == [8, 1, 1]
    assert mol.n_electrons == 10
    # Coordinates internally are Bohr; the O is near the origin.
    assert float(mol.coords[0, 2]) == pytest.approx(0.117176 * ANGSTROM_TO_BOHR, abs=1e-9)


def test_parse_xyz_bohr_unit():
    # Two hydrogens so the default singlet spin is consistent with electron parity.
    text = "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0\n"
    mol = parse_xyz_string(text, unit="bohr")
    assert float(mol.coords[1, 2]) == pytest.approx(1.0)


def test_parse_xyz_with_charge_and_spin():
    text = "2\n\nO 0 0 0\nO 0 0 2.282\n"
    mol = parse_xyz_string(text, unit="bohr", spin=2)
    assert mol.spin == 2
    assert mol.n_alpha == 9
    assert mol.n_beta == 7


def test_parse_xyz_rejects_short_input():
    with pytest.raises(ValueError):
        parse_xyz_string("")


def test_parse_xyz_rejects_truncated_body():
    text = "3\n\nH 0 0 0\nH 0 0 1\n"  # only 2 atoms
    with pytest.raises(ValueError):
        parse_xyz_string(text, unit="bohr")


def test_parse_xyz_rejects_malformed_line():
    text = "1\n\nH 0 0\n"  # missing z coordinate
    with pytest.raises(ValueError):
        parse_xyz_string(text)


def test_read_write_xyz_roundtrip(tmp_path):
    path = tmp_path / "water.xyz"
    path.write_text(WATER_XYZ, encoding="utf-8")
    mol = read_xyz(path)

    out = tmp_path / "roundtrip.xyz"
    write_xyz(out, mol, unit="angstrom", comment="roundtrip")
    mol2 = read_xyz(out)
    assert mol.elements == mol2.elements
    np.testing.assert_allclose(np.asarray(mol.coords), np.asarray(mol2.coords), atol=1e-9)


def test_write_xyz_bohr_matches_input():
    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )

    import io
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "h2.xyz"
        write_xyz(p, mol, unit="bohr", comment="h2 in bohr")
        text = p.read_text()
    lines = text.splitlines()
    assert lines[0].strip() == "2"
    # Second H should be at z = 1.4 Bohr in the output.
    z = float(lines[3].split()[-1])
    assert z == pytest.approx(1.4)


def test_write_xyz_default_unit_converts_to_angstrom():
    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "h2.xyz"
        write_xyz(p, mol)
        lines = p.read_text().splitlines()
    z = float(lines[3].split()[-1])
    assert z == pytest.approx(1.4 * BOHR_TO_ANGSTROM, abs=1e-9)
