"""Tests for jax_qc.core types and constants."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax_qc.core import constants
from jax_qc.core.types import (
    BasisSet,
    CalcConfig,
    Molecule,
    Primitive,
    Shell,
    make_molecule,
)


def test_element_lookup_roundtrip():
    for z in (1, 6, 8, 16, 26):
        sym = constants.z_to_symbol(z)
        assert constants.symbol_to_z(sym) == z


def test_symbol_lookup_case_insensitive():
    assert constants.symbol_to_z("h") == 1
    assert constants.symbol_to_z("H") == 1
    assert constants.symbol_to_z(" He ") == 2


def test_symbol_lookup_rejects_unknown():
    with pytest.raises(ValueError):
        constants.symbol_to_z("Xx")


def test_unit_conversion_roundtrip():
    assert constants.ANGSTROM_TO_BOHR * constants.BOHR_TO_ANGSTROM == pytest.approx(1.0)
    assert constants.HARTREE_TO_EV * constants.EV_TO_HARTREE == pytest.approx(1.0)


def test_primitive_is_pytree():
    p = Primitive(exponent=1.0, coeff=0.5, center=jnp.zeros(3))
    leaves, _ = jax.tree_util.tree_flatten(p)
    # exponent, coeff, center -> 3 leaves.
    assert len(leaves) == 3


def test_molecule_constructor_derives_electron_counts():
    mol = make_molecule(
        elements=("O", "H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([8, 1, 1], dtype=jnp.int32),
    )
    assert mol.n_electrons == 10
    assert mol.n_alpha == 5
    assert mol.n_beta == 5
    assert mol.charge == 0
    assert mol.spin == 0


def test_molecule_constructor_handles_charge_and_spin():
    mol = make_molecule(
        elements=("O", "O"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.282]]),
        atomic_numbers=jnp.array([8, 8], dtype=jnp.int32),
        charge=0,
        spin=2,  # triplet O2
    )
    assert mol.n_electrons == 16
    assert mol.n_alpha == 9
    assert mol.n_beta == 7
    assert mol.n_alpha - mol.n_beta == mol.spin


def test_molecule_rejects_inconsistent_spin():
    with pytest.raises(ValueError, match="Inconsistent spin"):
        make_molecule(
            elements=("H", "H"),
            coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
            atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
            spin=1,  # 2 electrons with doublet is impossible
        )


def test_molecule_rejects_overionization():
    with pytest.raises(ValueError, match="Negative electron count"):
        make_molecule(
            elements=("H",),
            coords=jnp.array([[0.0, 0.0, 0.0]]),
            atomic_numbers=jnp.array([1], dtype=jnp.int32),
            charge=2,
        )


def test_molecule_is_frozen():
    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    with pytest.raises((AttributeError, Exception)):
        mol.charge = 1  # type: ignore[misc]


def test_molecule_is_pytree():
    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    leaves, treedef = jax.tree_util.tree_flatten(mol)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.elements == mol.elements
    assert int(rebuilt.n_electrons) == mol.n_electrons


def test_shell_and_basis_set_pytree():
    shell = Shell(
        angular_momentum=0,
        exponents=jnp.array([3.42, 0.62, 0.17]),
        coefficients=jnp.array([0.15, 0.54, 0.44]),
        center=jnp.zeros(3),
        atom_index=0,
    )
    basis = BasisSet(
        shells=(shell,),
        n_basis=1,
        shell_to_basis=((0,),),
        basis_to_atom=jnp.array([0], dtype=jnp.int32),
        name="test",
        spherical=True,
    )
    leaves, treedef = jax.tree_util.tree_flatten(basis)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.n_basis == 1
    assert rebuilt.name == "test"


def test_calc_config_defaults():
    cfg = CalcConfig()
    assert cfg.method == "rhf"
    assert cfg.basis == "sto-3g"
    assert cfg.profile is False
    assert cfg.max_scf_iter == 128
