"""Core data types. All immutable, all pytree-compatible.

Rule: chex.dataclass(frozen=True) for complex types (>3 fields, optional, nested).
      NamedTuple for simple leaf types (2-3 fields, no optionals).
      Computation lives in the module that owns the relevant FP abstraction,
      never as methods on data types.

FP: These types are the "objects" of the category. Everything else is a
    morphism between them.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import chex
import jax.numpy as jnp


# --- Simple leaf types (NamedTuple: lightweight, auto-pytree) ---


class Primitive(NamedTuple):
    """Single primitive Gaussian: N * exp(-alpha * |r - center|^2).

    Fields:
        exponent: orbital exponent alpha (scalar).
        coeff:    contraction coefficient * normalization factor (scalar).
        center:   (3,) float array for the Gaussian center, in Bohr.
    """

    exponent: float
    coeff: float
    center: jnp.ndarray


# --- Complex types (chex.dataclass: rich, frozen, pytree-registered) ---


@chex.dataclass(frozen=True)
class Shell:
    """A contracted shell (e.g., one s-shell or one p-shell on an atom).

    angular_momentum: 0=s, 1=p, 2=d, 3=f, ...
    exponents:        (n_primitives,) Gaussian exponents alpha_i.
    coefficients:     (n_primitives,) contraction coefficients d_i *
                      normalization N_i. Pre-multiplied normalization means
                      the contracted shell evaluates to
                      sum_i d_i * N_i * exp(-alpha_i r^2).
    center:           (3,) array, atom position in Bohr.
    atom_index:       index into Molecule.coords / .atomic_numbers.
    """

    angular_momentum: int
    exponents: jnp.ndarray
    coefficients: jnp.ndarray
    center: jnp.ndarray
    atom_index: int


@chex.dataclass(frozen=True)
class BasisSet:
    """Complete basis set for a molecule.

    shells:          tuple of Shell objects (ordered by atom, then by shell).
    n_basis:         total number of contracted basis functions (counts
                     spherical-harmonic components: 2*l + 1 per shell).
    shell_to_basis:  tuple of tuples; shell_to_basis[i] lists the contiguous
                     basis-function indices owned by shells[i].
    basis_to_atom:   (n_basis,) int array; basis_to_atom[mu] is the atom
                     index that basis function mu is centered on.
    name:            basis set name (e.g., 'sto-3g').
    spherical:       True for spherical harmonics (default), False for Cartesian.
    """

    shells: tuple
    n_basis: int
    shell_to_basis: tuple
    basis_to_atom: jnp.ndarray
    name: str = ""
    spherical: bool = True


@chex.dataclass(frozen=True)
class Molecule:
    """Molecular system specification (coordinates in Bohr).

    coords:         (n_atoms, 3) float array.
    atomic_numbers: (n_atoms,) int array.
    elements:       tuple of element symbols, e.g. ('O', 'H', 'H').
    charge:         total molecular charge.
    spin:           2S (0=singlet, 1=doublet, 2=triplet).
    n_electrons:    total electron count (= sum(Z) - charge).
    n_alpha:        number of alpha electrons.
    n_beta:         number of beta electrons.
    """

    coords: jnp.ndarray
    atomic_numbers: jnp.ndarray
    elements: Tuple[str, ...]
    charge: int = 0
    spin: int = 0
    n_electrons: int = 0
    n_alpha: int = 0
    n_beta: int = 0


@chex.dataclass(frozen=True)
class IntegralSet:
    """All molecular integrals. Output of the Applicative integral layer.

    All fields are independent — computed in parallel, consumed by SCF.
    """

    S: jnp.ndarray
    T: jnp.ndarray
    V: jnp.ndarray
    H_core: jnp.ndarray
    ERI: jnp.ndarray
    E_nuc: float


@chex.dataclass(frozen=True)
class SCFState:
    """Immutable SCF iteration state.

    This is the 's' in State Monad: step :: s -> (s, a).
    """

    density: jnp.ndarray
    fock: jnp.ndarray
    coefficients: jnp.ndarray
    orbital_energies: jnp.ndarray
    energy: float
    iteration: int


@chex.dataclass(frozen=True)
class SCFResult:
    """Final result after SCF convergence."""

    converged: bool
    state: SCFState
    energy: float
    E_elec: float
    E_nuc: float
    n_iterations: int
    S: jnp.ndarray
    H_core: jnp.ndarray
    ERI: Optional[jnp.ndarray] = None


@chex.dataclass(frozen=True)
class CalcConfig:
    """Calculation configuration — user-facing input."""

    method: str = "rhf"
    basis: str = "sto-3g"
    task: str = "energy"
    max_scf_iter: int = 128
    scf_conv: float = 1e-10
    diis_space: int = 8
    damping: float = 0.0
    guess: str = "core"
    verbose: int = 1
    profile: bool = False


def make_molecule(
    elements: Tuple[str, ...],
    coords: jnp.ndarray,
    atomic_numbers: jnp.ndarray,
    charge: int = 0,
    spin: int = 0,
) -> Molecule:
    """Construct a Molecule and derive electron counts.

    FP: Pure function. Validates spin multiplicity vs electron parity.
    """
    n_electrons = int(jnp.sum(atomic_numbers)) - int(charge)
    if n_electrons < 0:
        raise ValueError(
            f"Negative electron count: charge {charge} exceeds nuclear charge."
        )
    if (n_electrons - int(spin)) % 2 != 0:
        raise ValueError(
            f"Inconsistent spin: n_electrons={n_electrons}, 2S={spin} "
            "have different parity."
        )
    n_beta = (n_electrons - int(spin)) // 2
    n_alpha = n_electrons - n_beta
    return Molecule(
        coords=coords,
        atomic_numbers=atomic_numbers,
        elements=tuple(elements),
        charge=int(charge),
        spin=int(spin),
        n_electrons=n_electrons,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )
