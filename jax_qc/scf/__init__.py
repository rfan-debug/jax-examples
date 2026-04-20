"""Self-consistent field (SCF) machinery.

Step 3 delivers RHF (restricted Hartree-Fock) on top of the Step 2
s-type integral layer. The module is organized so that every SCF
sub-step is a pure function (Applicative internals) and only the outer
iteration loop is Monadic.
"""

from jax_qc.scf.orthogonalize import (
    canonical_orthogonalization,
    symmetric_orthogonalization,
)
from jax_qc.scf.density import density_rhf
from jax_qc.scf.fock import build_fock_rhf
from jax_qc.scf.energy import electronic_energy_rhf
from jax_qc.scf.guess import core_guess
from jax_qc.scf.damping import damp
from jax_qc.scf.diis import DIISHistory, diis_extrapolate
from jax_qc.scf.rhf import run_rhf, scf_step_rhf
from jax_qc.scf.interface import run_scf

__all__ = [
    "canonical_orthogonalization",
    "symmetric_orthogonalization",
    "density_rhf",
    "build_fock_rhf",
    "electronic_energy_rhf",
    "core_guess",
    "damp",
    "DIISHistory",
    "diis_extrapolate",
    "run_rhf",
    "scf_step_rhf",
    "run_scf",
]
