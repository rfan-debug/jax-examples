"""Molecular integrals (Applicative).

Step 2 implements s-type (angular momentum l=0) analytical integrals:
overlap, kinetic, nuclear attraction, and two-electron repulsion.
Higher angular momentum arrives in Step 4 via Obara-Saika recurrence.
"""

from jax_qc.integrals.boys import boys_f0, boys_fn
from jax_qc.integrals.gaussian_product import (
    distance_squared,
    gaussian_product_center,
    gaussian_product_exponent,
)
from jax_qc.integrals.overlap import compute_overlap_matrix, overlap_primitive_ss
from jax_qc.integrals.kinetic import compute_kinetic_matrix, kinetic_primitive_ss
from jax_qc.integrals.nuclear import (
    compute_nuclear_matrix,
    nuclear_primitive_ss,
    nuclear_repulsion_energy,
)
from jax_qc.integrals.eri import compute_eri_tensor, eri_primitive_ssss
from jax_qc.integrals.interface import compute_integrals

__all__ = [
    "boys_f0",
    "boys_fn",
    "distance_squared",
    "gaussian_product_center",
    "gaussian_product_exponent",
    "compute_overlap_matrix",
    "overlap_primitive_ss",
    "compute_kinetic_matrix",
    "kinetic_primitive_ss",
    "compute_nuclear_matrix",
    "nuclear_primitive_ss",
    "nuclear_repulsion_energy",
    "compute_eri_tensor",
    "eri_primitive_ssss",
    "compute_integrals",
]
