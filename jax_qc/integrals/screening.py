"""Schwarz screening for two-electron integrals.

By the Cauchy-Schwarz inequality

    | (mu nu | lambda sigma) |
        <= sqrt( (mu nu | mu nu) ) * sqrt( (lambda sigma | lambda sigma) ).

We compute the diagonal "shell-pair magnitudes" Q_{ij} = sqrt((ij|ij))
once, then a quartet of shells (i, j, k, l) can be skipped whenever
Q_{ij} * Q_{kl} < threshold.

This module exposes:

* ``compute_shell_pair_bounds(basis)`` — returns the (n_shell, n_shell)
  matrix Q.
* ``estimate_eri_upper_bound(basis, Q, i, j, k, l)`` — Schwarz upper
  bound for a shell quartet.

FP: Applicative — Q_{ij} are independent.
"""

from __future__ import annotations

import math

import numpy as np

from jax_qc.core.types import BasisSet
from jax_qc.integrals.obara_saika import contracted_eri_block


def compute_shell_pair_bounds(basis: BasisSet) -> np.ndarray:
    """Return Q[i, j] = sqrt(max |(ij|ij)|) for every shell pair.

    The "max" is taken over all Cartesian components of the (ij|ij) block.
    This is conservative (independent of the spherical projection) but
    sufficient for screening since spherical transforms are unitary in
    norm.
    """
    n_shells = len(basis.shells)
    Q = np.zeros((n_shells, n_shells), dtype=np.float64)
    for i, sa in enumerate(basis.shells):
        for j in range(i, n_shells):
            sb = basis.shells[j]
            block = contracted_eri_block(sa, sb, sa, sb)
            # The diagonal of the unfolded (ij|ij) tensor maps (mu, nu)
            # to (mu, nu, mu, nu) in the 4-index block.
            nc_a = block.shape[0]
            nc_b = block.shape[1]
            max_val = 0.0
            for a in range(nc_a):
                for b in range(nc_b):
                    val = block[a, b, a, b]
                    if val > max_val:
                        max_val = val
            Q[i, j] = math.sqrt(max(max_val, 0.0))
            Q[j, i] = Q[i, j]
    return Q


def estimate_eri_upper_bound(
    Q: np.ndarray, i: int, j: int, k: int, l: int
) -> float:
    """Schwarz upper bound on |(ij|kl)|."""
    return float(Q[i, j] * Q[k, l])
