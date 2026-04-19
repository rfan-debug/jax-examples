"""Gaussian normalization utilities.

For a Cartesian Gaussian primitive
    g(r) = x^lx y^ly z^lz exp(-alpha r^2),
the integral of g^2 over R^3 is known analytically. We pre-compute
normalization factors so that each primitive has unit norm, and we also
normalize the contracted shell as a whole.

FP: All functions are pure. They operate on numpy/jax arrays and have no
side effects.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def double_factorial(n: int) -> int:
    """Compute the double factorial n!! = n * (n-2) * (n-4) * ... .

    By convention, (-1)!! == 0!! == 1!! == 1.
    """
    if n <= 0:
        return 1
    result = 1
    k = n
    while k > 1:
        result *= k
        k -= 2
    return result


def primitive_norm(alpha: float, l: int) -> float:
    """Normalization factor for a Cartesian primitive with total angular momentum l.

    For a specific Cartesian component (lx, ly, lz) with lx+ly+lz = l, the
    primitive is
        g(r) = N * x^lx y^ly z^lz exp(-alpha r^2).
    The standard choice is
        N = (2 alpha / pi)^(3/4) * sqrt((4 alpha)^l / ((2 lx - 1)!! (2 ly - 1)!! (2 lz - 1)!!))
    For the common case lx = l, ly = lz = 0 (i.e. the "pure l" component used
    by most normalization conventions), this simplifies to
        N = (2 alpha / pi)^(3/4) * sqrt((4 alpha)^l / (2 l - 1)!!).
    We adopt the latter convention so every primitive in a shell shares the
    same per-alpha normalization factor. Spherical harmonic projection is a
    linear combination of Cartesian components and preserves normalization.

    FP: Pure.
    """
    base = (2.0 * alpha / math.pi) ** 0.75
    if l == 0:
        return float(base)
    numerator = (4.0 * alpha) ** l
    denom = double_factorial(2 * l - 1)
    return float(base * math.sqrt(numerator / denom))


def contraction_norm(
    exponents: np.ndarray, coeffs: np.ndarray, l: int
) -> float:
    """Compute the overall normalization constant for a contracted shell.

    Given exponents ``{alpha_i}`` and already primitive-normalized
    coefficients ``{c_i = d_i * N_i}``, the contracted shell
        chi(r) = sum_i c_i exp(-alpha_i r^2) * (angular part)
    has self-overlap
        <chi|chi> = sum_ij c_i c_j * (pi / (alpha_i + alpha_j))^(3/2)
                    * (2 l - 1)!! / (2 (alpha_i + alpha_j))^l.
    The returned factor ``M`` satisfies ``<M chi | M chi> = 1``.

    FP: Pure. Accepts numpy arrays (not traced JAX) because this runs at
    basis-construction time, not inside jit.
    """
    exponents = np.asarray(exponents, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    alpha_sum = exponents[:, None] + exponents[None, :]
    prefactor = (math.pi / alpha_sum) ** 1.5
    angular = double_factorial(2 * l - 1) / (2.0 * alpha_sum) ** l if l > 0 else 1.0
    overlap_matrix = coeffs[:, None] * coeffs[None, :] * prefactor * angular
    total = float(np.sum(overlap_matrix))
    if total <= 0.0:
        raise ValueError(
            f"Non-positive self-overlap ({total}) for contracted shell with l={l}."
        )
    return 1.0 / math.sqrt(total)


def normalize_shell(
    exponents: np.ndarray, coeffs: np.ndarray, l: int
) -> np.ndarray:
    """Return coefficients combining primitive- and shell-level normalization.

    Input ``coeffs`` are the raw BSE contraction coefficients ``d_i``. Output
    is ``d_i * N_i * M`` where ``N_i`` normalizes each primitive and ``M``
    normalizes the contracted shell.

    FP: Pure.
    """
    exponents = np.asarray(exponents, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    prim_norms = np.array([primitive_norm(float(a), l) for a in exponents])
    c_pn = coeffs * prim_norms
    M = contraction_norm(exponents, c_pn, l)
    return c_pn * M
