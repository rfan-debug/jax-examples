"""Tests for the Boys function F_n(t)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_qc.integrals.boys import boys_f0, boys_fn


def _f0_ref(t: float) -> float:
    """Reference F_0(t) from the closed-form expression; accurate for t > 0."""
    if t == 0.0:
        return 1.0
    return 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t))


def test_boys_f0_at_zero():
    assert float(boys_f0(0.0)) == pytest.approx(1.0, abs=1e-14)


def test_boys_f0_series_region():
    # For |t| < _SMALL_T we take the Taylor series; must still match erf branch.
    for t in (1e-6, 1e-4, 1e-3, 5e-3):
        assert float(boys_f0(t)) == pytest.approx(_f0_ref(t), abs=1e-14)


def test_boys_f0_large_t():
    for t in (0.5, 1.0, 5.0, 20.0, 100.0):
        assert float(boys_f0(t)) == pytest.approx(_f0_ref(t), rel=1e-12, abs=1e-14)


def test_boys_f0_monotone_decreasing():
    ts = np.linspace(0.0, 10.0, 100)
    vals = np.array([float(boys_f0(t)) for t in ts])
    # F_0 is strictly decreasing.
    assert np.all(np.diff(vals) < 0)


def test_boys_f0_vectorized():
    t = jnp.linspace(0.0, 5.0, 50)
    vals = boys_f0(t)
    assert vals.shape == t.shape
    # First element is at t=0 => F_0 = 1.
    assert float(vals[0]) == pytest.approx(1.0)


def test_boys_f0_jit_compatible():
    jit_f0 = jax.jit(boys_f0)
    for t in (0.0, 0.001, 1.0, 10.0):
        assert float(jit_f0(t)) == pytest.approx(float(boys_f0(t)), abs=1e-14)


def test_boys_fn_zero_at_zero():
    # F_n(0) = 1 / (2n + 1).
    for n in range(5):
        assert float(boys_fn(n, 0.0)) == pytest.approx(1.0 / (2 * n + 1), abs=1e-14)


def test_boys_fn_recurrence_matches_direct():
    # For n=1 the recurrence gives F_1(t) = (F_0 - exp(-t)) / (2t).
    for t in (0.1, 1.0, 5.0):
        exp_neg_t = math.exp(-t)
        f0 = _f0_ref(t)
        f1_direct = (f0 - exp_neg_t) / (2.0 * t)
        assert float(boys_fn(1, t)) == pytest.approx(f1_direct, rel=1e-12, abs=1e-14)


def test_boys_fn_rejects_negative_n():
    with pytest.raises(ValueError):
        boys_fn(-1, 1.0)
