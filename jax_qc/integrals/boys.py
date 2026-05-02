"""Boys function F_n(t).

Definition:

    F_n(t) = integral_0^1  u^(2n) exp(-t u^2)  du

It appears in every molecular integral involving Coulomb-like interactions
(nuclear attraction, ERI). For s-type integrals (Step 2) only F_0 is
required; higher orders are used by the Obara-Saika recurrence in Step 4.

Closed form for F_0:

    F_0(t) = (1/2) sqrt(pi / t) * erf(sqrt(t))       for t > 0
    F_0(0) = 1

The ``erf`` branch loses precision as t -> 0, so we switch to a Taylor
series for small t (|t| < threshold).

Recurrence (upward, numerically stable for the small n used here):

    F_{n+1}(t) = ((2 n + 1) F_n(t) - exp(-t)) / (2 t)

FP: Pure functions, fully jit-compatible, scalar-or-broadcastable input.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_SMALL_T = 1e-2  # switch to Taylor series below this.


def _f0_series(t):
    """Taylor series for F_0(t) around t = 0.

    F_0(t) = sum_{k>=0} (-t)^k / (k! * (2k + 1))
    Keeping terms through t^7 gives relative error < 1e-15 for t <= 1e-2.
    """
    return (
        1.0
        - t / 3.0
        + t * t / 10.0
        - t ** 3 / 42.0
        + t ** 4 / 216.0
        - t ** 5 / 1320.0
        + t ** 6 / 9360.0
        - t ** 7 / 75600.0
    )


def _f0_erf(t):
    # Clamp the argument to avoid 1/sqrt(0) in the dead branch of jnp.where.
    safe_t = jnp.where(t > 0, t, 1.0)
    return 0.5 * jnp.sqrt(jnp.pi / safe_t) * jax.lax.erf(jnp.sqrt(safe_t))


def boys_f0(t):
    """F_0(t) = integral_0^1 exp(-t u^2) du.

    Valid for any t >= 0; uses a Taylor series near zero and the erf
    expression elsewhere.
    """
    t = jnp.asarray(t)
    return jnp.where(t < _SMALL_T, _f0_series(t), _f0_erf(t))


def _fn_series(n: int, t):
    """Taylor series for F_n(t) around t = 0.

    F_n(t) = sum_{k>=0} (-t)^k / (k! * (2 n + 2 k + 1))
    Keeping terms through t^7 gives relative error < 1e-14 for t <= 1e-2.
    """
    result = 1.0 / (2 * n + 1)
    sign = -1.0
    t_pow = t
    fact = 1.0
    for k in range(1, 8):
        fact *= k
        result = result + sign * t_pow / (fact * (2 * n + 2 * k + 1))
        sign = -sign
        t_pow = t_pow * t
    return result


def boys_fn(n: int, t):
    """F_n(t) for a non-negative integer n.

    * For |t| < _SMALL_T we evaluate the direct Taylor series for F_n
      (avoids cancellation in the upward recurrence).
    * For larger t we use the upward recurrence from F_0. That recurrence
      is numerically well-conditioned when t is not tiny and is adequate
      for n up to ~8 (enough for (dd|dd) integrals).
    """
    if n < 0:
        raise ValueError("boys_fn requires n >= 0")
    t = jnp.asarray(t)
    if n == 0:
        return boys_f0(t)

    # Upward recurrence from F_0 for the "large t" branch.
    safe_t = jnp.where(t > 0, t, 1.0)
    exp_neg_t = jnp.exp(-t)
    fn = boys_f0(t)
    for k in range(n):
        next_fn = ((2 * k + 1) * fn - exp_neg_t) / (2.0 * safe_t)
        next_at_zero = 1.0 / (2.0 * (k + 1) + 1.0)
        fn = jnp.where(t > 0, next_fn, next_at_zero)
    return jnp.where(t < _SMALL_T, _fn_series(n, t), fn)
