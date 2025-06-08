# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

# Fixed-step Runge-Kutta 4th order integrator
# Based on a simplified version of JAX Dormand-Prince integrator:
# https://github.com/google/jax/blob/main/jax/experimental/ode.py

from functools import partial

import jax
import jax.numpy as jnp
from jax._src import core
from jax import lax
from jax._src.util import safe_map, safe_zip
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves
from jax._src import linear_util as lu

map = safe_map
zip = safe_zip


def ravel_first_arg(f, unravel):
    return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped


@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


def runge_kutta_step(func, y0, t0, dt):
    # RK4 Butcher tableaux
    alpha = jnp.array([0, 1 / 2, 1 / 2, 1], dtype=dt.dtype)
    beta = jnp.array(
        [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]], dtype=y0.dtype
    )
    c_sol = jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=dt.dtype)

    def body_fun(i, k):
        ti = t0 + dt * alpha[i]
        yi = y0 + dt.astype(y0.dtype) * jnp.dot(beta[i, :], k)
        ft = func(yi, ti)
        return k.at[i, :].set(ft)

    k = jnp.zeros((4, y0.shape[0]), y0.dtype)
    k = lax.fori_loop(0, 4, body_fun, k)

    y1 = dt.astype(y0.dtype) * jnp.dot(c_sol, k) + y0
    return y1


def odeint(func, y0, t, *args, **kwargs):
    # NOTE: kwargs ignored - mostly pertain to adaptive step size.  Should handle
    # this in a clearer way
    for arg in tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            raise TypeError(
                f"The contents of odeint *args must be arrays or scalars, but got {arg}."
            )
    if not jnp.issubdtype(t.dtype, jnp.floating):
        raise TypeError(f"t must be an array of floats, but got {t}.")

    jnp.allclose(jnp.diff(t), t[1] - t[0]), "t must be evenly spaced"

    # converted, consts = custom_derivatives.closure_convert(func, y0, t[0], *args)
    # return _odeint_wrapper(converted, rtol, atol, mxstep, hmax, y0, t, *args, *consts)
    return _odeint_wrapper(func, y0, t, *args)


@partial(jax.jit, static_argnums=(0,))
def _odeint_wrapper(func, y0, ts, *args):
    y0, unravel = ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out = _odeint(func, y0, ts, *args)
    return jax.vmap(unravel)(out)


# @partial(jax.custom_vjp, nondiff_argnums=(0,))
def _odeint(func, y0, ts, *args):
    def func_(y, t):
        return func(y, t, *args)

    # TODO: If this works, allow a clearer specification of dt
    # in terms of hmin, etc
    dt = ts[1] - ts[0]

    def body_fun(i, ys):
        next_y = runge_kutta_step(func_, ys[i], ts[i], dt)
        return ys.at[i + 1, :].set(next_y)

    ys = jnp.zeros((len(ts), *y0.shape), y0.dtype).at[0, :].set(y0)
    ys = lax.fori_loop(0, len(ts), body_fun, ys)
    return ys
