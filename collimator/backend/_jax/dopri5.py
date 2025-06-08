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

"""JAX-based Dormand-Prince ODE integration with adaptive stepsize.

Modified from the built-in JAX `odeint` function:
https://github.com/google/jax/blob/main/jax/experimental/ode.py

This version is modified to return a dense interpolant as well as the time points.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import lax

from collimator.backend.typing import Array
from .ode_solver_impl import ODESolverImpl, ODESolverState

if TYPE_CHECKING:
    from ...framework.state import StateComponent

__all__ = [
    "Dopri5State",
    "Dopri5Solver",
]


def _raise_end_time_not_reached(tf, t):
    if (tf - t) / tf > 1e-3:
        raise RuntimeError(
            f"ODE solver failed to reach specified end time. End time={tf}. "
            f"Reached time={t}. This may also be an indication that the system is "
            "diverging, or that the dynamics are very stiff."
        )


@jax.jit
def error_ode_end_time_not_reached(tf, t):
    jax.debug.callback(_raise_end_time_not_reached, tf, t)


def interp_fit_dopri(y0, y1, k, dt):
    # Fit a polynomial to the results of a Runge-Kutta step.
    dps_c_mid = jnp.array(
        [
            6025192743 / 30085553152 / 2,
            0,
            51252292925 / 65400821598 / 2,
            -2691868925 / 45128329728 / 2,
            187940372067 / 1594534317056 / 2,
            -1776094331 / 19743644256 / 2,
            11237099 / 235043384 / 2,
        ],
        dtype=y0.dtype,
    )
    y_mid = y0 + dt.astype(y0.dtype) * jnp.dot(dps_c_mid, k)
    return jnp.asarray(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))


def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
    dt = dt.astype(y0.dtype)
    a = -2.0 * dt * dy0 + 2.0 * dt * dy1 - 8.0 * y0 - 8.0 * y1 + 16.0 * y_mid
    b = 5.0 * dt * dy0 - 3.0 * dt * dy1 + 18.0 * y0 + 14.0 * y1 - 32.0 * y_mid
    c = -4.0 * dt * dy0 + dt * dy1 - 11.0 * y0 - 5.0 * y1 + 16.0 * y_mid
    d = dt * dy0
    e = y0
    return a, b, c, d, e


def runge_kutta_step(func, y0, f0, t0, dt):
    # Dopri5 Butcher tableaux
    alpha = jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0, 0], dtype=dt.dtype)
    beta = jnp.array(
        [
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ],
        dtype=f0.dtype,
    )
    c_sol = jnp.array(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=f0.dtype
    )
    c_error = jnp.array(
        [
            35 / 384 - 1951 / 21600,
            0,
            500 / 1113 - 22642 / 50085,
            125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400,
            11 / 84 - 649 / 6300,
            -1.0 / 60.0,
        ],
        dtype=f0.dtype,
    )

    def body_fun(i, k):
        ti = t0 + dt * alpha[i - 1]
        yi = y0 + dt.astype(f0.dtype) * jnp.dot(beta[i - 1, :], k)
        ft = func(yi, ti)
        return k.at[i, :].set(ft)

    k = jnp.zeros((7, f0.shape[0]), f0.dtype).at[0, :].set(f0)
    k = lax.fori_loop(1, 7, body_fun, k)

    y1 = dt.astype(f0.dtype) * jnp.dot(c_sol, k) + y0
    y1_error = dt.astype(f0.dtype) * jnp.dot(c_error, k)
    f1 = k[-1]
    return y1, f1, y1_error, k


@dataclasses.dataclass
class Dopri5State(ODESolverState):
    # `t_return` is the time value to return in time series.  Will be inf when
    # the end time is reached.  Otherwise it should match `t`.
    t_return: float = None
    interp_coeff: Array = None  # Current dense interpolation coefficients
    n_acc: int = 0  # Number of accepted steps
    n_rej: int = 0  # Number of rejected steps
    accepted: bool = False  # Whether the most recent attempted step was accepted
    unravel: Callable = None  # Unravel the flattened vector to the original pytree

    def __post_init__(self):
        if self.t_return is None:
            self.t_return = self.t
        if self.interp_coeff is None:
            self.interp_coeff = jnp.array([self.y] * 5)
        if self.t_prev is None:
            self.t_prev = self.t

    # Inherits docstring from `ODESolverState`
    def eval_interpolant(self, t_eval: float) -> Array:
        if self.unravel is None:
            raise ValueError("Unravel function not set: cannot evaluate interpolant.")
        t, last_t = self.t, self.t_prev
        relative_time = (t_eval - last_t) / (t - last_t)
        return self.unravel(jnp.polyval(self.interp_coeff, relative_time))

    # Inherits docstring from `ODESolverState`
    @property
    def unraveled_state(self) -> StateComponent:
        return self.unravel(self.y)

    def ravel(self, x: StateComponent) -> Array:
        return jnp.concatenate([jnp.ravel(e) for e in jax.tree_util.tree_leaves(x)])

    # Inherits docstring from `ODESolverState`
    def update(
        self, error_ratio, next_y, next_f, next_t, dt, new_interp_coeff
    ) -> Dopri5State:
        """Conditionally update the state depending on the error ratio."""
        new = [
            next_y,
            next_t,
            next_f,
            dt,
            self.t,
            next_t,
            new_interp_coeff,
            self.n_acc + 1,
            self.n_rej,
            True,
        ]
        old = [
            self.y,
            self.t,
            self.f,
            dt,
            self.t_prev,
            self.t,
            self.interp_coeff,
            self.n_acc,
            self.n_rej + 1,
            False,
        ]
        return Dopri5State(
            *jax.tree.map(partial(jnp.where, (error_ratio <= 1.0)), new, old),
            self.unravel,
        )

    @property
    def step_variables(self):
        return self.y, self.f, self.t, self.dt

    @property
    def return_variables(self):
        return self.t_return, self.y, self.interp_coeff

    def tree_flatten(self):
        children = (
            self.y,
            self.t,
            self.f,
            self.dt,
            self.t_prev,
            self.t_return,
            self.interp_coeff,
            self.n_acc,
            self.n_rej,
            self.accepted,
        )
        aux_data = (self.unravel,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (unravel,) = aux_data
        return cls(*children, unravel)


jax.tree_util.register_pytree_node(
    Dopri5State,
    lambda state: state.tree_flatten(),
    Dopri5State.tree_unflatten,
)


class Dopri5Solver(ODESolverImpl):
    """JAX-based Dormand-Prince (4)5 ODE integration with adaptive stepsize.

    Modified from the built-in JAX `odeint` function:
    https://github.com/google/jax/blob/main/jax/experimental/ode.py

    This version is modified to return a dense interpolant as well as the time points.
    It is also modified to work with `SystemBase` and `ContextBase` objects instead of
    raw arrays and functions.
    """

    def _initialize_state(self, func, t0, xc0, _mass, unravel, *args, dt=None):
        # Initialize the solver state.
        order = 4
        f0 = func(xc0, t0, *args)
        if dt is None:
            dt = self.initial_step_size(func, xc0, t0, order, f0, *args)
        return Dopri5State(xc0, t0, f0, dt, unravel=unravel)

    def mean_error_ratio(self, error_estimate, y0, y1):
        err_tol = self.atol + self.rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
        err_ratio = error_estimate / err_tol.astype(error_estimate.dtype)
        return jnp.sqrt(jnp.mean(err_ratio**2))

    def optimal_step_size(
        self,
        last_step,
        mean_error_ratio,
        safety=0.9,
        ifactor=10.0,
        dfactor=0.2,
        order=5.0,
    ):
        """Compute optimal Runge-Kutta stepsize."""
        dfactor = jnp.where(mean_error_ratio < 1, 1.0, dfactor)

        factor = jnp.minimum(
            ifactor, jnp.maximum(mean_error_ratio ** (-1.0 / order) * safety, dfactor)
        )
        next_step = jnp.where(
            mean_error_ratio == 0, last_step * ifactor, last_step * factor
        )
        return jnp.clip(next_step, a_min=self.hmin, a_max=self.hmax)

    def attempt_rk_step(self, func, boundary_time, solver_state):
        y, f, t, dt = solver_state.step_variables
        dt = jnp.clip(dt, a_min=self.hmin, a_max=(boundary_time - t))
        next_y, next_f, next_y_error, k = runge_kutta_step(func, y, f, t, dt)
        next_t = t + dt
        error_ratio = self.mean_error_ratio(next_y_error, y, next_y)
        new_interp_coeff = interp_fit_dopri(y, next_y, k, dt)
        dt = self.optimal_step_size(dt, error_ratio)
        return solver_state.update(
            error_ratio, next_y, next_f, next_t, dt, new_interp_coeff
        )

    # Inherits docstring from `ODESolverBase`
    def step(self, func, boundary_time, solver_state):
        return lax.while_loop(
            lambda carry: ~carry.accepted,
            partial(self.attempt_rk_step, func, boundary_time),
            dataclasses.replace(solver_state, accepted=False),
        )
