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

"""RK4 integration utilities"""

# TODO: Integrate with utilities present in the simulator

from functools import partial
import jax


@partial(jax.jit, static_argnames=("ode_rhs",))
def _rk4_step_constant_u(t, x, u, dh, ode_rhs):
    """
    Performs a single step of the Runge-Kutta 4 (RK4) integration method for a given
    system with a constant control input.

    Parameters:
    t: float
        Current time at the beginning of the RK4 step.

    x: Array
        Current state vector of the system at the beginning of the RK4 step.

    u: Array
        Fixed control input that is applied throughout the RK4 step.

    dh: float
        Time step duration for this RK4 step.

    ode_rhs: Callable
        Callable computing the right-hand side (RHS) of the Ordinary
        Differential Equation (ODE). The function should have the
        signature: ode_rhs(x, u, t) --> dot_x.

    Returns:
    ndarray: The state vector of the system after completing the RK4 step.
    """

    k1 = ode_rhs(x, u, t)
    k2 = ode_rhs(x + dh / 2.0 * k1, u, t + dh / 2.0)
    k3 = ode_rhs(x + dh / 2.0 * k2, u, t + dh / 2.0)
    k4 = ode_rhs(x + dh * k3, u, t + dh)

    x_next = x + (dh / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_next


@partial(jax.jit, static_argnames=("ode_rhs", "nh"))
def rk4_major_step_constant_u(t0, x0, u, dt, nh, ode_rhs):
    """
    Integrates the system over one major step using the Runge-Kutta 4 (RK4) method,
    with a specified number of minor steps within this major step. The control input
    is assumed to be constant over the entire step.

    Parameters:
    t0: float
        Initial time at the beginning of the integration step.

    x0: Array
        Intial state vector of the system at the beginning of the integration step.

    u: Array
        Fixed control input that is applied throughout the integration step.

    dt: float
        Total duration of the major step.

    nh: int
        Number of minor steps within each major step. The minor step size is
        calculated as dh = dt / nh.

    ode_rhs: Callable
        Callable computing the right-hand side (RHS) of the Ordinary
        Differential Equation (ODE). The function should have the
        signature: ode_rhs(x, u, t) --> dot_x.

    Returns:
    Array: The state of the system at the end of the major integration step.
    """

    minor_step_size = dt / nh  # Minor step size

    def body_fun(i, x_current):
        t_minor = t0 + i * minor_step_size
        return _rk4_step_constant_u(t_minor, x_current, u, minor_step_size, ode_rhs)

    x_current = jax.lax.fori_loop(0, nh, body_fun, x0)

    return x_current
