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

import pytest
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from scipy.optimize import minimize, root
from jax.scipy.optimize import minimize as jax_minimize

from collimator.models import Pendulum, PlanarQuadrotor


def make_pendulum(x0=[0.0, 0.0], g=9.81, L=1.0, b=0.0):
    model = Pendulum(x0=x0, g=g, L=L, b=b, input_port=True)
    model.input_ports[0].fix_value(jnp.array([0.0]))
    return model, model.create_context()


def make_quadrotor(m=1.0, I_B=1.0, r=0.5, g=9.81):
    model = PlanarQuadrotor(m=m, I_B=I_B, r=r, g=g)
    model.input_ports[0].fix_value(jnp.array([0.0, 0.0]))
    return model, model.create_context()


def pendulum_ode(system, context, x0):
    perturbed_context = context.with_continuous_state(x0)
    xdot = system.eval_time_derivatives(perturbed_context)
    return xdot


def pendulum_cost(system, context, x0):
    xdot = pendulum_ode(system, context, x0)
    return jnp.dot(xdot, xdot)


def quadrotor_ode(system, context, unflatten, vec):
    x0, u0 = unflatten(vec)
    context = context.with_continuous_state(x0)
    system.input_ports[0].fix_value(u0)
    xdot = system.eval_time_derivatives(context)
    return xdot


def quadrotor_cost(system, context, unflatten, vec):
    xdot = quadrotor_ode(system, context, unflatten, vec)
    return jnp.dot(xdot, xdot)


@pytest.mark.parametrize("x_eq", [jnp.array([0.0, 0.0]), jnp.array([np.pi, 0.0])])
def test_trim_pendulum(x_eq):
    model, context = make_pendulum()
    func = jax.jit(partial(pendulum_cost, model, context))

    x0 = x_eq + jnp.array([0.1, 0.1])

    # With jax.scipy.optimize.minimize
    res = jax_minimize(func, x0, method="BFGS")
    print(res.success)
    print(res.x)
    print(x0)
    print(func(x0))
    print(func(res.x))
    assert jnp.allclose(res.x, x_eq, atol=1e-6)

    # With scipy.optimize.minimize
    res = minimize(func, x0, jac=jax.jacfwd(func), method="BFGS")
    assert jnp.allclose(res.x, x_eq, atol=1e-6)

    # With scipy.optimize.root
    func = jax.jit(partial(pendulum_ode, model, context))
    res = root(func, x0, jac=jax.jacfwd(func))
    assert jnp.allclose(res.x, x_eq)


def test_trim_quadrotor():
    m = 1.0
    g = 9.81
    model, context = make_quadrotor(m=m, g=g)

    x0 = jnp.zeros(6)
    u0 = jnp.zeros(2)

    # Combine the two vectors into one and get a function to retrieve the two
    # This approach should also work for tree-structured state data.
    vec, unflatten = ravel_pytree((x0, u0))

    func = jax.jit(partial(quadrotor_cost, model, context, unflatten))

    # With jax.scipy.optimize.minimize
    res = jax_minimize(func, vec, method="BFGS")
    x_eq, u_eq = unflatten(res.x)
    assert jnp.allclose(x_eq, jnp.zeros(6), atol=1e-6)
    assert jnp.allclose(u_eq, jnp.array([0.5 * g / m, 0.5 * g / m]), atol=1e-6)

    # With scipy.optimize.minimize
    res = minimize(func, vec, jac=jax.jacfwd(func), method="BFGS")
    x_eq, u_eq = unflatten(res.x)
    assert jnp.allclose(x_eq, jnp.zeros(6), atol=1e-6)
    assert jnp.allclose(u_eq, jnp.array([0.5 * g / m, 0.5 * g / m]), atol=1e-6)

    # With scipy.optimize.root
    # Note we have to use Levenberg-Marquardt because the input and output have
    # different dimensions
    #
    # FIXME: Stack overflow is offline, but why won't this work?
    # func = jax.jit(partial(quadrotor_ode, model, context, unflatten))
    # res = root(func, vec, jac=jax.jacfwd(func), method='lm')
    # x_eq, u_eq = unflatten(res.x)
    # assert jnp.allclose(x_eq, jnp.zeros(6), atol=1e-6)
    # assert jnp.allclose(u_eq, jnp.array([0.5*g/m, 0.5*g/m]), atol=1e-6)
