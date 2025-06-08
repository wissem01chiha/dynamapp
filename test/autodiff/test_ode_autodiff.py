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

import collimator
from collimator.library import (
    Integrator,
    Gain,
)

from collimator.testing import fd_grad

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("method", ["dopri5", "bdf"])
def test_scalar_linear(method):
    # Tests adjoint simulation of a simple continuous-time system
    # dx/dt = -a * x
    #
    # Exact forward solution:
    #  x(T) = x0 * exp(-a * T)
    #
    # Exact gradient:
    #  dx/dx0 = exp(-a * T)

    a = 1.5
    builder = collimator.DiagramBuilder()
    Gain_0 = builder.add(Gain(-a, name="Gain_0"))
    Integrator_0 = builder.add(Integrator(0.0, name="Integrator_0"))

    builder.connect(Gain_0.output_ports[0], Integrator_0.input_ports[0])
    builder.connect(Integrator_0.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()
    t0, t1 = 0.0, 2.0

    int_context = context[Integrator_0.system_id].with_continuous_state(jnp.array(4.0))
    context = context.with_subcontext(Integrator_0.system_id, int_context)

    options = collimator.SimulatorOptions(
        math_backend="jax",
        atol=1e-10,
        rtol=1e-8,
        enable_autodiff=True,
        ode_solver_method=method,
    )

    @jax.jit
    def fwd(x0, context):
        context = context.with_continuous_state(x0)
        results = collimator.simulate(diagram, context, (t0, t1), options=options)
        return results.context[Integrator_0.system_id].continuous_state

    grad_fwd = jax.jit(jax.grad(fwd))

    x0 = context.continuous_state
    xf = fwd(x0, context)  # [Gain, Integrator]

    assert jnp.allclose(xf, x0[0] * jnp.exp(-a * t1))
    dx0 = grad_fwd(x0, context)
    assert jnp.allclose(dx0[0], jnp.exp(-a * t1))


@pytest.mark.parametrize("method", ["dopri5", "bdf"])
def test_vector_linear(method):
    # Tests adjoint simulation of a linear vector-valued system
    # dx/dt = -a * x
    #
    # Exact forward solution:
    #  x(T) = x0 * exp(-a * T)
    #
    # Exact gradient:
    #  dx/dx0 = exp(-a * T)

    a = 1.5
    builder = collimator.DiagramBuilder()
    Gain_0 = builder.add(Gain(-a, name="Gain_0"))
    Integrator_0 = builder.add(Integrator(jnp.zeros(2), name="Integrator_0"))

    builder.connect(Gain_0.output_ports[0], Integrator_0.input_ports[0])
    builder.connect(Integrator_0.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()
    t0, t1 = 0.0, 2.0

    options = collimator.SimulatorOptions(
        math_backend="jax",
        atol=1e-10,
        rtol=1e-8,
        enable_autodiff=True,
        ode_solver_method=method,
    )

    @jax.jit
    def fwd(x0, context):
        int_context = context[Integrator_0.system_id].with_continuous_state(x0)
        context = context.with_subcontext(Integrator_0.system_id, int_context)
        results = collimator.simulate(diagram, context, (t0, t1), options=options)
        return results.context[Integrator_0.system_id].continuous_state

    jac = jax.jit(jax.jacrev(fwd))

    x0 = jnp.array([4.0, 5.0])

    assert jnp.allclose(fwd(x0, context), x0 * jnp.exp(-a * t1))
    dx0 = jac(x0, context)
    assert jnp.allclose(jnp.diag(dx0), jnp.exp(-a * t1))


@pytest.mark.parametrize("method", ["dopri5", "bdf"])
def test_scalar_nonlinear(method):
    # Tests adjoint simulation of a simple nonlinear system
    # dx/dt = -a * x^2
    #
    # Exact forward solution:
    #  x(T) = x0 / (1 + a * x0 * T)
    #
    # Exact gradient:
    #  dx/dx0 = 1 / (1 + a * x0 * T)^2
    #  dx/da = -x0**2 * T / (1 + a * x0 * T)^2

    class ScalarNonlinear(collimator.LeafSystem):
        def __init__(self, *args, a=1.0, **kwargs):
            super().__init__(*args, **kwargs)

            self.declare_dynamic_parameter("a", a)
            self.declare_continuous_state(shape=(), ode=self._ode)

        def _ode(self, time, state, **params):
            x = state.continuous_state
            return -params["a"] * x**2

    a = 1.5
    system = ScalarNonlinear(a=a)
    context = system.create_context()
    t0, t1 = 0.0, 2.0

    context = context.with_continuous_state(jnp.array(4.0))

    options = collimator.SimulatorOptions(
        math_backend="jax",
        rtol=1e-10,
        atol=1e-12,
        enable_autodiff=True,
        ode_solver_method=method,
    )

    @jax.jit
    def fwd(x0, a, context):
        context = context.with_continuous_state(x0)
        context.parameters["a"] = a
        results = collimator.simulate(system, context, (t0, t1), options=options)
        return results.context.continuous_state

    grad_fwd = jax.jit(jax.grad(fwd, argnums=(0, 1)))

    x0 = context.continuous_state
    xf = fwd(x0, a, context)  # [Gain, Integrator]

    assert jnp.allclose(xf, x0 / (1 + a * x0 * t1))
    dx0, da = grad_fwd(x0, a, context)

    dx0_exact = 1 / (1 + a * x0 * t1) ** 2
    da_exact = -(x0**2) * t1 / (1 + a * x0 * t1) ** 2
    print(f"{dx0=}, {da=}")
    print(f"{dx0_exact=}, {da_exact=}")

    assert jnp.allclose(dx0, dx0_exact)
    assert jnp.allclose(da, da_exact)


@pytest.mark.parametrize("method", ["dopri5", "bdf"])
def test_vector_nonlinear(method):
    # Test adjoint simulation of a nonlinear vector-valued system.
    #
    # The system is a falling object with quadratic drag.
    # dx/dt = v
    # dv/dt = -g - a * v^2
    #
    # Compare against finite-difference approximation of the gradient.

    class VectorNonlinear(collimator.LeafSystem):
        def __init__(self, *args, g=9.81, a=0.1, **kwargs):
            super().__init__(*args, **kwargs)

            self.declare_dynamic_parameter("g", g)
            self.declare_dynamic_parameter("a", a)
            self.declare_continuous_state(shape=(2,), ode=self._ode)

        def _ode(self, time, state, **params):
            g = params["g"]
            a = params["a"]
            x, v = state.continuous_state
            return jnp.array([v, -g - a * v**2 * jnp.sign(v)])

    g = 9.81
    a = 0.1
    system = VectorNonlinear(g=g, a=a)
    context = system.create_context()
    t0, t1 = 0.0, 0.5

    # Simulate and compare derivative against finite difference.
    options = collimator.SimulatorOptions(
        math_backend="jax",
        rtol=1e-10,
        atol=1e-12,
        enable_autodiff=True,
        ode_solver_method=method,
    )

    @jax.jit
    def fwd(x0, g, a, context):
        context = context.with_continuous_state(x0)
        context.parameters["g"] = g
        context.parameters["a"] = a
        results = collimator.simulate(system, context, (t0, t1), options=options)
        return results.context.continuous_state[0]

    grad_fwd = jax.jit(jax.grad(fwd, argnums=(0, 1, 2)))

    x0 = np.array([4.0, 5.0])
    dx0_ad, dg_ad, da_ad = grad_fwd(x0, g, a, context)

    dx0_fd, dg_fd, da_fd = fd_grad(partial(fwd, context=context), x0, g, a)

    assert jnp.allclose(dx0_ad, dx0_fd)
    assert jnp.allclose(dg_ad, dg_fd)
    assert jnp.allclose(da_ad, da_fd)


def test_mass_matrix():
    A0 = np.array([[-1.0, 0.0], [0.0, -1.0]])
    B0 = np.array([[0.0], [1.0]])

    class VectorLinear(collimator.LeafSystem):
        # M @ ẋ = A @ x + B @ u

        def __init__(self, mass_matrix=None, A=None, B=None, x0=None, name=None):
            super().__init__(name=name)
            if A is None:
                A = A0
            if B is None:
                B = B0
            self.declare_dynamic_parameter("A", A)
            self.declare_dynamic_parameter("B", B)

            if x0 is None:
                x0 = np.zeros(A.shape[0])

            self.declare_continuous_state(
                default_value=x0,
                ode=self.ode,
                mass_matrix=mass_matrix,
            )
            self.declare_input_port(name="u_in")
            self.declare_continuous_state_output(name="x")

        def ode(self, time, state, u, **parameters):
            A, B = parameters["A"], parameters["B"]
            x = state.continuous_state
            return A @ x + B @ u

    M = np.diag([1.0, 0.0])
    u0 = np.array([2.0])
    x0 = np.array([1.0, 2.0])
    system = VectorLinear(mass_matrix=M)
    system.input_ports[0].fix_value(u0)

    context = system.create_context()
    t0 = 0.0
    t1 = 2.0
    options = collimator.SimulatorOptions(
        ode_solver_method="bdf",
        enable_autodiff=True,
        rtol=1e-10,
        atol=1e-12,
    )

    @jax.jit
    def fwd(x0, B, context):
        # Ensure that the initial condition is consistent
        # This is necessary for finite differencing, but not autodiff
        # Otherwise the finite differencing perturbations will lead to
        # inconsistent initial conditions
        x0 = jnp.array(x0).at[1].set(B[1, 0] * u0[0])
        context = context.with_continuous_state(x0).with_parameters({"B": B})
        results = collimator.simulate(system, context, (t0, t1), options=options)
        return results.context.continuous_state[0]

    grad_fwd = jax.jit(jax.grad(fwd, argnums=(0, 1)))

    dx0, dB = grad_fwd(x0, B0, context)
    dx0_fd, dB_fd = fd_grad(partial(fwd, context=context), x0, B0)

    assert np.allclose(dx0, dx0_fd)
    assert np.allclose(dB, dB_fd)


if __name__ == "__main__":
    from collimator import logging

    logging.set_log_level(logging.DEBUG)
    logging.set_file_handler("test.log")

    test_scalar_linear("bdf")
