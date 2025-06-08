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

import jax
import jax.numpy as jnp

import collimator
from collimator.library import (
    Integrator,
    Gain,
    Adder,
)

pytestmark = pytest.mark.slow


class Staircase(collimator.LeafSystem):
    """Simple discrete-time system with periodic events

    Update rule:
        x[n+1] = x[n] + 1.0
        y[n]   = x[n]
    """

    def __init__(self, name="staircase", period=1.0):
        super().__init__(name=name)

        self.declare_discrete_state(default_value=0.0)

        self.declare_output_port(
            self.output,
            period=period,
            offset=0.0,
            name="staircase:y",
        )
        self.declare_periodic_update(
            self.update,
            period=period,
            offset=0.0,
        )

    # x[n+1] = x[n] + 1.0
    def update(self, time, state, *inputs):
        x = state.discrete_state
        return x + 1.0

    # y[n] = x[n]
    def output(self, time, state, *inputs):
        return state.discrete_state


def test_scalar_linear_hybrid():
    # Test adjoint simulation of a simple hybrid system with periodic events
    # u[n+1] = u[n] + 1
    # dx/dt = -a * (x - u)
    #
    # The exact solution can be determined recursively:
    # x[n+1] = x[n] * exp(-a * dt) + u[n] * (1 - exp(-a * dt))
    #
    # The exact gradient is:
    # dx[N]/dx[0] = exp(-a * N * dt)

    a = -1.5
    tau = 1.0

    builder = collimator.DiagramBuilder()
    Gain_0 = builder.add(Gain(-a, name="Gain_0"))
    Integrator_0 = builder.add(Integrator(0.0, name="Integrator_0"))
    Adder_0 = builder.add(Adder(2, name="Adder_0", operators="+-"))
    Staircase_0 = builder.add(Staircase(period=tau, name="Staircase_0"))

    builder.connect(Integrator_0.output_ports[0], Adder_0.input_ports[0])
    builder.connect(Staircase_0.output_ports[0], Adder_0.input_ports[1])
    builder.connect(Adder_0.output_ports[0], Gain_0.input_ports[0])
    builder.connect(Gain_0.output_ports[0], Integrator_0.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    options = collimator.SimulatorOptions(
        math_backend="jax",
        atol=1e-10,
        rtol=1e-8,
        enable_autodiff=True,
        max_major_steps=100,
    )

    @jax.jit
    def fwd_sim(x0, context, tf):
        # context = context.with_continuous_state(x0)
        int_context = context[Integrator_0.system_id].with_continuous_state(x0)
        context = context.with_subcontext(Integrator_0.system_id, int_context)
        results = collimator.simulate(diagram, context, (0.0, tf), options=options)
        return results.context[Integrator_0.system_id].continuous_state

    @partial(jax.jit, static_argnums=(1,))
    def fwd_exact(x0, N):
        x = x0
        for n in range(N):
            x = x * jnp.exp(-a * tau) + n * (1 - jnp.exp(-a * tau))
        return x

    def grad_exact(x0, N):
        return jnp.exp(-a * N * tau)

    x0 = jnp.array(4.0)
    N = 3  # Number of staircase steps
    tf = N * tau

    xf = fwd_sim(x0, context, tf)
    assert jnp.allclose(xf, fwd_exact(x0, N))

    dxf = jax.jit(jax.grad(fwd_sim))(x0, context, tf)
    assert jnp.allclose(dxf, grad_exact(x0, N))
