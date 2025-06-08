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

import jax.numpy as jnp

import collimator
from collimator.framework import DependencyTicket


pytestmark = pytest.mark.minimal


# Test the case where no ODE is provided: the time derivatives should
# automatically be zeroed
class StaticContinuousState(collimator.LeafSystem):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.declare_continuous_state(shape=(2,))
        self.declare_continuous_state_output(name="x")


def test_no_ode():
    model = StaticContinuousState()
    ctx = model.create_context()

    x0 = jnp.array([1.5, 2.5])
    ctx = ctx.with_continuous_state(x0)

    t0, t1 = 0.0, 2.0
    results = collimator.simulate(
        model,
        ctx,
        (t0, t1),
    )
    xf = results.context.continuous_state

    # yf = forward_map(y0)
    assert jnp.allclose(xf, x0)


class ScalarLinear(collimator.LeafSystem):
    #
    # dx/dt = -a * x + u
    #
    def __init__(self, a=1.5, name=None):
        super().__init__(name=name)
        self.a = a

        self.declare_continuous_state(shape=(), ode=self.ode)
        self.declare_input_port(name="u_in")

        # Three kinds of output port dependencies:
        # 1. Continuous state
        self.declare_continuous_state_output(name="x")

        # 2. Time derivatives of continuous state
        self.declare_output_port(
            self.ode,
            name="xdot",
            prerequisites_of_calc=[DependencyTicket.xcdot],
        )

        # 3. Feedthrough
        def _feedthrough_callback(time, state, *inputs, **parameters):
            return inputs[0]

        # TODO: Convenience function for declaring feedthrough ports?
        self.declare_output_port(
            _feedthrough_callback,
            name="u_out",
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def ode(self, time, state, u):
        x = state.continuous_state
        return -self.a * x + u


class TestScalarLinear:
    def test_forward(self):
        a = 1.5
        x0 = 4.0
        u0 = 0.0

        model = ScalarLinear(a=a)
        model.input_ports[0].fix_value(u0)

        ctx = model.create_context()
        ctx = ctx.with_continuous_state(x0)

        # TODO: This should use LogVectorOutput
        t0, t1 = 0.0, 2.0
        results = collimator.simulate(
            model,
            ctx,
            (t0, t1),
        )
        xf = results.context.continuous_state

        # yf = forward_map(y0)
        assert jnp.allclose(xf, x0 * jnp.exp(-a * t1))

    def test_dependencies(self):
        from collimator.framework.dependency_graph import mark_cache

        a = 1.5
        x0 = 4.0
        u0 = 0.0

        model = ScalarLinear(a=a)

        # Input port dependency should be empty
        assert model.input_ports[0].prerequisites_of_calc == []

        # Continuous state output port should depend on the composite xc
        assert model.output_ports[0].prerequisites_of_calc == [DependencyTicket.xc]

        # Time derivative output port should depend on the composite xcdot
        assert model.output_ports[1].prerequisites_of_calc == [DependencyTicket.xcdot]

        # Feedthrough output port should depend directly on the input port
        assert model.output_ports[2].prerequisites_of_calc == [
            model.input_ports[0].ticket
        ]

        # Check that the "dependency probe" works
        model.input_ports[0].fix_value(u0)
        assert model.input_ports[0].prerequisites_of_calc == [DependencyTicket.nothing]

        ctx = model.create_context()
        ctx = ctx.with_continuous_state(x0)

        # Perturbing the input should note the change in the feedthrough output
        input_port = model.input_ports[0]
        output_port = model.output_ports[2]

        # Mark the feedthrough output cache as up to date
        cache = model._create_dependency_cache()
        cache = mark_cache(cache, output_port.callback_index, is_out_of_date=False)

        # Invalidate the input port cache
        input_tracker = model.dependency_graph[input_port.ticket]
        cache = input_tracker.notify_subscribers(cache, model.dependency_graph)

        # Check that the feedthrough output cache is now out of date
        assert cache[output_port.callback_index].is_out_of_date

    def test_feedthrough_pairs(self):
        a = 1.5
        u0 = 0.0

        model = ScalarLinear(a=a)
        model.input_ports[0].fix_value(u0)

        # Expect to find one feedthrough pairs: input[0] -> output[2]
        feedthrough_pairs = model.get_feedthrough()
        assert feedthrough_pairs == [(0, 2)]
