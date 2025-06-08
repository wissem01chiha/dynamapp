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
from collimator.framework import parameters, Parameter
from collimator.library import Gain

from collimator.logging import logger


pytest.mark.minimal


# Define the system.
class ScalarSystem(collimator.LeafSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_continuous_state(1, ode=self.ode)  # Two state variables.
        self.declare_dynamic_parameter("a", 0.5)

        self.declare_continuous_state_output()

    def ode(self, time, state, **params):
        # g = self.numeric_parameters["g"]
        a = params["a"]
        x = state.continuous_state
        return -a * x


def test_leaf_params():
    model = ScalarSystem()
    model.pprint(logger.debug)

    ctx = model.create_context()

    assert jnp.allclose(ctx.continuous_state, jnp.zeros(1))

    # Set initial state
    x0 = jnp.array([1.5])
    ctx = ctx.with_continuous_state(x0)
    assert jnp.allclose(ctx.continuous_state, x0)

    x0dot_eval = model.eval_time_derivatives(ctx)
    a = ctx.parameters["a"]
    x0dot_true = -a * x0[0]
    assert jnp.allclose(x0dot_eval, x0dot_true)

    # Check that the context didn't change
    assert jnp.allclose(ctx.continuous_state, x0)


def test_leaf_set_params():
    model = ScalarSystem()

    ctx = model.create_context()

    assert jnp.allclose(ctx.continuous_state, jnp.zeros(1))

    # Set initial state
    x0 = jnp.array([1.5])
    ctx = ctx.with_continuous_state(x0)
    assert jnp.allclose(ctx.continuous_state, x0)

    # Test setting parameters
    a = 0.1
    ctx = ctx.with_parameter("a", a)
    assert jnp.allclose(ctx.parameters["a"], a)

    x0dot_eval = model.eval_time_derivatives(ctx)
    x0dot_true = -a * x0[0]
    assert jnp.allclose(x0dot_eval, x0dot_true)


def test_tree_params():
    k = 2.0
    builder = collimator.DiagramBuilder()
    ode = builder.add(ScalarSystem(name="ode"))
    gain = builder.add(Gain(k, name="gain"))

    builder.connect(ode.output_ports[0], gain.input_ports[0])
    builder.export_output(gain.output_ports[0])

    diagram = builder.build()

    ctx = diagram.create_context()

    # Set initial state
    x0 = jnp.array([1.5])
    ode_ctx = ctx[ode.system_id].with_continuous_state(x0)
    ctx = ctx.with_subcontext(ode.system_id, ode_ctx)
    assert jnp.allclose(ctx[ode.system_id].continuous_state, x0)

    x0dot_eval = diagram.eval_time_derivatives(ctx)
    a = ctx[ode.system_id].parameters["a"]
    x0dot_true = -a * x0[0]
    assert jnp.allclose(x0dot_eval[0], x0dot_true)

    # Check that the context didn't change
    assert jnp.allclose(ctx[ode.system_id].continuous_state, x0)

    y = diagram.output_ports[0].eval(ctx)
    assert jnp.allclose(y, k * x0[0])


def test_tree_set_params():
    k = 2.0
    builder = collimator.DiagramBuilder()
    ode = builder.add(ScalarSystem(name="ode"))
    gain = builder.add(Gain(k, name="gain"))

    builder.connect(ode.output_ports[0], gain.input_ports[0])
    builder.export_output(gain.output_ports[0])

    diagram = builder.build()

    ctx = diagram.create_context()

    # Set initial state
    x0 = jnp.array([1.5])
    ode_ctx = ctx[ode.system_id].with_continuous_state(x0)
    ctx = ctx.with_subcontext(ode.system_id, ode_ctx)
    assert jnp.allclose(ctx[ode.system_id].continuous_state, x0)

    # Test setting parameters
    a = 0.1
    ode_ctx = ctx[ode.system_id].with_parameter("a", a)
    ctx = ctx.with_subcontext(ode.system_id, ode_ctx)
    assert jnp.allclose(ctx[ode.system_id].parameters["a"], a)

    # Check thet this actually updated the parameter
    x0dot_eval = diagram.eval_time_derivatives(ctx)
    x0dot_true = -a * x0[0]
    assert jnp.allclose(x0dot_eval[0], x0dot_true)

    # Check that the context didn't change
    assert jnp.allclose(ctx[ode.system_id].continuous_state, x0)

    y = diagram.output_ports[0].eval(ctx)
    assert jnp.allclose(y, k * x0[0])


def test_system_with_params():
    class SystemWithParams(collimator.LeafSystem):
        @parameters(static=["S"], dynamic=["D"])
        def __init__(self, S, D, *args, **kwargs):
            super().__init__(*args, **kwargs)

    system = SystemWithParams(S=1, D=2)

    assert system.parameters["S"].is_static
    assert not system.parameters["D"].is_static

    p = Parameter(name="p", value=1)
    system = SystemWithParams(S=p, D=p)

    assert len(p.static_dependents()) == 1
    static_dependent = list(p.static_dependents())[0]
    assert system.parameters["S"] is static_dependent
    assert system.parameters["S"].is_static
    assert not system.parameters["D"].is_static

    p_plus_one = p + 1
    assert len(p_plus_one.static_dependents()) == 0
