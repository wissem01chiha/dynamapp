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

# Test various aspects of Diagram creation on using the Pendulum test model
#
#
import pytest

import jax
import jax.numpy as jnp

import collimator
from collimator.models import Pendulum
from collimator.library import Constant


@pytest.fixture
def plant():
    return Pendulum(
        name="pendulum", x0=[0.0, 0.0], input_port=True, full_state_output=True
    )


@pytest.fixture
def source():
    return Constant(1.5)


@pytest.fixture
def x0():
    return jnp.array([1.0, 0.0])


@pytest.fixture
def xdot0(x0, plant, source):
    L = plant.dynamic_parameters["L"].get()
    g = plant.dynamic_parameters["g"].get()
    b = plant.dynamic_parameters["b"].get()
    tau = source.dynamic_parameters["value"].get()
    θ, dθ = x0
    ddθ = tau - (g / L) * jnp.sin(θ) - b * dθ
    return jnp.array([dθ, ddθ])


@pytest.fixture
def diagram(plant, source):
    builder = collimator.DiagramBuilder()
    builder.add(plant, source)
    builder.connect(source.output_ports[0], plant.input_ports[0])

    return builder.build()


@pytest.fixture
def root_context(diagram):
    return diagram.create_context()


class TestPendulumDiagram:
    def test_pendulum_create(self, plant):
        assert plant.name == "pendulum"

    def test_pendulum_input_port_config(self, plant):
        # Check input ports
        assert plant.num_input_ports == 1
        inport = plant.input_ports[0]
        assert inport.name == "u"
        assert inport.system is plant

    def test_pendulum_output_port_config(self, plant):
        assert plant.num_output_ports == 1
        outport = plant.output_ports[0]
        assert outport.name == "x"
        assert outport.system is plant

    def test_source_input_port_config(self, source):
        assert source.num_input_ports == 0

    def test_source_output_port_config(self, source):
        assert source.num_output_ports == 1
        outport = source.output_ports[0]
        assert outport.name == "out_0"
        assert outport.system is source

    def test_diagram_create(self, plant, source, diagram):
        assert diagram.num_systems == 2

        # Check that the systems are in the diagram for the
        # purposes of the test.
        assert diagram[plant.name] is plant
        assert diagram[source.name] is source
        assert len(diagram.nodes) == 2
        assert plant in diagram.nodes
        assert source in diagram.nodes

        # Check for the input port in the connection map
        src_locator = source.output_ports[0].locator
        dst_locator = plant.input_ports[0].locator
        assert dst_locator in diagram.connection_map
        assert diagram.connection_map[dst_locator] == src_locator

    def test_diagram_context(self, plant, source, diagram):
        ctx = diagram.create_context()
        assert isinstance(ctx, collimator.framework.DiagramContext)

        plant_ctx = ctx[plant.system_id]
        assert isinstance(plant_ctx, collimator.framework.LeafContext)
        assert plant_ctx.system_id == plant.system_id

        source_ctx = ctx[source.system_id]
        assert isinstance(source_ctx, collimator.framework.LeafContext)
        assert source_ctx.system_id == source.system_id

    def test_diagram_context_continuous_state(self, x0, diagram, plant, source):
        ctx = diagram.create_context()

        assert ctx.num_continuous_states == 2

        assert jnp.allclose(ctx[plant.system_id].continuous_state, jnp.zeros(2))

        # Set the plant state
        plant_ctx = ctx[plant.system_id].with_continuous_state(x0)
        ctx = ctx.with_subcontext(plant.system_id, plant_ctx)

        # Check this updated both the leaf and diagram contexts
        assert jnp.allclose(plant_ctx.continuous_state, x0)
        assert jnp.allclose(ctx[plant.system_id].continuous_state, x0)

        assert ctx[source.system_id].continuous_state is None

    def _check_val_and_ports(self, val, plant, source):
        assert val == source.dynamic_parameters["value"]
        # Check the ports didnt get changed somehow
        assert source.num_output_ports == 1
        assert plant.num_output_ports == 1
        assert plant.num_input_ports == 1

    def test_direct_output_port_evaluation(self, plant, source, root_context):
        val = jax.jit(source.output_ports[0].eval)(root_context)
        self._check_val_and_ports(val, plant, source)

    def test_subsystem_output_port_evaluation(
        self, plant, source, root_context, diagram
    ):
        eval_port = jax.jit(diagram.eval_subsystem_output_port, static_argnums=(1,))
        val = eval_port(root_context, source.output_ports[0].locator)
        self._check_val_and_ports(val, plant, source)

    def test_subsystem_input_port_evaluation(
        self, plant, source, root_context, diagram
    ):
        eval_port = jax.jit(diagram.eval_subsystem_input_port, static_argnums=(1,))
        val = eval_port(root_context, plant.input_ports[0].locator)
        self._check_val_and_ports(val, plant, source)

    def test_direct_input_port_evaluation(self, plant, source, root_context):
        val = jax.jit(plant.input_ports[0].eval)(root_context)
        self._check_val_and_ports(val, plant, source)

    def test_input_port_evaluation_from_leaf(self, plant, source, root_context):
        val = jax.jit(plant.eval_input)(root_context)
        self._check_val_and_ports(val, plant, source)

    def test_subsystem_time_derivatives(self, x0, xdot0, plant, root_context):
        plant_ctx = root_context[plant.system_id].with_continuous_state(x0)
        root_context = root_context.with_subcontext(plant.system_id, plant_ctx)
        xdot0_calc = plant.eval_time_derivatives(root_context)
        assert jnp.allclose(xdot0, xdot0_calc)

    def test_root_time_derivatives(self, x0, xdot0, root_context, diagram, plant):
        plant_ctx = root_context[plant.system_id].with_continuous_state(x0)
        root_context = root_context.with_subcontext(plant.system_id, plant_ctx)
        assert jnp.allclose(root_context[plant.system_id].continuous_state, x0)
        time_derivs = jax.jit(diagram.eval_time_derivatives)(root_context)
        (xdot0_calc,) = time_derivs
        assert jnp.allclose(xdot0, xdot0_calc)

        # Check the original context is unchanged
        assert jnp.allclose(root_context[plant.system_id].continuous_state, x0)
