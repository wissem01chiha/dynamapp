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

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node
import numpy as np

from collimator.framework import DiagramBuilder, LeafSystem, Parameter
from collimator.library import Constant, Gain, CustomPythonBlock, ReferenceSubdiagram


class CustomPytree:
    def __init__(self, a):
        self.a = jnp.asarray(a)


register_pytree_node(
    CustomPytree,
    lambda tree: ((tree.a,), None),
    lambda _, args: CustomPytree(*args),
)


def test_context_with_pytree_param():
    class CustomBlock(LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter(
                "pytree", default_value=CustomPytree([1, 2, 3]), as_array=False
            )

    builder = DiagramBuilder()
    custom_block = builder.add(CustomBlock())
    diagram = builder.build()

    context = diagram.create_context()
    custom_block_context = context[custom_block.system_id]

    np.testing.assert_array_equal(
        custom_block_context.parameters["pytree"].a, jnp.array([1, 2, 3])
    )

    @jax.jit
    def update_context(context):
        return context.with_parameter("pytree", CustomPytree([4, 5, 6]))

    custom_block_context = update_context(custom_block_context)
    np.testing.assert_array_equal(
        custom_block_context.parameters["pytree"].a, jnp.array([4, 5, 6])
    )


def test_context_update_param_in_jit():
    class CustomBlock(LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter(
                "param", default_value=jnp.array([1, 2, 3]), as_array=False
            )

    builder = DiagramBuilder()
    custom_block = builder.add(CustomBlock())
    diagram = builder.build()

    context = diagram.create_context()
    custom_block_context = context[custom_block.system_id]

    np.testing.assert_array_equal(
        custom_block_context.parameters["param"], np.array([1, 2, 3])
    )

    @jax.jit
    def update_context(context):
        return context.with_parameter("param", jnp.array([4, 5, 6]))

    custom_block_context = update_context(custom_block_context)
    np.testing.assert_array_equal(
        custom_block_context.parameters["param"], np.array([4, 5, 6])
    )


def test_context_update_param_twice():
    def _make_diagram():
        k = Parameter(name="k", value=1)
        builder = DiagramBuilder()
        builder.add(Constant(k + 1, name="Constant"))
        return builder.build(parameters={"k": k})

    diagram = _make_diagram()
    context = diagram.create_context()
    value_param = context.owning_system["Constant"].dynamic_parameters["value"]

    assert value_param.get() == 2

    context = context.with_parameter("k", 2)
    assert value_param.get() == 3

    context = context.with_parameter("k", 3)
    assert value_param.get() == 4


def test_context_update_param_in_submodel():
    def _make_submodel(instance_name, parameters):
        builder = DiagramBuilder()
        gain = builder.add(Gain(parameters["gain"]))
        builder.export_input(gain.input_ports[0])
        builder.export_output(gain.output_ports[0])
        return builder.build(name=instance_name)

    submodel_id = ReferenceSubdiagram.register(_make_submodel, [Parameter("gain", 1)])

    def _make_diagram():
        gain_param = Parameter(name="gain", value=1)
        builder = DiagramBuilder()
        submodel = ReferenceSubdiagram.create_diagram(
            submodel_id,
            instance_parameters={"gain": gain_param},
            instance_name="submodel",
        )
        builder.add(submodel)
        return builder.build(parameters={"gain": gain_param})


def test_initialize_static_data_node_order():
    builder = DiagramBuilder()
    c2 = builder.add(Constant(2, name="c2"))
    c1 = builder.add(Constant(1, name="c1"))
    p1 = builder.add(
        CustomPythonBlock(
            name="p1",
            dt=0.1,
            init_script="out_0 = 0",
            user_statements="out_0 = in_0",
            time_mode="agnostic",
            inputs=["in_0"],
            outputs=["out_0"],
        )
    )
    p2 = builder.add(
        CustomPythonBlock(
            name="p2",
            dt=0.1,
            init_script="out_0 = 0",
            user_statements="out_0 = in_0 + in_1",
            time_mode="agnostic",
            inputs=["in_0", "in_1"],
            outputs=["out_0"],
        )
    )

    builder.connect(c1.output_ports[0], p1.input_ports[0])
    builder.connect(p1.output_ports[0], p2.input_ports[0])
    builder.connect(c2.output_ports[0], p2.input_ports[1])

    diagram = builder.build()
    diagram.create_context()  # calls initialize_static_data
    sorted_events = [
        f"{cb.system.name}.{cb.name}" for cb in diagram.sorted_event_callbacks
    ]

    # Note that p2.in_1 (connected to a constant) comes before any p1 events,
    # which is a valid topological order but we can't initialize static data for
    # p2 before p1 because it depends on return_dtype of p1 inferred during
    # initialize data call of p1.
    # This test is to make sure that the initialize_static_data call above did
    # not fail due to this topological order.
    assert sorted_events == [
        "c2.out_0",
        "c1.out_0",
        "p2.in_1",
        "p1.in_0",
        "p1.out_0",
        "p1.cache_0",
        "p2.in_0",
        "p2.out_0",
        "p2.cache_0",
    ]
