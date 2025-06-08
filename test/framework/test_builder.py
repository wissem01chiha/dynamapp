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
from collimator.library import (
    Sine,
    Integrator,
    Gain,
    Offset,
)
from collimator.framework.diagram import AlgebraicLoopError
from collimator.framework.diagram_builder import (
    DisconnectedInputError,
    BuilderError,
    SystemNameNotUniqueError,
    EmptyDiagramError,
)

from collimator import logging


logging.set_file_handler("test.log")

pytestmark = pytest.mark.minimal


class TestBasic:
    def test_compile_twice(self):
        x0 = -1.0
        v0 = 0.0

        # Could add these systems directly to the builder, but this tests the
        # return value of add()
        builder = collimator.DiagramBuilder()

        Sin_0 = builder.add(Sine(name="Sin_0"))
        Integrator_0 = builder.add(Integrator(x0, name="Integrator_0"))
        Integrator_1 = builder.add(Integrator(v0, name="Integrator_1"))

        builder.connect(Sin_0.output_ports[0], Integrator_0.input_ports[0])
        builder.connect(Integrator_0.output_ports[0], Integrator_1.input_ports[0])

        builder.build()
        assert builder._already_built

        with pytest.raises(
            BuilderError,
            match=r"DiagramBuilder: build has already been called.*",
        ):
            builder.build()

    def test_repeated_names(self):
        Gain_0 = Gain(1.0, name="Gain_0")
        Gain_1 = Gain(2.0, name="Gain_0")
        builder = collimator.DiagramBuilder()

        builder.add(Gain_0)
        with pytest.raises(
            SystemNameNotUniqueError, match=r"System name Gain_0 is not unique.*"
        ):
            builder.add(Gain_1)

    def test_check_registered(self):
        # Could add these systems directly to the builder, but this tests the
        # return value of add()
        Gain_0 = Gain(1.0, name="Gain_0")
        builder = collimator.DiagramBuilder()

        builder._check_system_not_registered(Gain_0)

        with pytest.raises(BuilderError, match=r"System Gain_0 is not registered.*"):
            builder._check_system_is_registered(Gain_0)

        builder.add(Gain_0)
        builder._check_system_is_registered(Gain_0)

    def test_check_input_not_connected(self):
        builder = collimator.DiagramBuilder()

        Gain_0 = builder.add(Gain(1.0, name="Gain_0"))
        Sin_0 = builder.add(Sine(name="Sin_0"))

        builder._check_input_not_connected(Gain_0.input_ports[0].locator)
        builder.connect(Sin_0.output_ports[0], Gain_0.input_ports[0])

        with pytest.raises(
            BuilderError, match=r"Input port .*Gain.*0.* is already connected"
        ):
            builder._check_input_not_connected(Gain_0.input_ports[0].locator)

    def test_check_input_connected(self):
        builder = collimator.DiagramBuilder()

        Gain_0 = builder.add(Gain(1.0, name="Gain_0"))
        Gain_1 = builder.add(Gain(1.0, name="Gain_1"))

        builder.connect(Gain_0.output_ports[0], Gain_1.input_ports[0])
        builder._check_input_is_connected(Gain_1.input_ports[0].locator)

        with pytest.raises(DisconnectedInputError, match=r".*Gain_0.*0.*"):
            builder._check_input_is_connected(Gain_0.input_ports[0].locator)

    def test_empty(self):
        builder = collimator.DiagramBuilder()
        with pytest.raises(
            EmptyDiagramError, match=r"Cannot compile an empty diagram.*"
        ):
            builder.build()

    def test_add_to_already_built(self):
        builder = collimator.DiagramBuilder()
        builder.add(Sine(name="Sin_0"))
        builder.build()

        Gain_0 = Gain(1.0, name="Gain_0")
        with pytest.raises(
            BuilderError,
            match=r"DiagramBuilder: build has already been called.*",
        ):
            builder.add(Gain_0)

    def test_add_twice(self):
        builder = collimator.DiagramBuilder()

        Gain_0 = builder.add(Gain(1.0, name="Gain_0"))
        with pytest.raises(BuilderError, match=r"System Gain_0 is already registered"):
            builder.add(Gain_0)

    def test_connect_unregistered(self):
        builder = collimator.DiagramBuilder()

        Gain_0 = Gain(1.0, name="Gain_0")
        Sin_0 = builder.add(Sine(name="Sin_0"))

        with pytest.raises(BuilderError, match=r"System Gain_0 is not registered"):
            builder.connect(Sin_0.output_ports[0], Gain_0.input_ports[0])

    def test_connect_already_connected(self):
        builder = collimator.DiagramBuilder()

        Gain_0 = builder.add(Gain(1.0, name="Gain_0"))
        Sin_0 = builder.add(Sine(name="Sin_0"))

        builder.connect(Sin_0.output_ports[0], Gain_0.input_ports[0])
        with pytest.raises(
            BuilderError, match=r"Input port .*Gain.*0.* is already connected"
        ):
            builder.connect(Sin_0.output_ports[0], Gain_0.input_ports[0])

    def test_invalid_connection_map(self):
        # Manually add an invalid connection between unregistered systems and
        #  test that the error is caught.

        builder = collimator.DiagramBuilder()

        Gain_0 = Gain(1.0, name="Gain_0")
        Sin_0 = builder.add(Sine(name="Sin_0"))

        builder._connection_map = {
            Gain_0.input_ports[0].locator: Sin_0.output_ports[0].locator
        }

        with pytest.raises(BuilderError, match=r"System Gain_0 is not registered"):
            builder.build()

    def test_export(self):
        builder = collimator.DiagramBuilder()
        Gain_0 = builder.add(Gain(3.0, name="Gain_0"))
        builder.export_input(Gain_0.input_ports[0], name="u")
        builder.export_output(Gain_0.output_ports[0], name="y")

        assert builder._registered_systems == [Gain_0]
        diagram = builder.build()
        diagram.input_ports[0].fix_value(1.5)

        ctx = diagram.create_context()
        assert diagram.input_ports[0].name == "u"
        assert diagram.output_ports[0].name == "y"
        assert len(diagram.input_ports) == 1
        assert len(diagram.output_ports) == 1

        val = diagram.input_ports[0].eval(ctx)
        assert jnp.allclose(val, 1.5)

        val = diagram.output_ports[0].eval(ctx)
        assert jnp.allclose(val, 4.5)

        assert diagram.get_feedthrough() == [(0, 0)]


class TestAlgebraicLoops:
    def test_integrator_loop(self):
        # Should check that no algebraic loop is detected in a
        # single integrator

        x0 = 0.5
        builder = collimator.DiagramBuilder()

        Integrator_0 = builder.add(Integrator(x0, name="Integrator_0"))

        builder.connect(Integrator_0.output_ports[0], Integrator_0.input_ports[0])

        diagram = builder.build()
        assert diagram["Integrator_0"] is Integrator_0

        ctx = diagram.create_context()

        t0, t1 = 0.0, 2.0
        results = collimator.simulate(
            diagram,
            ctx,
            (t0, t1),
        )
        xf = results.context[Integrator_0.system_id].continuous_state

        # Check against analytic solution
        assert jnp.allclose(xf, x0 * jnp.exp(t1))

    def test_algebraic_loop_single_block(self):
        builder = collimator.DiagramBuilder()
        Gain_0 = builder.add(Gain(1.0, name="Gain_0"))
        builder.connect(Gain_0.output_ports[0], Gain_0.input_ports[0])

        with pytest.raises(AlgebraicLoopError, match=r"Algebraic loop detected"):
            diagram = builder.build(name="root")
            diagram.check_no_algebraic_loops()

    def test_algebraic_loop_multi_block(self):
        builder = collimator.DiagramBuilder()
        Gain_0 = builder.add(Gain(1.0, name="Gain_0"))
        Gain_1 = builder.add(Gain(1.0, name="Gain_1"))
        builder.connect(Gain_0.output_ports[0], Gain_1.input_ports[0])
        builder.connect(Gain_1.output_ports[0], Gain_0.input_ports[0])

        with pytest.raises(AlgebraicLoopError, match=r"Algebraic loop detected"):
            diagram = builder.build(name="root")
            diagram.check_no_algebraic_loops()

    def test_algebraic_loop_submodel(self):
        builder = collimator.DiagramBuilder()
        Gain_0 = builder.add(Gain(1.0, name="Gain_0"))
        Gain_1 = builder.add(Gain(1.0, name="Gain_1"))
        builder.connect(Gain_0.output_ports[0], Gain_1.input_ports[0])
        builder.export_input(Gain_0.input_ports[0], name="u")
        builder.export_output(Gain_1.output_ports[0], name="y")

        submodel = builder.build()

        builder = collimator.DiagramBuilder()
        Gain_2 = builder.add(Gain(1.0, name="Gain_2"))
        builder.add(submodel)
        builder.connect(Gain_2.output_ports[0], submodel.input_ports[0])
        builder.connect(submodel.output_ports[0], Gain_2.input_ports[0])

        with pytest.raises(AlgebraicLoopError, match=r"Algebraic loop detected"):
            diagram = builder.build(name="root")
            diagram.check_no_algebraic_loops()


class TestComposition:
    def test_simple_submodel(self):
        a = 2.0
        b = 1.0
        x0 = 4.0
        builder = collimator.DiagramBuilder()

        Gain_0 = builder.add(Gain(-a, name="Gain_0"))
        Offset_0 = builder.add(Offset(b, name="Offset_0"))

        builder.connect(Gain_0.output_ports[0], Offset_0.input_ports[0])
        builder.export_input(Gain_0.input_ports[0], name="u")
        builder.export_output(Offset_0.output_ports[0], name="y")

        dynamics = builder.build(name="dynamics")

        # Start with a new DiagramBuilder and embed the subsystem
        builder = collimator.DiagramBuilder()
        Integrator_0 = builder.add(Integrator(x0, name="Integrator_0"))
        builder.add(dynamics)
        builder.connect(Integrator_0.output_ports[0], dynamics.input_ports[0])
        builder.connect(dynamics.output_ports[0], Integrator_0.input_ports[0])

        diagram = builder.build()
        assert diagram["Integrator_0"] is Integrator_0
        assert diagram["dynamics"] is dynamics
        assert diagram["dynamics"]["Gain_0"] is Gain_0
        assert diagram["dynamics"]["Offset_0"] is Offset_0

        ctx = diagram.create_context()

        t0, t1 = 0.0, 2.0
        results = collimator.simulate(
            diagram,
            ctx,
            (t0, t1),
        )
        xf = results.context[Integrator_0.system_id].continuous_state

        # Check against analytic solution
        C = x0 - b / a
        x_true = C * jnp.exp(-a * t1) + b / a

        assert jnp.allclose(xf, x_true)

    def _make_nested_diagram(self, a=2.0, b=1.0, x0=0.0):
        builder = collimator.DiagramBuilder()

        Integrator_0 = builder.add(Integrator(x0, name="Integrator_0"))
        Gain_0 = builder.add(Gain(-a, name="Gain_0"))

        builder.connect(Integrator_0.output_ports[0], Gain_0.input_ports[0])
        builder.export_input(Integrator_0.input_ports[0], name="submodel:u")
        builder.export_output(Gain_0.output_ports[0], name="submodel:y")
        submodel = builder.build(name="submodel")

        # Start with a new DiagramBuilder and embed the subsystem
        builder = collimator.DiagramBuilder()
        Offset_0 = builder.add(Offset(b, name="Offset_0"))
        builder.add(submodel)
        builder.connect(Offset_0.output_ports[0], submodel.input_ports[0])
        builder.connect(submodel.output_ports[0], Offset_0.input_ports[0])

        diagram = builder.build()
        return diagram

    def test_dependencies(self):
        from collimator.framework import DependencyTicket

        diagram = self._make_nested_diagram()
        offset = diagram["Offset_0"]
        submodel = diagram["submodel"]
        integrator = submodel["Integrator_0"]
        gain = submodel["Gain_0"]

        # Check that the diagram-level xcdot depends on the
        # submodel-level xcdot
        diagram_tracker = diagram.dependency_graph[DependencyTicket.xcdot]
        for subsys in diagram.nodes:
            submodel_tracker = subsys.dependency_graph[DependencyTicket.xcdot]
            assert diagram_tracker in submodel_tracker.subscribers
            assert submodel_tracker in diagram_tracker.prerequisites

        # Check that the submodel "all inputs" depends on the
        # submodel input port
        sub_u_tracker = submodel.dependency_graph[DependencyTicket.u]
        sub_ui_tracker = submodel.dependency_graph[submodel.input_ports[0].ticket]
        assert sub_u_tracker in sub_ui_tracker.subscribers
        assert sub_ui_tracker in sub_u_tracker.prerequisites

        # Check that the integrator input port depends on the
        # submodel input port
        int_ui_tracker = integrator.dependency_graph[integrator.input_ports[0].ticket]
        assert int_ui_tracker in sub_ui_tracker.subscribers
        assert sub_ui_tracker in int_ui_tracker.prerequisites

        # Check that the submodel input port depends on the offset output port
        off_y_tracker = offset.dependency_graph[offset.output_ports[0].ticket]
        assert sub_ui_tracker in off_y_tracker.subscribers
        assert off_y_tracker in sub_ui_tracker.prerequisites

        # Check that the offset output port depends exclusively on the
        # offset input port and is a subscriber of the offset input port
        off_ui_tracker = offset.dependency_graph[offset.input_ports[0].ticket]
        assert off_y_tracker.prerequisites == [off_ui_tracker]
        assert off_y_tracker in off_ui_tracker.subscribers

        # Check that the offset input port depends on the submodel output port
        sub_y_tracker = submodel.dependency_graph[submodel.output_ports[0].ticket]
        assert off_ui_tracker in sub_y_tracker.subscribers
        assert sub_y_tracker in off_ui_tracker.prerequisites

        # Check that the submodel output port depends on the gain output port
        gain_y_tracker = gain.dependency_graph[gain.output_ports[0].ticket]
        assert sub_y_tracker in gain_y_tracker.subscribers
        assert gain_y_tracker in sub_y_tracker.prerequisites

        # Check that the gain output port depends exclusively on the
        # gain input port and is a subscriber of the gain input port
        gain_ui_tracker = gain.dependency_graph[gain.input_ports[0].ticket]
        assert gain_y_tracker.prerequisites == [gain_ui_tracker]
        assert gain_y_tracker in gain_ui_tracker.subscribers

        # Check that the gain input port depends on the integrator output port
        int_y_tracker = integrator.dependency_graph[integrator.output_ports[0].ticket]
        assert gain_ui_tracker in int_y_tracker.subscribers
        assert int_y_tracker in gain_ui_tracker.prerequisites

        # Check that the integrator output port depends exclusively on the
        # integrator continous state
        int_xc_tracker = integrator.dependency_graph[DependencyTicket.xc]
        assert int_y_tracker.prerequisites == [int_xc_tracker]

    def test_submodel_with_integrator(self):
        a = 2.0
        b = 1.0
        x0 = 4.0
        diagram = self._make_nested_diagram(a=a, b=b, x0=x0)

        ctx = diagram.create_context()

        # Test the initial time derivative value
        xdot = -a * x0 + b
        result = diagram.eval_time_derivatives(ctx)
        xdot_eval = result[0]
        assert jnp.allclose(xdot_eval, xdot)

        # Change the input value and check for a change
        x0 += 1.0
        integrator = diagram["submodel"]["Integrator_0"]
        int_ctx = ctx[integrator.system_id].with_continuous_state(x0)
        ctx = ctx.with_subcontext(integrator.system_id, int_ctx)

        # Test the updated time derivative value
        xdot = -a * x0 + b
        result = diagram.eval_time_derivatives(ctx)
        xdot_eval = result[0]
        assert jnp.allclose(xdot_eval, xdot)

        # Test integration forward in time
        t0, t1 = 0.0, 2.0
        results = collimator.simulate(
            diagram,
            ctx,
            (t0, t1),
        )
        xf = results.context[integrator.system_id].continuous_state

        # Check against analytic solution
        C = x0 - b / a
        x_true = C * jnp.exp(-a * t1) + b / a

        assert jnp.allclose(xf, x_true)


class TestPrimitiveSystems:
    # The validity of this is tested in models/test_double_integrator.py
    #   This just tests the construction via primitives using DiagramBuilder.
    def test_double_integrator(self):
        x0 = -1.0
        v0 = 0.0

        # Could add these systems directly to the builder, but this tests
        # the return value of add()
        sys_1 = Sine(name="Sin_0")
        sys_2 = Integrator(x0, name="Integrator_0")
        sys_3 = Integrator(v0, name="Integrator_1")

        builder = collimator.DiagramBuilder()

        Sin_0 = builder.add(sys_1)
        assert sys_1 is Sin_0

        Integrator_0 = builder.add(sys_2)  # x
        assert sys_2 is Integrator_0

        Integrator_1 = builder.add(sys_3)  # v
        assert sys_3 is Integrator_1

        for system in [sys_1, sys_2, sys_3]:
            builder._check_system_is_registered(system)

        builder.connect(Sin_0.output_ports[0], Integrator_0.input_ports[0])
        assert len(builder._connection_map) == 1
        assert Integrator_0.input_ports[0].locator in builder._connection_map
        assert (
            builder._connection_map[Integrator_0.input_ports[0].locator]
            == Sin_0.output_ports[0].locator
        )

        builder.connect(Integrator_0.output_ports[0], Integrator_1.input_ports[0])
        assert len(builder._connection_map) == 2
        assert Integrator_1.input_ports[0].locator in builder._connection_map
        assert (
            builder._connection_map[Integrator_1.input_ports[0].locator]
            == Integrator_0.output_ports[0].locator
        )

        builder._check_not_already_built()
        builder._check_contents_are_complete()

        diagram = builder.build()

        diagram.check_no_algebraic_loops()

        assert len(diagram.connection_map) == 2
        assert diagram.num_systems == 3

        assert Integrator_0.input_ports[0].locator in diagram.connection_map
        assert (
            diagram.connection_map[Integrator_0.input_ports[0].locator]
            == Sin_0.output_ports[0].locator
        )
        assert Integrator_1.input_ports[0].locator in diagram.connection_map
        assert (
            diagram.connection_map[Integrator_1.input_ports[0].locator]
            == Integrator_0.output_ports[0].locator
        )

        for sys in [Sin_0, Integrator_0, Integrator_1]:
            assert sys in diagram.nodes

        # Check convenience indexing
        assert diagram["Sin_0"] is Sin_0
        assert diagram["Integrator_0"] is Integrator_0
        assert diagram["Integrator_1"] is Integrator_1

        # Check context and dependency graph
        context = diagram.create_context()
        assert context.num_continuous_states == 2

    # The validity of this is tested in models/test_scalar_affine.py
    #   This just tests the construction via primitives using DiagramBuilder.
    def test_scalar_linear(self):
        a = 1.5
        x0 = 4.0
        builder = collimator.DiagramBuilder()

        # Test adding multiple systems at once
        sys_1 = Gain(-a, name="Gain_0")
        sys_2 = Integrator(x0, name="Integrator_0")

        # pylint: disable=unpacking-non-sequence
        Gain_0, Integrator_0 = builder.add(sys_1, sys_2)
        assert sys_1 is Gain_0
        assert sys_2 is Integrator_0

        builder.connect(Gain_0.output_ports[0], Integrator_0.input_ports[0])
        assert len(builder._connection_map) == 1
        assert Integrator_0.input_ports[0].locator in builder._connection_map
        assert (
            builder._connection_map[Integrator_0.input_ports[0].locator]
            == Gain_0.output_ports[0].locator
        )

        builder.connect(Integrator_0.output_ports[0], Gain_0.input_ports[0])
        assert len(builder._connection_map) == 2
        assert Gain_0.input_ports[0].locator in builder._connection_map
        assert (
            builder._connection_map[Gain_0.input_ports[0].locator]
            == Integrator_0.output_ports[0].locator
        )

        builder._check_not_already_built()
        builder._check_contents_are_complete()
        diagram = builder.build()

        diagram.check_no_algebraic_loops()

        # Test constructed system
        assert len(diagram.connection_map) == 2
        assert diagram.num_systems == 2

        assert Integrator_0.input_ports[0].locator in diagram.connection_map
        assert (
            diagram.connection_map[Integrator_0.input_ports[0].locator]
            == Gain_0.output_ports[0].locator
        )
        assert Gain_0.input_ports[0].locator in diagram.connection_map
        assert (
            diagram.connection_map[Gain_0.input_ports[0].locator]
            == Integrator_0.output_ports[0].locator
        )

        for sys in [Gain_0, Integrator_0]:
            assert sys in diagram.nodes

        # Check convenience indexing
        assert diagram["Gain_0"] is Gain_0
        assert diagram["Integrator_0"] is Integrator_0

        # Test context and dependency graph
        context = diagram.create_context()
        assert context.num_continuous_states == 1


if __name__ == "__main__":
    # tester = TestBasic()
    # tester.test_export_output()

    tester = TestComposition()
    # tester.test_composition_with_integrators()
    tester.test_submodel_with_integrator()
