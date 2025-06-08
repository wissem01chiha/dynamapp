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
from enum import IntEnum

import jax
import jax.numpy as jnp
from jax import lax

import collimator

from collimator.library import Integrator
from collimator.logging import logger

from collimator import logging

pytestmark = pytest.mark.slow


class ModeSwitchingIntegrator(collimator.LeafSystem):
    # Simple piecewise constant dynamics with a reset map
    #
    #  xdot = -a,  mode 0
    #  xdot = a,   mode 1
    #
    # Transition from mode 0 to mode 1 when x <= 0.0
    #
    # The analytic solution starting from x0 is:
    #  x(t) = x0 - a * t,  t <= x0/a
    #  x(t) = a * t - x0,  t > x0/a

    class Stage(IntEnum):
        MODE_0 = 0
        MODE_1 = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_dynamic_parameter("a", 1.0)
        self.declare_default_mode(self.Stage.MODE_0)
        self.declare_continuous_state(shape=(), ode=self._ode)
        self.declare_continuous_state_output()

        self.declare_zero_crossing(
            guard=self._guard,
            name="time_reset",
            start_mode=self.Stage.MODE_0,
            end_mode=self.Stage.MODE_1,
        )

    def _ode(self, time, state, **params):
        mode = state.mode
        a = params["a"]

        mode_dynamics = (lambda: -a, lambda: a)  # Mode 0  # Mode 1
        return lax.switch(mode, mode_dynamics)

    def _guard(self, time, state, **params):
        return state.continuous_state


def test_mode_switching_forward():
    collimator.set_backend("jax")

    a = 1.0
    x0 = 1.0

    model = ModeSwitchingIntegrator()
    context = model.create_context()

    context = context.with_continuous_state(x0)
    context = context.with_parameter("a", a)

    # Have to speciy max_major_steps here because we're calling `advance_to`
    # directly rather than using the `simulate` interface, which would estimate
    # it automatically.
    options = collimator.SimulatorOptions(
        max_major_steps=100,
        atol=1e-8,
        rtol=1e-6,
    )
    sim = collimator.Simulator(model, options=options)

    advance_to = jax.jit(sim.advance_to)

    # Check value before transition
    tf = 0.75
    sim_state = advance_to(tf, context)
    final_context = sim_state.context
    logger.debug(f"xf: {final_context.continuous_state}, expect {x0 - a * tf}")
    assert jnp.allclose(final_context.continuous_state, x0 - a * tf)

    # Check value after transition
    tf = 1.25
    sim_state = advance_to(tf, context)
    final_context = sim_state.context
    logger.debug(f"xf: {final_context.continuous_state}, expect {a * tf - x0}")
    assert jnp.allclose(final_context.continuous_state, a * tf - x0)


def test_mode_switching_adjoint():
    # Given the exact solution, the gradient with respect to the final state
    # should be:
    #
    # dxf/dx0 = 1.0 if tf <= x0/a, else -1.0
    # dxf/dtf = -a if tf <= x0/a, else a
    # dxf/da = -tf if tf <= x0/a, else tf
    collimator.set_backend("jax")

    a = 1.0
    x0 = 1.0

    model = ModeSwitchingIntegrator()
    context = model.create_context()

    # Have to speciy max_major_steps here because we're calling `advance_to`
    # directly rather than using the `simulate` interface, which would estimate
    # it automatically.
    options = collimator.SimulatorOptions(
        enable_autodiff=True,
        max_major_steps=100,
        atol=1e-8,
        rtol=1e-6,
    )
    sim = collimator.Simulator(model, options=options)

    def forward(context, x0, tf, a):
        context = context.with_continuous_state(x0)
        context = context.with_parameter("a", a)

        sim_state = sim.advance_to(tf, context)
        final_context = sim_state.context
        return final_context.continuous_state

    if sim.enable_tracing:
        forward = jax.jit(forward)
        grad_forward = jax.jit(jax.grad(forward, argnums=(1, 2, 3)))

    # Check value before transition
    tf = 0.8
    xf = forward(context, x0, tf, a)
    xe = x0 - a * tf
    logger.debug(f"{xf=}, {xe=}")
    assert jnp.allclose(xf, xe)

    # Check value after transition
    tf = 1.3
    xf = forward(context, x0, tf, a)
    xe = a * tf - x0
    logger.debug(f"{xf=}, {xe=}")
    assert jnp.allclose(xf, xe)

    # Check gradients before transition
    tf = 0.9
    dx0, dtf, da = grad_forward(context, x0, tf, a)
    logger.debug(f"{dx0=}, {dtf=}, {da=}")
    assert jnp.allclose(dx0, 1.0)
    assert jnp.allclose(dtf, -a)
    assert jnp.allclose(da, -tf)

    # Check gradients after transition
    tf = 1.1
    dx0, dtf, da = grad_forward(context, x0, tf, a)
    logger.debug(f"{dx0=}, {dtf=}, {da=}")
    assert jnp.allclose(dx0, -1.0)
    assert jnp.allclose(dtf, a)
    assert jnp.allclose(da, tf)


def test_diagram_adjoint():
    # Repeat the above test, but where the model is nested in a diagram
    collimator.set_backend("jax")

    a = 1.0
    x0 = 1.0

    builder = collimator.DiagramBuilder()
    plant = builder.add(ModeSwitchingIntegrator(name="plant"))
    builder.export_output(plant.output_ports[0])
    submodel = builder.build(name="submodel")

    builder = collimator.DiagramBuilder()
    submodel = builder.add(submodel)
    integrator = builder.add(Integrator(0.0, name="integrator"))
    builder.connect(submodel.output_ports[0], integrator.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    # Have to speciy max_major_steps here because we're calling `advance_to`
    # directly rather than using the `simulate` interface, which would estimate
    # it automatically.
    options = collimator.SimulatorOptions(
        enable_autodiff=True,
        max_major_steps=100,
        atol=1e-8,
        rtol=1e-6,
    )
    sim = collimator.Simulator(diagram, options=options)

    def forward(context, x0, tf, a):
        plant_context = context[plant.system_id].with_continuous_state(x0)
        plant_context = plant_context.with_parameter("a", a)
        context = context.with_subcontext(plant.system_id, plant_context)

        sim_state = sim.advance_to(tf, context)
        final_context = sim_state.context
        return final_context[plant.system_id].continuous_state

    if sim.enable_tracing:
        forward = jax.jit(forward)
        grad_forward = jax.jit(jax.grad(forward, argnums=(1, 2, 3)))

    # Check value before transition
    tf = 0.8
    xf = forward(context, x0, tf, a)
    xe = x0 - a * tf
    logger.debug(f"{xf=}, {xe=}")
    assert jnp.allclose(xf, xe)

    # Check value after transition
    tf = 1.3
    xf = forward(context, x0, tf, a)
    xe = a * tf - x0
    logger.debug(f"{xf=}, {xe=}")
    assert jnp.allclose(xf, xe)

    # Check gradients before transition
    tf = 0.9
    dx0, dtf, da = grad_forward(context, x0, tf, a)
    logger.debug(f"{dx0=}, {dtf=}, {da=}")
    assert jnp.allclose(dx0, 1.0)
    assert jnp.allclose(dtf, -a)
    assert jnp.allclose(da, -tf)

    # Check gradients after transition
    tf = 1.1
    dx0, dtf, da = grad_forward(context, x0, tf, a)
    logger.debug(f"{dx0=}, {dtf=}, {da=}")
    assert jnp.allclose(dx0, -1.0)
    assert jnp.allclose(dtf, a)
    assert jnp.allclose(da, tf)


if __name__ == "__main__":
    logging.set_log_level(logging.DEBUG)
    logging.set_file_handler("test.log")

    test_diagram_adjoint()
