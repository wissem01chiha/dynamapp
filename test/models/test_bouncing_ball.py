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

# Idealized hybrid dynamics of a ball dropping from a height and bouncing on a
# surface
#
# Modified from drake: examples/bouncing_ball/bouncing_ball.h

import pytest

import numpy as np
import matplotlib.pyplot as plt

import collimator

from collimator.library import (
    Integrator,
    Demultiplexer,
    Multiplexer,
    Constant,
    FeedthroughBlock,
    Gain,
)

from collimator.models import BouncingBall


def test_leaf_simulate_to_contact():
    g = 10.0
    e = 1.0
    model = BouncingBall(g=g, e=e)
    context = model.create_context()

    # Set initial conditions.
    y0 = 10.0
    t1 = np.sqrt(2 * y0 / g)
    x0 = np.array([y0, 0.0])
    context = context.with_continuous_state(x0)

    # Simulate to contact with the ground
    options = collimator.SimulatorOptions(rtol=1e-8)
    results = collimator.simulate(
        model,
        context,
        (0.0, t1),
        options=options,
    )
    context = results.context

    # Check that the ball integrated to near-contact
    assert np.allclose(context.continuous_state, np.array([0.0, -g * t1]))


def test_leaf_simulate_through_contact():
    g = 10.0
    e = 1.0
    model = BouncingBall(g=g, e=e)
    context = model.create_context()

    # Set initial conditions.
    y0 = 10.0
    t1 = np.sqrt(2 * y0 / g)
    x0 = np.array([y0, 0.0])

    context = context.with_continuous_state(x0)
    context = context.with_time(0.0)

    # Simulate just beyond contact with the ground
    rtol = 1e-10
    options = collimator.SimulatorOptions(rtol=rtol)
    results = collimator.simulate(
        model,
        context,
        (0.0, t1 * (1 + rtol)),
        options=options,
    )
    context = results.context
    assert np.allclose(context.continuous_state, np.array([0.0, g * t1]))


def test_leaf_simulate_full_period():
    g = 10.0
    e = 1.0
    model = BouncingBall(g=g, e=e)
    context = model.create_context()

    # Set initial conditions.
    y0 = 10.0
    t1 = np.sqrt(2 * y0 / g)
    x0 = np.array([y0, 0.0])

    # Integrate to 2*t1 - with a perfect elastic collision this should
    #  return the ball to its initial condition
    context = context.with_continuous_state(x0)

    options = collimator.SimulatorOptions(rtol=1e-10, atol=1e-12)
    results = collimator.simulate(
        model,
        context,
        (0.0, 2 * t1),
        options=options,
    )
    context = results.context
    assert np.allclose(context.continuous_state, x0)


def test_leaf_simulate_multiple_periods(show_plot=False):
    g = 10.0
    e = 1.0
    model = BouncingBall(g=g, e=e)
    context = model.create_context()

    # Set initial conditions.
    y0 = 10.0
    t1 = np.sqrt(2 * y0 / g)  # This is one half-period of the bouncing motion
    x0 = np.array([y0, 0.0])

    print(x0)

    context = context.with_continuous_state(x0)

    options = collimator.SimulatorOptions(
        rtol=1e-10, atol=1e-12, max_major_step_length=1.0
    )
    recorded_signals = {"y": model.output_ports[0]}
    results = collimator.simulate(
        model,
        context,
        (0.0, 4 * t1),
        options=options,
        recorded_signals=recorded_signals,
    )
    context = results.context
    print(f"context.continuous_state={context.continuous_state} x0={x0}")

    time = results.time
    y = results.outputs["y"]
    print(y.shape)

    if show_plot:
        fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
        ax1.plot(time, results.outputs["y"][:, 0], label="pos")
        ax1.plot(time, results.outputs["y"][:, 1], label="vel")
        ax1.grid(True)
        ax1.legend()
        plt.show()

    assert np.allclose(context.continuous_state, x0)


def test_leaf_simulate_with_restitution():
    g = 10.0
    e = 0.7
    model = BouncingBall(g=g, e=e)
    context = model.create_context()

    # Set initial conditions.
    y0 = 10.0
    t1 = np.sqrt(2 * y0 / g)
    x0 = np.array([y0, 0.0])

    context = context.with_continuous_state(x0)

    rtol = 1e-10
    options = collimator.SimulatorOptions(rtol=rtol)
    results = collimator.simulate(
        model,
        context,
        (0.0, t1 * (1 + rtol)),
        options=options,
    )
    context = results.context
    assert np.allclose(context.continuous_state, np.array([0.0, e * g * t1]))


def test_diagram_simulate():
    g = 10.0
    e = 1.0

    # Define initial conditions
    y0 = 10.0
    t1 = np.sqrt(2 * y0 / g)
    x0 = np.array([y0, 0.0])

    builder = collimator.DiagramBuilder()
    ball = builder.add(BouncingBall(g=g, e=e))
    dummy_gain = builder.add(Gain(1.0, name="dummy_gain"))
    builder.connect(ball.output_ports[0], dummy_gain.input_ports[0])

    diagram = builder.build()

    context = diagram.create_context()

    # Integrate to 2*t1 - with a perfect elastic collision this should
    #  return the ball to its initial condition
    ball_context = context[ball.system_id].with_continuous_state(x0)
    context = context.with_subcontext(ball.system_id, ball_context)
    context = context.with_time(0.0)

    options = collimator.SimulatorOptions(rtol=1e-10, atol=1e-12)
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 2 * t1),
        options=options,
    )
    context = results.context
    assert np.allclose(context[ball.system_id].continuous_state, x0)


@pytest.mark.slow
def test_from_primitives():
    g = 10.0
    e = 1.0
    y0 = 10.0
    x0 = np.array([y0, 0.0])

    t1 = np.sqrt(2 * y0 / g)  # Contact time

    builder = collimator.DiagramBuilder()

    # Dynamics blocks
    integrator = builder.add(
        Integrator(
            initial_state=[0.0, 0.0],
            enable_reset=True,
            enable_external_reset=True,
            name="integrator",
        )
    )
    demux = builder.add(Demultiplexer(2, name="demux"))
    mux_ode = builder.add(Multiplexer(2, name="mux_ode"))
    constant = builder.add(Constant(-g, name="constant"))

    # Connect the dynamics blocks
    builder.connect(integrator.output_ports[0], demux.input_ports[0])
    builder.connect(demux.output_ports[1], mux_ode.input_ports[0])
    builder.connect(constant.output_ports[0], mux_ode.input_ports[1])
    builder.connect(mux_ode.output_ports[0], integrator.input_ports[0])

    # Reset blocks
    comparator = builder.add(FeedthroughBlock(lambda x: x <= 0, name="comparator"))
    reset_v = builder.add(Gain(-e, name="reset_v"))
    reset_y = builder.add(FeedthroughBlock(abs, name="reset_y"))
    mux_reset = builder.add(Multiplexer(2, name="mux_reset"))

    # Connect the reset blocks
    builder.connect(demux.output_ports[0], comparator.input_ports[0])
    builder.connect(
        comparator.output_ports[0], integrator.input_ports[1]
    )  # Reset trigger
    builder.connect(demux.output_ports[0], reset_y.input_ports[0])
    builder.connect(reset_y.output_ports[0], mux_reset.input_ports[0])
    builder.connect(demux.output_ports[1], reset_v.input_ports[0])
    builder.connect(reset_v.output_ports[0], mux_reset.input_ports[1])
    builder.connect(mux_reset.output_ports[0], integrator.input_ports[2])  # Reset value

    diagram = builder.build()
    context = diagram.create_context()

    int_context = context[integrator.system_id].with_continuous_state(x0)
    context = context.with_subcontext(integrator.system_id, int_context)

    options = collimator.SimulatorOptions(rtol=1e-10, atol=1e-8)
    results = collimator.simulate(
        diagram,
        context,
        (0.0, 4 * t1),
        options=options,
    )
    context = results.context

    print(f"Final state: {context[integrator.system_id].continuous_state}")
    assert np.allclose(context[integrator.system_id].continuous_state, x0, atol=1e-6)


if __name__ == "__main__":
    # test_from_primitives()
    test_leaf_simulate_multiple_periods(show_plot=True)
