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
import numpy as np

import collimator
from collimator.framework import DiagramBuilder
from collimator.library.generic import FeedthroughBlock
from collimator.library.primitives import (
    Constant,
    Demultiplexer,
    Multiplexer,
    Gain,
    Integrator,
)
from collimator.testing import make_benchmark

pytestmark = pytest.mark.minimal


def _make_benchmark_pendulum(t0=0.0, tf=100.0, rtol=1e-6, atol=1e-8, run_once=True):
    from collimator.models import Pendulum

    system = Pendulum()
    recorded_signals = {"y": system.output_ports[0]}
    return make_benchmark(
        system,
        t0,
        tf,
        rtol=rtol,
        atol=atol,
        run_once=run_once,
        recorded_signals=recorded_signals,
    )


def _make_benchmark_pendulum_diagram(
    t0=0.0, tf=100.0, rtol=1e-6, atol=1e-8, run_once=True
):
    from collimator.models import PendulumDiagram

    system = PendulumDiagram()
    recorded_signals = {"y": system.output_ports[0]}

    return make_benchmark(
        system,
        t0,
        tf,
        rtol=rtol,
        atol=atol,
        run_once=run_once,
        recorded_signals=recorded_signals,
    )


def _make_benchmark_lotka_volterra(
    t0=0.0,
    tf=100.0,
    atol=1e-14,
    rtol=1e-12,
    run_once=True,
):
    from collimator.models import LotkaVolterra

    system = LotkaVolterra()
    recorded_signals = {"y": system.output_ports[0]}

    return make_benchmark(
        system,
        t0,
        tf,
        rtol=rtol,
        atol=atol,
        run_once=run_once,
        recorded_signals=recorded_signals,
    )


def _make_benchmark_fitzhugh_nagumo(
    t0=0.0, tf=100.0, atol=1e-14, rtol=1e-12, run_once=True
):
    from collimator.models import FitzHughNagumo

    system = FitzHughNagumo()
    recorded_signals = {"y": system.output_ports[0]}
    return make_benchmark(
        system,
        t0,
        tf,
        rtol=rtol,
        atol=atol,
        run_once=run_once,
        recorded_signals=recorded_signals,
    )


def _make_benchmark_bouncing_ball(
    num_periods=4, y0=10.0, atol=1e-8, rtol=1e-6, run_once=True
):
    g = 10.0
    e = 1.0
    y0 = 10.0

    t0 = 0.0
    t1 = np.sqrt(2 * y0 / g)  # Contact time
    tf = 2 * num_periods * t1

    builder = DiagramBuilder()

    # Dynamics blocks
    integrator = builder.add(
        Integrator(
            initial_state=[y0, 0.0],
            enable_reset=True,
            enable_external_reset=True,
            name="integrator",
        )
    )
    demux = builder.add(Demultiplexer(2, name="demux"))
    mux_ode = builder.add(Multiplexer(2, name="mux_ode"))
    constant = builder.add(Constant(-g, name="constant"))

    # Connect the dynamics blocks
    # pylint: disable=no-member
    builder.connect(integrator.output_ports[0], demux.input_ports[0])
    builder.connect(demux.output_ports[1], mux_ode.input_ports[0])
    builder.connect(constant.output_ports[0], mux_ode.input_ports[1])
    builder.connect(mux_ode.output_ports[0], integrator.input_ports[0])

    # Reset blocks
    comparator = builder.add(FeedthroughBlock(lambda x: x <= 0, name="comparator"))
    reset_v = builder.add(Gain(-e, name="reset_v"))
    reset_y = builder.add(FeedthroughBlock(lambda x: jnp.abs(x), name="reset_y"))
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
    # pylint: enable=no-member

    diagram = builder.build()
    return make_benchmark(
        diagram,
        t0,
        tf,
        rtol=rtol,
        atol=atol,
        use_simulator=True,
        run_once=run_once,
    )


def test_benchmark_pendulum_leaf():
    collimator.set_backend("jax")
    benchmark_pendulum = _make_benchmark_pendulum()
    benchmark_pendulum()


def test_benchmark_pendulum_primitives():
    collimator.set_backend("jax")
    benchmark_pendulum_diagram = _make_benchmark_pendulum_diagram()
    benchmark_pendulum_diagram()


def test_benchmark_lotka_volterra():
    collimator.set_backend("jax")
    benchmark_lotka_volterra = _make_benchmark_lotka_volterra()
    benchmark_lotka_volterra()


def test_benchmark_fitzhugh_nagumo():
    collimator.set_backend("jax")
    benchmark_fitzhugh_nagumo = _make_benchmark_fitzhugh_nagumo()
    benchmark_fitzhugh_nagumo()
