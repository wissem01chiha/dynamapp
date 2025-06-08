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

"""Tests for LQR algorithms"""

import pytest  # noqa

import control as pycontrol
import jax.numpy as jnp

import collimator

from collimator.models import PlanarQuadrotor

from collimator.library import (
    linearize,
    Constant,
    Adder,
    ZeroOrderHold,
    LinearQuadraticRegulator,
    DiscreteTimeLinearQuadraticRegulator,
    FiniteHorizonLinearQuadraticRegulator,
)

# Define a global variable for absolute tolerance
ATOL = 1e-4


def test_lqr_continuous_time():
    config = {"m": 1.0, "I_B": 1.0, "r": 0.5, "g": 9.81}

    nx = 6
    nu = 2

    weight = config["m"] * config["g"]
    u_eq = (weight / nu) * jnp.ones(nu)
    x_eq = jnp.zeros(nx)

    Q = jnp.diag(jnp.array([10, 10, 10, 1, 1, config["r"] / 2.0 / jnp.pi]))
    R = jnp.array([[0.1, 0.05], [0.05, 0.1]])

    # Linearize
    planar_quadcopter = PlanarQuadrotor(**config, name="planar_quadrotor")
    planar_quadcopter.input_ports[0].fix_value(u_eq)
    base_context = planar_quadcopter.create_context()
    eq_context = base_context.with_continuous_state(x_eq)
    linear_planar_quadcopter = linearize(planar_quadcopter, eq_context)

    A, B = linear_planar_quadcopter.A, linear_planar_quadcopter.B

    builder = collimator.DiagramBuilder()

    plant = PlanarQuadrotor(**config, name="planar_quadrotor")
    controller_bar = LinearQuadraticRegulator(A, B, Q, R, name="controller_bar")
    builder.add(plant)
    builder.add(controller_bar)

    controller = builder.add(Adder(2, operators="++", name="controller"))
    u_eq_block = builder.add(Constant(u_eq, name="u_eq_block"))

    builder.connect(controller_bar.output_ports[0], controller.input_ports[0])
    builder.connect(u_eq_block.output_ports[0], controller.input_ports[1])

    builder.connect(plant.output_ports[0], controller_bar.input_ports[0])
    builder.connect(controller.output_ports[0], plant.input_ports[0])

    diagram = builder.build()

    x0 = jnp.array([1.5, 1.5, jnp.pi / 4.0] + [0.5] * 3)
    context = diagram.create_context()
    context = context.with_continuous_state([x0])

    recorded_signals = {
        "x": plant.output_ports[0],
        "u_opt": controller.output_ports[0],
    }

    Tsolve = 10.0

    results = collimator.simulate(
        diagram,
        context,
        (0.0, Tsolve),
        recorded_signals=recorded_signals,
    )
    assert jnp.allclose(results.outputs["x"][-1], x_eq, atol=ATOL)


def test_lqr_discrete_time():
    config = {"m": 1.0, "I_B": 1.0, "r": 0.5, "g": 9.81}

    nx = 6
    nu = 2

    weight = config["m"] * config["g"]
    u_eq = (weight / nu) * jnp.ones(nu)
    x_eq = jnp.zeros(nx)

    Q = jnp.diag(jnp.array([10, 10, 10, 1, 1, config["r"] / 2.0 / jnp.pi]))
    R = jnp.array([[0.1, 0.05], [0.05, 0.1]])

    # Linearize
    planar_quadcopter = PlanarQuadrotor(**config, name="planar_quadrotor")
    planar_quadcopter.input_ports[0].fix_value(u_eq)
    base_context = planar_quadcopter.create_context()
    eq_context = base_context.with_continuous_state(x_eq)
    linear_planar_quadcopter = linearize(planar_quadcopter, eq_context)

    A, B, C, D = (
        linear_planar_quadcopter.A,
        linear_planar_quadcopter.B,
        linear_planar_quadcopter.C,
        linear_planar_quadcopter.D,
    )

    # Discrete couterpart
    dt = 0.1
    sysc = pycontrol.ss(A, B, C, D)
    sys = sysc.sample(dt)
    Ad, Bd = sys.A, sys.B

    builder = collimator.DiagramBuilder()

    plant_continuous = PlanarQuadrotor(**config, name="planar_quadrotor")
    lqr = DiscreteTimeLinearQuadraticRegulator(Ad, Bd, Q, R, dt, name="controller_bar")

    builder.add(plant_continuous)
    builder.add(lqr)

    zoh = builder.add(ZeroOrderHold(dt, name="zoh"))

    controller = builder.add(Adder(2, operators="++", name="controller"))
    u_eq_block = builder.add(Constant(u_eq, name="u_eq_block"))

    builder.connect(plant_continuous.output_ports[0], zoh.input_ports[0])

    builder.connect(zoh.output_ports[0], lqr.input_ports[0])

    builder.connect(lqr.output_ports[0], controller.input_ports[0])
    builder.connect(u_eq_block.output_ports[0], controller.input_ports[1])

    builder.connect(controller.output_ports[0], plant_continuous.input_ports[0])

    diagram = builder.build()

    x0 = jnp.array([1.5, 1.5, jnp.pi / 4.0] + [0.5] * 3)
    context = diagram.create_context()
    context = context.with_continuous_state([x0])

    recorded_signals = {
        "x": plant_continuous.output_ports[0],
        "u_opt": controller.output_ports[0],
    }

    Tsolve = 10.0

    results = collimator.simulate(
        diagram,
        context,
        (0.0, Tsolve),
        recorded_signals=recorded_signals,
    )
    assert jnp.allclose(results.outputs["x"][-1], x_eq, atol=ATOL)


def test_lqr_finite_horizon_continuous_time():
    config = {"m": 1.0, "I_B": 1.0, "r": 0.5, "g": 9.81}

    nx = 6
    nu = 2

    t0 = 0.0
    tf = 10.0

    weight = config["m"] * config["g"]
    u_0 = (weight / nu) * jnp.ones(nu)
    x_0 = jnp.zeros(nx)

    u_d = u_0
    x_d = x_0

    Q = jnp.diag(jnp.array([10, 10, 10, 1, 1, config["r"] / 2.0 / jnp.pi]))
    R = jnp.array([[0.1, 0.05], [0.05, 0.1]])

    # With make method
    builder = collimator.DiagramBuilder()

    plant = PlanarQuadrotor(**config, name="planar_quadrotor")
    controller = FiniteHorizonLinearQuadraticRegulator(
        t0,
        tf,
        PlanarQuadrotor(**config, name="planar_quadrotor"),
        Q,  # Qf
        lambda t: Q,
        lambda t: R,
        lambda t: jnp.zeros((nx, nu)),
        lambda t: x_0,
        lambda t: u_0,
        lambda t: x_d,
        lambda t: u_d,
        name="controller",
    )

    builder.add(plant)
    builder.add(controller)

    builder.connect(plant.output_ports[0], controller.input_ports[0])
    builder.connect(controller.output_ports[0], plant.input_ports[0])

    diagram = builder.build()

    x0 = jnp.array([1.5, 1.5, jnp.pi / 4.0] + [0.5] * 3)
    context = diagram.create_context()
    context = context.with_continuous_state([x0])

    recorded_signals = {
        "x": plant.output_ports[0],
        "u_opt": controller.output_ports[0],
    }

    results = collimator.simulate(
        diagram,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
    )
    assert jnp.allclose(results.outputs["x"][-1], x_0, atol=ATOL)


if __name__ == "__main__":
    test_lqr_continuous_time()
