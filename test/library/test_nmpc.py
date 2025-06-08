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

import pytest  # noqa

import jax.numpy as jnp

import collimator
from collimator.models import PendulumDiagram

from collimator.library import (
    Constant,
    DirectShootingNMPC,
    DirectTranscriptionNMPC,
    HermiteSimpsonNMPC,
)


def test_nmpc_direct_transcription():
    """
    The `DirectTranscriptionNMPC` LeafSystem raises an Exception if the NLP solver
    of CyIPPOT fails. Getting analytical solutions for a test NMPC problem is
    challenging. This test creates a pendulum model with a `DirectTranscriptionNMPC`
    controller and simulates the system for a short duration. The LeafSystem is
    considered to have passed if the CyIPOPT exception is not raised.
    """
    collimator.set_backend("jax")
    plant = PendulumDiagram(input_port=True, full_state_output=True)

    x0 = jnp.array([1.0, 0.0])

    nu = 1

    N = 5
    nh = 2
    dt = 0.1

    x0 = jnp.array([1.0, 0.0])
    xf = jnp.array([0.0, 0.0])
    ur = jnp.array([0.0])
    x_ref = jnp.tile(xf, (N + 1, 1))  # target reference x -- a constant
    u_ref = jnp.tile(ur, (N, 1))  # target reference u -- a constant

    lb_x = jnp.array([-jnp.pi / 2, -5.0])
    ub_x = jnp.array([+jnp.pi / 2, +5.0])

    lb_u = jnp.array([-10.0])
    ub_u = jnp.array([+10.0])
    Q = jnp.eye(2)
    QN = Q
    R = 0.1 * jnp.eye(1)

    x_optvars_0 = jnp.tile(x0, (N + 1, 1))
    u_optvars_0 = jnp.tile(jnp.zeros(nu), (N, 1))

    builder = collimator.DiagramBuilder()

    mpc = builder.add(
        DirectTranscriptionNMPC(
            plant,
            Q,
            QN,
            R,
            N,
            nh,
            dt,
            lb_x=lb_x,
            ub_x=ub_x,
            lb_u=lb_u,
            ub_u=ub_u,
            x_optvars_0=x_optvars_0,
            u_optvars_0=u_optvars_0,
            name="controller",
        )
    )

    plant = builder.add(PendulumDiagram(x0=x0, input_port=True, full_state_output=True))
    ref_u = builder.add(Constant(u_ref, name="ref_u"))
    ref_x = builder.add(Constant(x_ref, name="ref_x"))

    builder.connect(plant.output_ports[0], mpc.input_ports[0])
    builder.connect(ref_x.output_ports[0], mpc.input_ports[1])
    builder.connect(ref_u.output_ports[0], mpc.input_ports[2])

    builder.connect(mpc.output_ports[0], plant.input_ports[0])

    diagram = builder.build()

    context = diagram.create_context()

    collimator.simulate(diagram, context, (0.0, 0.2))


def test_nmpc_direct_shooting():
    """
    The `DirectShootingNMPC` LeafSystem raises an Exception if the NLP solver
    of CyIPPOT fails. Getting analytical solutions for a test NMPC problem is
    challenging. This test creates a pendulum model with a `DirectShootingNMPC`
    controller and simulates the system for a short duration. The LeafSystem is
    considered to have passed if the CyIPOPT exception is not raised.
    """
    collimator.set_backend("jax")
    plant = PendulumDiagram(input_port=True, full_state_output=True)

    x0 = jnp.array([1.0, 0.0])

    nu = 1

    N = 5
    nh = 2
    dt = 0.1

    x0 = jnp.array([1.0, 0.0])
    xf = jnp.array([0.0, 0.0])
    ur = jnp.array([0.0])
    x_ref = jnp.tile(xf, (N + 1, 1))  # target reference x -- a constant
    u_ref = jnp.tile(ur, (N, 1))  # target reference u -- a constant

    lb_u = jnp.array([-10.0])
    ub_u = jnp.array([+10.0])
    Q = jnp.eye(2)
    QN = Q
    R = 0.1 * jnp.eye(1)

    u_optvars_0 = jnp.tile(jnp.zeros(nu), (N, 1))

    builder = collimator.DiagramBuilder()

    mpc = builder.add(
        DirectShootingNMPC(
            plant,
            Q,
            QN,
            R,
            N,
            nh,
            dt,
            lb_u=lb_u,
            ub_u=ub_u,
            u_optvars_0=u_optvars_0,
            name="controller",
        )
    )

    plant = builder.add(PendulumDiagram(x0=x0, input_port=True, full_state_output=True))
    ref_u = builder.add(Constant(u_ref, name="ref_u"))
    ref_x = builder.add(Constant(x_ref, name="ref_x"))

    builder.connect(plant.output_ports[0], mpc.input_ports[0])
    builder.connect(ref_x.output_ports[0], mpc.input_ports[1])
    builder.connect(ref_u.output_ports[0], mpc.input_ports[2])

    builder.connect(mpc.output_ports[0], plant.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    collimator.simulate(diagram, context, (0.0, 0.2))


def test_nmpc_hermite_simpson():
    """
    The `HermiteSimpsonNMPC` LeafSystem raises an Exception if the NLP solver
    of CyIPPOT fails. Getting analytical solutions for a test NMPC problem is
    challenging. This test creates a pendulum model with a `DirectTranscriptionNMPC`
    controller and simulates the system for a short duration. The LeafSystem is
    considered to have passed if the CyIPOPT exception is not raised.
    """
    collimator.set_backend("jax")
    plant = PendulumDiagram(input_port=True, full_state_output=True)

    x0 = jnp.array([1.0, 0.0])

    nu = 1

    N = 5
    dt = 0.1

    x0 = jnp.array([1.0, 0.0])
    xf = jnp.array([0.0, 0.0])
    ur = jnp.array([0.0])
    x_ref = jnp.tile(xf, (N + 1, 1))  # target reference x -- a constant
    u_ref = jnp.tile(ur, (N + 1, 1))  # target reference u -- a constant

    lb_x = jnp.array([-jnp.pi / 2, -5.0])
    ub_x = jnp.array([+jnp.pi / 2, +5.0])

    lb_u = jnp.array([-10.0])
    ub_u = jnp.array([+10.0])
    Q = jnp.eye(2)
    QN = Q
    R = 0.1 * jnp.eye(1)

    x_optvars_0 = jnp.tile(x0, (N + 1, 1))
    u_optvars_0 = jnp.tile(jnp.zeros(nu), (N + 1, 1))

    builder = collimator.DiagramBuilder()

    mpc = builder.add(
        HermiteSimpsonNMPC(
            plant,
            Q,
            QN,
            R,
            N,
            dt,
            lb_x=lb_x,
            ub_x=ub_x,
            lb_u=lb_u,
            ub_u=ub_u,
            x_optvars_0=x_optvars_0,
            u_optvars_0=u_optvars_0,
            name="controller",
        )
    )

    plant = builder.add(PendulumDiagram(x0=x0, input_port=True, full_state_output=True))
    ref_u = builder.add(Constant(u_ref, name="ref_u"))
    ref_x = builder.add(Constant(x_ref, name="ref_x"))

    builder.connect(plant.output_ports[0], mpc.input_ports[0])
    builder.connect(ref_x.output_ports[0], mpc.input_ports[1])
    builder.connect(ref_u.output_ports[0], mpc.input_ports[2])

    builder.connect(mpc.output_ports[0], plant.input_ports[0])

    diagram = builder.build()

    context = diagram.create_context()

    collimator.simulate(diagram, context, (0.0, 0.2))


def test_hermite_simpson_trajectory_optimization():
    """
    The `HermiteSimpsonTrajectoryOptimization` class `solve` method raises an
    Exception if the NLP solver of CyIPPOT fails. Getting analytical solutions for a
    test NMPC problem is challenging. This test creates a pendulum model with a
    `DirectTranscriptionNMPC` controller and simulates the system for a short duration.
    The LeafSystem is considered to have passed if the CyIPOPT exception is not raised.
    """
    collimator.set_backend("jax")
    plant = PendulumDiagram(input_port=True, full_state_output=True)

    x0 = jnp.array([1.0, 0.0])

    nu = 1

    Tf = 0.2
    N = 5

    x0 = jnp.array([1.0, 0.0])
    xf = jnp.array([0.0, 0.0])
    ur = jnp.array([0.0])
    x_ref = jnp.tile(xf, (N + 1, 1))  # target reference x -- a constant
    u_ref = jnp.tile(ur, (N + 1, 1))  # target reference u -- a constant

    lb_x = jnp.array([-jnp.pi / 2, -5.0])
    ub_x = jnp.array([+jnp.pi / 2, +5.0])

    lb_u = jnp.array([-10.0])
    ub_u = jnp.array([+10.0])
    Q = jnp.eye(2)
    QN = Q
    R = 0.1 * jnp.eye(1)

    x_optvars_0 = jnp.tile(x0, (N + 1, 1))
    u_optvars_0 = jnp.tile(jnp.zeros(nu), (N + 1, 1))

    mpc = HermiteSimpsonNMPC(
        plant,
        Q,
        QN,
        R,
        N,
        Tf / N,
        lb_x=lb_x,
        ub_x=ub_x,
        lb_u=lb_u,
        ub_u=ub_u,
    )

    mpc.solve_trajectory_optimzation(0.0, x0, x_ref, u_ref, x_optvars_0, u_optvars_0)


if __name__ == "__main__":
    # test_nmpc_direct_shooting()
    # test_nmpc_direct_transcription()
    # test_nmpc_hermite_simpson()
    test_hermite_simpson_trajectory_optimization()
