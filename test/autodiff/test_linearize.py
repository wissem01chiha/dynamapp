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

import jax.numpy as jnp

import control

import collimator
from collimator.library import linearize
from collimator.models import PendulumDiagram


def test_pendulum_linearize_down():
    collimator.set_backend("jax")
    g = 9.81
    L = 1.0
    b = 0.5
    x0 = jnp.array([0.0, 0.0])
    plant = PendulumDiagram(
        x0=x0, g=g, L=L, b=b, input_port=True, full_state_output=False
    )
    u0 = jnp.array([0.0])
    plant.input_ports[0].fix_value(u0)
    context = plant.create_context()

    lin_sys = linearize(plant, context)

    A = lin_sys.dynamic_parameters["A"].value
    B = lin_sys.dynamic_parameters["B"].value
    C = lin_sys.dynamic_parameters["C"].value
    D = lin_sys.dynamic_parameters["D"].value

    assert jnp.allclose(A, jnp.array([[0.0, 1.0], [-g / L, -b]]))
    assert jnp.allclose(B, jnp.array([[0.0], [1.0]]))
    assert jnp.allclose(C, jnp.array([[1.0, 0.0]]))
    assert jnp.allclose(D, jnp.array([[0.0]]))

    # Test conversion to control.ss
    control.ss(A, B, C, D)


# Tests bug from https://collimator.atlassian.net/browse/WC-38
def test_repeated_linearization():
    collimator.set_backend("jax")
    g = 9.81
    L = 1.0
    b = 0.5
    x0 = jnp.array([0.0, 0.0])
    plant = PendulumDiagram(
        x0=x0, g=g, L=L, b=b, input_port=True, full_state_output=False
    )
    u0 = jnp.array([0.0])
    plant.input_ports[0].fix_value(u0)
    context = plant.create_context()

    # This formerly threw a tracer error because the value got stuck
    # in the fixed input port.
    linearize(plant, context)
    linearize(plant, context)


def test_pendulum_linearize_up():
    collimator.set_backend("jax")
    g = 9.81
    L = 1.0
    b = 0.5
    x0 = jnp.array([jnp.pi, 0.0])
    plant = PendulumDiagram(
        x0=x0, g=g, L=L, b=b, input_port=True, full_state_output=False
    )
    u0 = jnp.array([0.0])
    plant.input_ports[0].fix_value(u0)
    context = plant.create_context()

    lin_sys = linearize(plant, context)

    # Check that the linearization matches the pendulum in the "up" position
    A = lin_sys.dynamic_parameters["A"].value
    B = lin_sys.dynamic_parameters["B"].value
    C = lin_sys.dynamic_parameters["C"].value
    D = lin_sys.dynamic_parameters["D"].value
    assert jnp.allclose(A, jnp.array([[0.0, 1.0], [g / L, -b]]))
    assert jnp.allclose(B, jnp.array([[0.0], [1.0]]))
    assert jnp.allclose(C, jnp.array([[1.0, 0.0]]))
    assert jnp.allclose(D, jnp.array([[0.0]]))

    # Test conversion to control.ss
    control.ss(A, B, C, D)
