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

from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import collimator  # noqa: E402

from collimator.framework import LeafSystem  # noqa: E402

from .transformations import (  # noqa: E402
    euler_to_rotation_matrix,
    get_Wn,
    get_Wn_inv,
    get_dot_Wn_inv,
)


@jax.jit
def quad_ode_rhs(
    state,
    control,
    time,
    Ixx=1.0,
    Iyy=1.0,
    Izz=2.0,
    k=1.0,
    b=0.5,
    l=1.0 / 3,  # noqa: E741
    m=2.0,
    g=9.81,
):
    # Parse state and control
    xi = state[:3]
    eta = state[3:6]
    dot_xi = state[6:9]
    dot_eta = state[9:]

    x, y, z = xi
    phi, theta, psi = eta
    dot_x, dot_y, dot_z = dot_xi
    dot_phi, dot_theta, dot_psi = dot_eta

    u1, u2, u3, u4 = control

    # Get transformation matrices
    R = euler_to_rotation_matrix(phi, theta, psi)
    Wn, Wn_inv = get_Wn(phi, theta, psi), get_Wn_inv(phi, theta, psi)
    dot_Wn_inv = get_dot_Wn_inv(phi, theta, psi, dot_phi, dot_theta)

    # EOM for position
    G = jnp.array([0.0, 0.0, -m * g])
    T = jnp.sum(k * control)  # noqa: E741
    TB = jnp.array([0.0, 0.0, T])

    ddot_xi = (1.0 / m) * (G + R.dot(TB))

    # EOM for orientation
    tauB = jnp.array([l * k * (-u2 + u4), l * k * (-u1 + u3), b * (-u1 + u2 - u3 + u4)])

    nu = Wn @ dot_eta
    I = jnp.diag(jnp.array([Ixx, Iyy, Izz]))  # noqa: E741
    I_inv = jnp.diag(jnp.array([1.0 / Ixx, 1.0 / Iyy, 1 / Izz]))

    dot_nu = I_inv @ (tauB - jnp.cross(nu, jnp.matmul(I, nu)))

    ddot_eta = dot_Wn_inv @ nu + Wn_inv @ dot_nu

    dot_state = jnp.hstack([dot_xi, dot_eta, ddot_xi, ddot_eta])

    return dot_state


class Quadcopter(LeafSystem):
    "Custom LeafSystem for the Quadcopter implementing the ODE"

    def __init__(
        self,
        Ixx=1.0,
        Iyy=1.0,
        Izz=2.0,
        k=1.0,
        b=0.5,
        l=1.0 / 3,  # noqa: E741
        m=2.0,
        g=9.81,
        initial_state=jnp.zeros(12),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.declare_dynamic_parameter("Ixx", Ixx)
        self.declare_dynamic_parameter("Iyy", Iyy)
        self.declare_dynamic_parameter("Izz", Izz)
        self.declare_dynamic_parameter("k", k)
        self.declare_dynamic_parameter("b", b)
        self.declare_dynamic_parameter("l", l)
        self.declare_dynamic_parameter("m", m)
        self.declare_dynamic_parameter("g", g)

        self.nx = 12

        self.declare_input_port()  # port for the control input

        self.declare_continuous_state(
            shape=(self.nx,), ode=self.ode, default_value=jnp.array(initial_state)
        )

        self.declare_continuous_state_output()

    def ode(self, time, sys_state, sys_input, **sys_params):
        state = sys_state.continuous_state
        control = sys_input

        dot_state = quad_ode_rhs(state, control, time, **sys_params)

        return dot_state


def make_quadcopter(config=None, initial_state=jnp.zeros(12), name="quadcopter"):
    builder = collimator.DiagramBuilder()
    if config is None:
        quadcopter = builder.add(
            Quadcopter(initial_state=initial_state, name="quadcopter")
        )
    else:
        quadcopter = builder.add(
            Quadcopter(**config, initial_state=initial_state, name="quadcopter")
        )

    builder.export_input(quadcopter.input_ports[0])
    builder.export_output(quadcopter.output_ports[0])

    diagram = builder.build(name=name)
    return diagram
