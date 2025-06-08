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

from .transformations import get_Wn_inv


@jax.jit
def differentially_flat_state_and_control(
    sigma,  # differnetially flat input
    sigma_d1,  # first derivative
    sigma_d2,  # second derivative
    sigma_d3,  # third derivative
    sigma_d4,  # fourth derivative
    Ixx=1.0,
    Iyy=1.0,
    Izz=2.0,
    k=1.0,
    b=0.5,
    l=1.0 / 3,  # noqa: E741
    m=2.0,
    g=9.81,
):
    # Parse differential inputs
    sigma_1, sigma_2, sigma_3, sigma_4 = sigma
    sigma_1_d1, sigma_2_d1, sigma_3_d1, sigma_4_d1 = sigma_d1
    sigma_1_d2, sigma_2_d2, sigma_3_d2, sigma_4_d2 = sigma_d2
    sigma_1_d3, sigma_2_d3, sigma_3_d3, sigma_4_d3 = sigma_d3
    sigma_1_d4, sigma_2_d4, sigma_3_d4, sigma_4_d4 = sigma_d4

    # Find the body frame and the corresponding rotation matrix
    alpha = jnp.array([sigma_1_d2, sigma_2_d2, sigma_3_d2 + g])
    beta = alpha

    xc = jnp.array([jnp.cos(sigma_4), jnp.sin(sigma_4), 0])
    yc = jnp.array([-jnp.sin(sigma_4), jnp.cos(sigma_4), 0])

    xb = jnp.cross(yc, alpha) / jnp.linalg.norm(jnp.cross(yc, alpha))
    yb = jnp.cross(beta, xb) / jnp.linalg.norm(jnp.cross(beta, xb))
    zb = jnp.cross(xb, yb)

    R = jnp.vstack([xb, yb, zb]).T
    phi = jnp.arctan2(R[2, 1], R[2, 2])
    theta = jnp.arctan2(-R[2, 0], jnp.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    psi = jnp.arctan2(R[1, 0], R[0, 0])  # sigma_4

    eta = jnp.array([phi, theta, psi])

    # Compute angular velocities
    c = zb.dot(alpha)
    dot_a = jnp.array([sigma_1_d3, sigma_2_d3, sigma_3_d3])

    q = xb.dot(dot_a) / c
    p = -yb.dot(dot_a) / c
    r = (sigma_4_d1 * xc.dot(xb) + q * yc.dot(zb)) / jnp.linalg.norm(jnp.cross(yc, zb))

    nu = jnp.array([p, q, r])

    Wn_inv = get_Wn_inv(phi, theta, psi)

    dot_eta = Wn_inv @ nu

    xi = jnp.array([sigma_1, sigma_2, sigma_3])
    dot_xi = jnp.array([sigma_1_d1, sigma_2_d1, sigma_3_d1])
    state = jnp.hstack([xi, eta, dot_xi, dot_eta])

    # Compute reference control
    I = jnp.diag(jnp.array([Ixx, Iyy, Izz]))  # noqa: E741

    dot_c = zb.dot(dot_a)
    ddot_a = jnp.array([sigma_1_d4, sigma_2_d4, sigma_3_d4])

    dot_q = (1.0 / c) * (xb.dot(ddot_a) - 2.0 * dot_c * q - c * p * r)
    dot_p = (1.0 / c) * (-yb.dot(ddot_a) - 2.0 * dot_c * p + c * q * r)
    dot_r = (1.0 / jnp.linalg.norm(jnp.cross(yc, zb))) * (
        dot_q * yc.dot(zb)
        + sigma_4_d2 * xc.dot(xb)
        + 2.0 * sigma_4_d1 * r * xc.dot(yb)
        - 2.0 * sigma_4_d1 * q * xc.dot(zb)
        - p * q * yc.dot(yb)
        - p * r * yc.dot(zb)
    )

    dot_nu = jnp.array([dot_p, dot_q, dot_r])

    tau_b = I @ dot_nu + jnp.cross(nu, I @ nu)

    S = jnp.array(
        [
            [k, k, k, k],
            [0.0, -k * l, 0.0, k * l],
            [-k * l, 0.0, k * l, 0.0],
            [-b, b, -b, b],
        ]
    )

    u = jnp.linalg.inv(S) @ jnp.hstack([jnp.array([m * c]), tau_b])

    return jnp.hstack([state, u])


# \sigma is the differentially flat input \sigma = [x, y, z, \psi]
def get_sigma(t):
    tfac = 1.0
    x = 2.0 * jnp.cos(jnp.sqrt(2) * t / tfac)
    y = 2.0 * jnp.sin(jnp.sqrt(2) * t / tfac) * jnp.cos(jnp.sqrt(2) * t / tfac)
    z = 0.0
    phi = 0.0
    return jnp.array([x, y, z, phi])


get_sigma_d1 = jax.jacobian(get_sigma)
get_sigma_d2 = jax.jacobian(get_sigma_d1)
get_sigma_d3 = jax.jacobian(get_sigma_d2)
get_sigma_d4 = jax.jacobian(get_sigma_d3)


def get_state_and_control(t, quad_params):
    traj_and_u_ref = differentially_flat_state_and_control(
        get_sigma(t),
        get_sigma_d1(t),
        get_sigma_d2(t),
        get_sigma_d3(t),
        get_sigma_d4(t),
        **quad_params,
    )
    return traj_and_u_ref


generate_trajectory = jax.vmap(get_state_and_control, (0, None))
