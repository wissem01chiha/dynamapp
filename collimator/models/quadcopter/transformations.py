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

import jax.numpy as jnp  # noqa: E402


def euler_to_rotation_matrix(phi, theta, psi):
    R_x = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(phi), -jnp.sin(phi)],
            [0.0, jnp.sin(phi), jnp.cos(phi)],
        ]
    )

    R_y = jnp.array(
        [
            [jnp.cos(theta), 0.0, jnp.sin(theta)],
            [0.0, 1.0, 0.0],
            [-jnp.sin(theta), 0.0, jnp.cos(theta)],
        ]
    )

    R_z = jnp.array(
        [
            [jnp.cos(psi), -jnp.sin(psi), 0.0],
            [jnp.sin(psi), jnp.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return jnp.matmul(R_z, jnp.matmul(R_y, R_x))


def get_Wn(phi, theta, psi):
    Wn = jnp.array(
        [
            [1.0, 0.0, -jnp.sin(theta)],
            [0.0, jnp.cos(phi), jnp.cos(theta) * jnp.sin(phi)],
            [0.0, -jnp.sin(phi), jnp.cos(theta) * jnp.cos(phi)],
        ]
    )
    return Wn


def get_Wn_inv(phi, theta, psi):
    Wn_inv = jnp.array(
        [
            [1.0, jnp.sin(phi) * jnp.tan(theta), jnp.cos(phi) * jnp.tan(theta)],
            [0.0, jnp.cos(phi), -jnp.sin(phi)],
            [0.0, jnp.sin(phi) / jnp.cos(theta), jnp.cos(phi) / jnp.cos(theta)],
        ]
    )
    return Wn_inv


def get_dot_Wn_inv(phi, theta, psi, dot_phi, dot_theta):
    C_phi = jnp.cos(phi)
    S_phi = jnp.sin(phi)
    C_theta = jnp.cos(theta)
    S_theta = jnp.sin(theta)  # noqa: F841
    T_theta = jnp.tan(theta)

    dot_Wn_inv = jnp.array(
        [
            [
                0.0,
                dot_phi * C_phi * T_theta + dot_theta * S_phi / C_theta**2,
                -dot_phi * S_phi * T_theta + dot_theta * C_phi / C_theta**2,
            ],
            [0.0, -dot_phi * S_phi, -dot_phi * C_phi],
            [
                0.0,
                dot_phi * C_phi / C_theta + dot_theta * S_phi * T_theta / C_theta,
                -dot_phi * S_phi / C_theta + dot_theta * C_phi * T_theta / C_theta,
            ],
        ]
    )

    return dot_Wn_inv
