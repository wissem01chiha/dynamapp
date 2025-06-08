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

from math import ceil
import jax.numpy as jnp

nx = 2
nu = 1
ny = 1

Q = 1.0e-01 * jnp.eye(nu)  # process noise
R = 1.0e-02 * jnp.eye(ny)  # measurement noise

u_eq = jnp.array([0.0])  # Equilibrium control for the down configuration
x_eq = jnp.array([0.0, 0.0])  # Equilibrium state for the down configuration

x0 = jnp.array([jnp.pi / 20, 0.0])  # Initial state
x_hat_bar_0 = jnp.array(
    [jnp.pi / 15.0, 0.1]
)  # Initial state estimate relative to equilibrium
P_hat_bar_0 = 0.01 * jnp.eye(
    nx
)  # Initial covariance estimate of state relative to equilibrium

dt = 0.01  # time-step for discretization

Tsim = 10.0  # total simulation time
nseg = ceil(Tsim / dt)
