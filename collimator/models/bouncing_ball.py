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

from ..framework import LeafSystem


class BouncingBall(LeafSystem):
    def __init__(
        self, *args, g=9.81, e=1.0, b=0.0, h0=0.0, hdot=0.0, name="ball", **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self.declare_continuous_state(2, ode=self.ode)  # Two state variables.
        self.declare_continuous_state_output(name=f"{name}:y")
        self.declare_dynamic_parameter("g", g)
        self.declare_dynamic_parameter(
            "e", e
        )  # Resitiution coefficent (0.0 <= e <= 1.0)
        self.declare_dynamic_parameter("b", b)  # Quadratic drag coefficient
        self.declare_dynamic_parameter("hdot", hdot)  # Speed of floor
        self.declare_dynamic_parameter("h0", h0)  # Initial floor height

        self.declare_zero_crossing(
            guard=self._signed_distance,
            reset_map=self._reset,
            name="time_reset",
            direction="positive_then_non_positive",
        )

    def ode(self, time, state, **parameters):
        g = parameters["g"]
        b = parameters["b"]
        x, v = state.continuous_state
        return jnp.array([v, -g - b * v**2 * jnp.sign(v)])

    def floor_height(self, time, state, **parameters):
        h0 = parameters["h0"]
        hdot = parameters["hdot"]
        return h0 + hdot * time

    def _signed_distance(self, time, state, **parameters):
        x, v = state.continuous_state
        h = self.floor_height(time, state, **parameters)
        return x - h

    def _reset(self, time, state, **parameters):
        # Update velocity using Newtonian restitution model.
        x, v = state.continuous_state
        e = parameters["e"]
        hdot = parameters["hdot"]
        h = self.floor_height(time, state, **parameters)

        xc_post = jnp.array([h + jnp.abs(x - h), -e * v + hdot])
        return state.with_continuous_state(xc_post)
