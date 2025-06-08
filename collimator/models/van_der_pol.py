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


class VanDerPol(LeafSystem):
    def __init__(self, x0=[0.0, 0.0], mu=1.0, input_port=False, name="van_der_pol"):
        super().__init__(name=name)
        self.declare_dynamic_parameter("mu", mu)

        if input_port:
            self.declare_input_port(name="u")

        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        x, y = state.continuous_state
        mu = parameters["mu"]
        dy = mu * (1 - x**2) * y - x

        if inputs:
            (u,) = inputs
            dy += u

        return jnp.array([y, dy])
