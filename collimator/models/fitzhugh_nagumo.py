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


# Define the system.
class FitzHughNagumo(LeafSystem):
    def __init__(self, x0=[0.0, 0.0], I_ext=1.0, R=1.0, a=0.7, b=0.8, tau=12.5):
        super().__init__(self)
        self.declare_dynamic_parameter("I_ext", I_ext)
        self.declare_dynamic_parameter("R", R)
        self.declare_dynamic_parameter("a", a)
        self.declare_dynamic_parameter("b", b)
        self.declare_dynamic_parameter("tau", tau)
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        v, w = state.continuous_state
        I_ext = parameters["I_ext"]
        R = parameters["R"]
        a = parameters["a"]
        b = parameters["b"]
        tau = parameters["tau"]
        return jnp.array([(v - w - v**3 / 3.0 + R * I_ext), (v + a - b * w) / tau])
