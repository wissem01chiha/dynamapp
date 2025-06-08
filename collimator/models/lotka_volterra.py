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


class LotkaVolterra(LeafSystem):
    def __init__(self, x0=[10.0, 10.0], alpha=1.1, beta=0.4, gamma=0.4, delta=0.1):
        super().__init__(self)
        self.declare_dynamic_parameter("alpha", alpha)
        self.declare_dynamic_parameter("beta", beta)
        self.declare_dynamic_parameter("gamma", gamma)
        self.declare_dynamic_parameter("delta", delta)
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        x, y = state.continuous_state
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        gamma = parameters["gamma"]
        delta = parameters["delta"]
        return jnp.array([(alpha * x - beta * x * y), (delta * x * y - gamma * y)])
