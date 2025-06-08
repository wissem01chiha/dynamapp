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

"""
Cost and loss function structures meant for utilization in optimal control,
optimization, ML tasks, etc.
"""

import jax.numpy as jnp

from ..library import ReduceBlock


class QuadraticCost(ReduceBlock):
    """LQR-type quadratic cost function for a state and input.

    Computes the cost as x'Qx + u'Ru, where Q and R are the cost matrices.
    In order to compute a running cost, combine this with an `Integrator`
    or `IntegratorDiscrete` block.
    """

    def __init__(self, Q, R, name=None):
        super().__init__(2, self._cost, name=name)
        self.Q = Q
        self.R = R

    def _cost(self, inputs):
        x, u = inputs
        J = jnp.dot(x, jnp.dot(self.Q, x)) + jnp.dot(u, jnp.dot(self.R, u))
        return J.squeeze()
