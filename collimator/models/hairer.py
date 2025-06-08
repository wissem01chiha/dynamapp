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

import numpy as np
from ..backend import numpy_api as cnp
from ..framework import LeafSystem

__all__ = [
    "EulerRigidBody",
    "ArenstorfOrbit",
]


class EulerRigidBody(LeafSystem):
    def __init__(
        self, I1=0.5, I2=2.0, I3=3.0, x0=np.array([1.0, 0.0, 0.9]), name="euler"
    ):
        # Euler's equation of rotation for a rigid body
        super().__init__(name=name)

        self.declare_dynamic_parameter("I1", I1)
        self.declare_dynamic_parameter("I2", I2)
        self.declare_dynamic_parameter("I3", I3)

        self.declare_continuous_state(default_value=x0, ode=self._ode)
        self.declare_continuous_state_output()

    def _ode(self, time, state, *inputs, **parameters):
        I1, I2, I3 = parameters["I1"], parameters["I2"], parameters["I3"]
        x = state.continuous_state
        f = cnp.where(
            (3 * np.pi <= time) & (time <= 4 * np.pi),
            0.25 * cnp.sin(time) ** 2,
            0.0,
        )
        return cnp.array(
            [
                (I2 - I3) * x[1] * x[2] / I1,
                (I3 - I1) * x[2] * x[0] / I2,
                ((I1 - I2) * x[0] * x[1] + f) / I3,
            ]
        )


class ArenstorfOrbit(LeafSystem):
    def __init__(self, name="arenstorf"):
        super().__init__(name=name)
        self.declare_dynamic_parameter("mu", 0.012277471)
        y0 = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])

        self.declare_continuous_state(default_value=y0, ode=self._ode)
        self.declare_continuous_state_output()

    def _ode(self, time, state, *inputs, **parameters):
        mu = parameters["mu"]
        y1, y2, y3, y4 = state.continuous_state
        r1 = ((y1 + mu) ** 2 + y2**2) ** 1.5
        r2 = ((y1 - 1 + mu) ** 2 + y2**2) ** 1.5
        return cnp.array(
            [
                y3,
                y4,
                y1 + 2 * y4 - (1 - mu) * (y1 + mu) / r1 - mu * (y1 - 1 + mu) / r2,
                y2 - 2 * y3 - (1 - mu) * y2 / r1 - mu * y2 / r2,
            ]
        )


class Lorenz(LeafSystem):
    def __init__(self, name="lorenz"):
        super().__init__(name=name)
        self.declare_dynamic_parameter("sigma", 10.0)
        self.declare_dynamic_parameter("rho", 28.0)
        self.declare_dynamic_parameter("beta", 8.0 / 3.0)
        y0 = np.array([-8.0, 8.0, 27.0])
        self.declare_continuous_state(default_value=y0, ode=self._ode)
        self.declare_continuous_state_output()

    def _ode(self, time, state, *inputs, **parameters):
        sigma = parameters["sigma"]
        rho = parameters["rho"]
        beta = parameters["beta"]
        y = state.continuous_state
        return cnp.array(
            [
                sigma * (y[1] - y[0]),
                y[0] * (rho - y[2]) - y[1],
                y[0] * y[1] - beta * y[2],
            ]
        )


class Pleiades(LeafSystem):
    def __init__(self, name="pleiades"):
        super().__init__(name=name)
        self.declare_dynamic_parameter(
            "m", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        )
        x0 = np.array([3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0])
        y0 = np.array([3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0])
        dx0 = np.zeros(7)
        dx0[5] = 1.75
        dx0[6] = -1.5
        dy0 = np.zeros(7)
        dy0[3] = -1.25
        dy0[4] = 1.0

        init_state = np.concatenate([x0, y0, dx0, dy0])
        self.declare_continuous_state(default_value=init_state, ode=self._ode)
        self.declare_continuous_state_output()

    def _ode(self, time, state, *inputs, **parameters):
        m = parameters["m"]
        xc = state.continuous_state
        x, y, dx, dy = xc[:7], xc[7:14], xc[14:21], xc[21:]
        ddx = cnp.zeros(7)
        ddy = cnp.zeros(7)
        for i in range(7):
            for j in range(7):
                r = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** (3 / 2)
                r = cnp.maximum(r, 1e-6)
                ddx = cnp.where(
                    i != j,
                    ddx.at[i].add(m[j] * (x[j] - x[i]) / r),
                    ddx,
                )
                ddy = cnp.where(
                    i != j,
                    ddy.at[i].add(m[j] * (y[j] - y[i]) / r),
                    ddy,
                )
        return cnp.concatenate([dx, dy, ddx, ddy])
