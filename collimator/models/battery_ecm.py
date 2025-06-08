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

from collimator.framework import LeafSystem


class Battery(LeafSystem):
    """
    Continuous-time system for an Equivalent Circuit Model of a battery.
    open-circuit voltage, Resistances and Capacitances are linearly
    interpolated against SoC from provided input data.
    """

    def __init__(
        self,
        soc_0=1.0,  # initial value of SoC
        vc1_0=0.0,  # initial value of voltage across capacitor
        Q=100.0,  # total capacity of the battery
        soc_points=jnp.linspace(
            0, 1, 11
        ),  # discrete SoC points at which v_0, Rs, R1, and C1 are specified
        v0_points=5.0 * jnp.ones(11),  # discrete values of v0 at soc_points
        Rs_points=15e-03 * jnp.ones(11),  # discrete values of Rs at soc_points
        R1_points=25e-03 * jnp.ones(11),  # discrete values of R1 at soc_points
        C1_points=3e03 * jnp.ones(11),  # discrete values of C1 at soc_points
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.soc_0 = soc_0
        self.vc1_0 = vc1_0
        self.Q = Q

        self.soc_points = soc_points
        self.declare_dynamic_parameter(
            "v0_points", v0_points
        )  # declare as parameters so that these can be changed in the context
        self.declare_dynamic_parameter("Rs_points", Rs_points)
        self.declare_dynamic_parameter("R1_points", R1_points)
        self.declare_dynamic_parameter("C1_points", C1_points)

        self.declare_input_port()  # input port for the discharge current

        self.declare_continuous_state(
            shape=(2,),  # Two continuous states: soc and vc1
            ode=self.ode,
            default_value=jnp.array([self.soc_0, self.vc1_0]),
        )

        self.declare_output_port(
            self._eval_output_soc, default_value=0.5
        )  # output port for soc
        self.declare_output_port(
            self._eval_output_vt, default_value=2.5
        )  # output port for terminal voltage vt

    def _eval_output_soc(
        self, time, state, current, **params
    ):  # output port evaluation for soc
        x = state.continuous_state
        soc = x[0]
        return soc

    def _eval_output_vt(
        self, time, state, current, **params
    ):  # output port evaluation for vt
        x = state.continuous_state
        soc = x[0]
        vc1 = x[1]

        v0 = jnp.interp(soc, self.soc_points, params["v0_points"])
        Rs = jnp.interp(soc, self.soc_points, params["Rs_points"])

        vt = v0 - current * Rs - vc1  # compute terminal voltage
        return vt

    def ode(self, time, state, current, **params):  # ODE system in soc and vt
        x = state.continuous_state
        soc = x[0]
        vc1 = x[1]

        R1 = jnp.interp(soc, self.soc_points, params["R1_points"])
        C1 = jnp.interp(soc, self.soc_points, params["C1_points"])
        dot_soc = -current / 3600.0 / self.Q  # compute d/dt (soc)
        dot_vc1 = current / C1 - vc1 / R1 / C1  # compute d/dt (vc1)
        return jnp.array([dot_soc, dot_vc1])
