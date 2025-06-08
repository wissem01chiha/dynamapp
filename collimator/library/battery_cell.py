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

from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple

from ..framework import LeafSystem, DependencyTicket, parameters
from ..backend import numpy_api as cnp

if TYPE_CHECKING:
    from ..backend.typing import Array


__all__ = ["BatteryCell"]

Ah_to_As = 3600.0  # Convert Ah to As


class BatteryCell(LeafSystem):
    """Dynamic electro-checmical Li-ion cell model.

    Based on [Tremblay and Dessaint (2009)](https://doi.org/10.3390/wevj3020289).

    By using appropriate parameters, the cell model can be used to model a battery pack
    with the assumption that the cells of the pack behave as a single unit.

    Parameters E0, K, A, below are abstract parameters used in the model presented in
    the reference paper. As described in the reference paper, these parameters can be
    extracted from typical cell manufacturer datasheets; see section 3. Section 3 also
    provides a table of example values for these parameters.

    Input ports:
        (0) The current (A) flowing through the cell. Positive is discharge.

    Output ports:
        (0) The voltage across the cell terminals (V)
        (1) The state of charge of the cell (normalized between 0 and 1)

    Parameters:
        E0: described as "battery constant voltage (V)" by the reference paper.
        K: described as "polarization constant (V/Ah)" by the reference paper.
        Q: battery capacity in Ah
        R: internal resistance (Ohms)
        A: described as "exponential zone amplitude (V)" by the reference paper.
        B:
            described as "exponential zone time constant inverse (1/Ah)" by the
            reference paper.
        initial_SOC: initial state of charge, normalized between 0 and 1.
    """

    class BatteryStateType(NamedTuple):
        soc: float
        i_star: float
        i_lb: float

    class FirstOrderFilter(NamedTuple):
        A: float
        B: float
        C: float

    @parameters(dynamic=["E0", "K", "Q", "R", "tau", "A", "B", "initial_SOC"])
    def __init__(
        self,
        E0: float = 3.366,
        K: float = 0.0076,
        Q: float = 2.3,
        R: float = 0.01,
        tau: float = 30.0,
        A: float = 0.26422,
        B: float = 26.5487,
        initial_SOC: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.declare_input_port()  # Current flowing through the cell

        self.declare_output_port(
            self._voltage_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            name="voltage",
        )

        self.declare_output_port(
            self._soc_output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
            name="soc",
        )

    def initialize(self, E0, K, Q, R, tau, A, B, initial_SOC):
        # Filter for input current
        self.current_filter = self.FirstOrderFilter(-0.05, 1.0, 0.05)

        # Filter for loop-breaker
        self.lb_filter = self.FirstOrderFilter(-10.0, 1.0, 10.0)

        initial_state = self.BatteryStateType(
            soc=initial_SOC,
            i_star=0.0,  # Filtered input current
            i_lb=0.0,  # Filtered current for loop-breaking
        )

        self.declare_continuous_state(
            default_value=initial_state,
            as_array=False,
            ode=self._ode,
        )

    def _ode(self, _time, state, *inputs, **parameters) -> BatteryStateType:
        xc = state.continuous_state
        Q = parameters["Q"]

        (u,) = inputs

        soc_der_unsat = -u / (Q * Ah_to_As)

        # SoC must be between 0 and 1
        llim_violation = (xc.soc <= 0.0) & (soc_der_unsat < 0.0)
        ulim_violation = (xc.soc >= 1.0) & (soc_der_unsat > 0.0)

        # Saturated time derivative
        soc_der = cnp.where(llim_violation | ulim_violation, 0.0, soc_der_unsat)

        # Derivative of istar, the filtered current signal
        i_star_der = self.current_filter.A * xc.i_star + self.current_filter.B * u

        # Derivative of ilb, the filtered current signal for loop-breaking
        i_lb_der = self.lb_filter.A * xc.i_lb + self.lb_filter.B * u

        return self.BatteryStateType(
            soc=soc_der,
            i_star=i_star_der,
            i_lb=i_lb_der,
        )

    def _voltage_output(self, _time, state, *_inputs, **parameters) -> Array:
        E0 = parameters["E0"]
        Q = parameters["Q"]
        K = parameters["K"]
        A = parameters["A"]
        B = parameters["B"]
        R = parameters["R"]
        xc = state.continuous_state

        # Filtered input current
        i_star = self.current_filter.C * xc.i_star

        # Loop-breaking current
        i_lb = self.lb_filter.C * xc.i_lb

        # Apply limits to state of charge
        soc = cnp.clip(xc.soc, 0.0, 1.0)

        # Undo normalization by Q - this is ∫i*dt, the integral of current
        i_int = Q * (1 - soc)

        chg_mode_Q_gain = 0.1
        vdyn_den = cnp.where(i_star >= 0, Q - i_int, i_int + chg_mode_Q_gain * Q)
        vdyn = i_star * K * Q / vdyn_den

        vbatt_ulim = 2 * E0  # Reasonable upper limit on battery voltage
        vbatt_presat = (
            E0 - R * i_lb - i_int * K * Q / (Q - i_int) + A * cnp.exp(-B * i_int) - vdyn
        )
        return cnp.clip(vbatt_presat, 0.0, vbatt_ulim)

    def _soc_output(self, _time, state, *_inputs, **_parameters) -> Array:
        return state.continuous_state.soc
