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

from typing import TYPE_CHECKING
from collimator.lazy_loader import LazyLoader

from .base import SymKind, EqnKind
from .component_base import ComponentBase

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")

"""
1D thernal (heat transfer) components similar to Modelica Standard Library.

flow variable:Units = heat_flux:Joules/second
potential variable:Units = temperature:Kelvin
"""


class ThermalOnePort(ComponentBase):
    """Partial component class for a thermal component with one port."""

    def __init__(self, ev, name, T_ic=300, T_ic_fixed=False, p="port"):
        super().__init__()
        self.T, self.Q = self.declare_thermal_port(
            ev, p, T_ic=T_ic, T_ic_fixed=T_ic_fixed
        )
        self.port_idx_to_name = {-1: p}


class ThermalTwoPort(ComponentBase):
    """Partial component class for a thermal component with two
    port that can have different temperature relative to each other.
    """

    def __init__(self, ev, name, p1="port_a", p2="port_b"):
        super().__init__()
        self.T1, self.Q1 = self.declare_thermal_port(ev, p1)
        self.T2, self.Q2 = self.declare_thermal_port(ev, p2)
        self.dT = self.declare_symbol(ev, "dT", name, kind=SymKind.var)
        self.add_eqs([sp.Eq(self.dT.s, self.T1.s - self.T2.s)])
        self.port_idx_to_name = {-1: p1, 1: p2}


class HeatCapacitor(ThermalOnePort):
    """
    Ideal capacitor(thermal mass) in thermal domain. The characteristic equation is:
    heatflow(t) = derivative(T(t))*C, where C is the product of the mass and the
    specific heat. The units of C are in Watt/degK.

    Agrs:
        C (number):
            Mass * specific heat.
        initial_temperature (number);
            initial temperature.
    """

    def __init__(
        self,
        ev,
        name=None,
        C=1.0,
        initial_temperature=300,
        initial_temperature_fixed=False,
    ):
        self.name = self.__class__.__name__ if name is None else name

        super().__init__(
            ev,
            self.name,
            T_ic=initial_temperature,
            T_ic_fixed=initial_temperature_fixed,
        )
        C = self.declare_symbol(
            ev,
            "C",
            self.name,
            kind=SymKind.param,
            val=C,
            validator=lambda C: C > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have C>0",
        )
        derT = self.declare_symbol(
            ev, "derT", self.name, kind=SymKind.var, int_sym=self.T, ic=0.0
        )
        self.T.der_sym = derT

        # energy relationship
        self.add_eqs([sp.Eq(self.Q.s, C.s * derT.s)])


class HeatflowSensor(ThermalTwoPort):
    """
    Ideal heatflow sensor in thermal domain.
    Measures heatflow between port_a and port_b.
    """

    def __init__(
        self,
        ev,
        name=None,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        Q_flow = self.declare_symbol(ev, "Q_flow", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(Q_flow.s, self.Q1.s), kind=EqnKind.outp)
        self.add_eqs([sp.Eq(self.T1.s, self.T2.s)])


class HeatflowSource(ComponentBase):
    """
    Ideal heatflow source in thermal domain.

    Args:
        Q_flow (number):
            Heatflow value when enable_torque_port=False.
        enable_heat_port (bool):
            When true, the heatflow value is from a input signal. When false,
            heatflow value is from 'Q_flow'.
        enable_port_b (bool):
            When port_b enabled, applies heatflow between port_a and port_b.
            when port_b disbaled, applies absolute heatflow to port_a.
    """

    def __init__(
        self,
        ev,
        name=None,
        Q_flow=0.0,
        enable_heat_port=False,
        enable_port_b=True,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        T1, Q1 = self.declare_thermal_port(ev, "port_a")
        self.port_idx_to_name = {-1: "port_a"}

        if enable_heat_port:
            Q_flow = self.declare_symbol(ev, "Q_flow", self.name, kind=SymKind.inp)
        else:
            Q_flow = self.declare_symbol(
                ev, "Q_flow", self.name, kind=SymKind.param, val=Q_flow
            )

        self.add_eqs([sp.Eq(Q_flow.s, Q1.s)])
        if enable_port_b:
            T2, Q2 = self.declare_thermal_port(ev, "port_b")
            self.port_idx_to_name[1] = "port_b"
            self.add_eqs([sp.Eq(Q_flow.s, -Q2.s)])


class TemperatureSensor(ComponentBase):
    """
    Ideal temperature sensor in the thermal domain.

    Agrs:
        enable_port_b(bool):
            When port_b enabled, measures between port_a and port_b.
            When port_b disbaled, measures the absolute temperature.
    """

    def __init__(
        self,
        ev,
        name=None,
        enable_port_b=True,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        T1, Q1 = self.declare_thermal_port(ev, "port_a")
        self.port_idx_to_name = {-1: "port_a"}
        T_rel = self.declare_symbol(ev, "T_rel", self.name, kind=SymKind.outp)
        if enable_port_b:
            T2, Q2 = self.declare_thermal_port(ev, "port_b")
            self.port_idx_to_name[1] = "port_b"
            self.declare_equation(sp.Eq(T_rel.s, T1.s - T2.s), kind=EqnKind.outp)
            self.add_eqs([sp.Eq(Q1.s, 0), sp.Eq(Q2.s, 0)])
        else:
            self.declare_equation(sp.Eq(T_rel.s, T1.s), kind=EqnKind.outp)
            self.add_eqs([sp.Eq(Q1.s, 0)])


class TemperatureSource(ThermalOnePort):
    """
    Ideal temperature source in the thermal domain.

    Args:
        temperature (number):
            temperature value when enable_temperature_port=False.
        enable_temperature_port (bool):
            When true, the temperature value is from a input signal. When false, temperature
            value is from 'temperature'.
    """

    def __init__(self, ev, name=None, temperature=300.0, enable_temperature_port=False):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        if enable_temperature_port:
            kind = SymKind.inp
            val = None
        else:
            kind = SymKind.param
            val = temperature

        # in this case we have to create an additional symbol since it is not OK
        # to change the kind of a potential/flow variable.
        temperature = self.declare_symbol(
            ev, "temperature", self.name, kind=kind, val=val
        )

        # temperature source equality
        self.add_eqs([sp.Eq(self.T.s, temperature.s)])


class ThermalInsulator(ThermalTwoPort):
    """
    Ideal insulator in thermal domain. The characteristic equation is:
    T1(t) - T2(t) = heatflow(t)*R, where R is the insulation coefficient in degK/Watt.

    Can be used to model heat transfer by conduction, and/or convection.

    Args:
        R (number):
            the insulation coefficient in degK/Watt
        enable_resistance_port (bool):
            When true, the value of R is from a input signal.
    """

    def __init__(self, ev, name=None, R=1.0, enable_resistance_port=False):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        if enable_resistance_port:
            R = self.declare_symbol(ev, "R", self.name, kind=SymKind.inp)
        else:
            R = self.declare_symbol(
                ev,
                "R",
                self.name,
                kind=SymKind.param,
                val=R,
                validator=lambda R: R > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have R>0",
            )

        self.add_eqs(
            [
                # does not store energy
                sp.Eq(0, self.Q1.s + self.Q2.s),
                # thermal conduction equation. use Q1 due to sign convention
                sp.Eq(self.dT.s, R.s * self.Q1.s),
            ]
        )


class ThermalRadiation(ThermalTwoPort):
    """
    models heat transfer by radiation. The characteristic equation is:
    T1(t)**4 - T2(t)**4 = heatflow(t)*Gr*signam, where Gr is the radiation coefficient in m**2.

    Args:
        Gr (number):
            the radiation coefficient in m**2
        enable_Gr_port (bool):
            When true, the value of Gr is from a input signal.
    """

    def __init__(self, ev, name=None, Gr=1.0, enable_Gr_port=False):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        if enable_Gr_port:
            Gr = self.declare_symbol(ev, "Gr", self.name, kind=SymKind.inp)
        else:
            Gr = self.declare_symbol(
                ev,
                "Gr",
                self.name,
                kind=SymKind.param,
                val=Gr,
                validator=lambda Gr: Gr > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Gr>0",
            )

        self.add_eqs(
            [
                # does not store energy
                sp.Eq(0, self.Q1.s + self.Q2.s),
                # thermal conduction equation. use Q1 due to sign convention
                sp.Eq(self.T1.s**4 - self.T2.s**4, Gr.s * self.Q1.s),
            ]
        )
