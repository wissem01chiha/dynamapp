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
'fluid_i' stands for 'fluid incompressible', e.g. water, hydraulic fluid, etc. when operating
under conditions where their compressibility can be neglected.

The concepts are heavily based on these sources:
Modelica Fluids library
https://doc.modelica.org/Modelica%204.0.0/Resources/helpDymola/Modelica_Fluid.html#Modelica.Fluid

Modelica Stream connectors concepts:
https://doc.modelica.org/Modelica%204.0.0/Resources/Documentation/Fluid/Stream-Connectors-Overview-Rationale.pdf

This implementation omits the stream connector/variables presently.

Defining the properties of the fluid in the components is achieved using the FluidProperties component.
FluidProperties has one acausal port. this port is connected anywhere in the network of components that
all share the same fluid. In this way, it is possible to to have a single AcausalDiagram, but with several
fluid networks, each having different fluid properties. The DiagramProcessing class handles appropriately
assigning the fluid properties from the FluidProperties component to the ports of the components connected
to the network.

Incomporessible Fluid Domain variables:
flow:Units = massflow:kg/s
potential:Units = pressure:Pa
"""


class FluidOnePort(ComponentBase):
    """Partial component class for an fluid component with only
    one port.
    """

    def __init__(self, ev, name, P_ic=None, P_ic_fixed=False, p="port"):
        super().__init__()
        self.P, self.M = self.declare_fluid_port(
            ev, p, P_ic=P_ic, P_ic_fixed=P_ic_fixed
        )
        self.port_idx_to_name = {-1: p}


class FluidTwoPort(ComponentBase):
    """Partial component class for an fluid component with
    two ports.
    """

    def __init__(self, ev, name, p1="port_a", p2="port_b"):
        super().__init__()
        self.P1, self.M1 = self.declare_fluid_port(ev, p1)
        self.P2, self.M2 = self.declare_fluid_port(ev, p2)
        self.dP = self.declare_symbol(ev, "dP", name, kind=SymKind.var)
        self.add_eqs([sp.Eq(self.dP.s, self.P1.s - self.P2.s)])
        self.port_idx_to_name = {-1: p1, 1: p2}


class Accumulator(FluidOnePort):
    """
    Accumulator in the incompressible fluid domain. Pressure increases when mass flows in, and vice versa.
    The relationship between internal pressure and mass flow is spring law. (ideal gass law coming soon)
    There is no restrictor at the port, no pressure loss a function of flow rate.

    Note: Initial pressure P_ic must be set. It has been observed that trying to use weak ICs for this
    i.e. leaving P_ic as None, produces problematic system equations.

    Spring law equations.
        massflow/(fluid.density) = der(V)
        force = (V/area)*k
        pressure = force/area

        V_init = f*area/k = pressure*area*area/k (N/m*m)*(m*m)*(m*m)/(N/m)->(N)*(m*m)/(N/m)->m**3

    Args:
        P_ic (number):
            Initial pressure of the accumulator.
        area (number):
            The surface area acted on by the accumulator internal pressure.
        k (number):
            Spring stiffness.

    """

    def __init__(
        self,
        ev,
        name=None,
        P_ic=0.0,
        P_ic_fixed=False,
        area=1.0,
        k=1.0,
    ):
        self.name = self.__class__.__name__ if name is None else name

        super().__init__(ev, self.name, P_ic=P_ic, P_ic_fixed=P_ic_fixed)
        V_ic = P_ic * area * area / k
        area = self.declare_symbol(
            ev,
            "area",
            self.name,
            kind=SymKind.param,
            val=area,
            validator=lambda area: area > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have area>0",
        )
        k = self.declare_symbol(
            ev,
            "k",
            self.name,
            kind=SymKind.param,
            val=k,
            validator=lambda k: k > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have k>0",
        )
        f = self.declare_symbol(ev, "f", self.name, kind=SymKind.var)
        V = self.declare_symbol(ev, "V", self.name, kind=SymKind.var, ic=V_ic)
        self.dV = self.declare_symbol(
            ev, "dV", self.name, kind=SymKind.var, int_sym=V, ic=0.0
        )
        V.der_sym = self.dV
        self.add_eqs(
            [
                # pressure-force relationship
                sp.Eq(self.P.s, f.s / area.s),
                # force-volume relationship for 'spring' accumulator.
                sp.Eq(f.s, V.s * k.s / area.s),
            ]
        )

    def finalize(self):
        # volume-massflow relationship.
        # this happens here because the port fluid properties are not yet assigned
        # at the time __init__ is called.
        fluid = self.ports["port"].fluid
        self.add_eqs([sp.Eq(self.dV.s, self.M.s / fluid.density.s)])


class FluidProperties(ComponentBase):
    """Component for assigning a Fluid to a set of components by connecting
    the FluidProperties component to the network via its 'prop' port.
    """

    def __init__(self, ev, fluid, name="FluidProperties"):
        super().__init__()
        self.name = name
        # this may not be strictly necessary, but defining the port like this
        # means thta all ports of all components conform to the same data, so
        # the same validation check can be used everywhere.
        P, M = self.declare_fluid_port(ev, "prop")
        self.add_eqs([sp.Eq(0, M.s)])
        self.fluid = fluid
        self.port_idx_to_name = {-1: "prop"}


class HydraulicActuatorLinear(FluidTwoPort):
    """
    Converters pressure to force, and mass_flow to translational motion.
    Component has 2 mechanical connections like a spring, flange_a and flange_b.
    Components has 2 fluid connections, like a hydraulic actuator, port_a, and port_b.
    dP=P1-P2
    dF=F1-F2
    Assuming all Ps and Fs are >0, when dP*area>pF, p1 and p2 get further apart, and vice versa.
    I'm not sure they have to all be positive for that to hold, but I know when they are, it does.
    """

    def __init__(self, ev, name=None, area=1.0, x_ic=0.0, x_ic_fixed=False):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        f1, x1, v1, a1 = self.declare_translational_port(ev, "flange_a")
        f2, x2, v2, a2 = self.declare_translational_port(
            ev,
            "flange_b",
            x_ic=x_ic,
            x_ic_fixed=x_ic_fixed,
        )
        self.add_eqs([sp.Eq(0, f1.s + f2.s)])

        V_ic = x_ic * area
        V = self.declare_symbol(ev, "V", self.name, kind=SymKind.var, ic=V_ic)
        area = self.declare_symbol(
            ev,
            "area",
            self.name,
            kind=SymKind.param,
            val=area,
            validator=lambda area: area > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have area>0",
        )
        self.dV = self.declare_symbol(
            ev, "dV", self.name, kind=SymKind.var, int_sym=V, ic=0.0
        )
        V.der_sym = self.dV

        self.add_eqs(
            [
                # force<->pressure relationships
                sp.Eq(self.dP.s * area.s, -f2.s),
                # the 'tracked' volume of fluid increases when the actuator gets longer.
                sp.Eq(V.s / area.s, x1.s - x2.s),
                # mass_flow constrain. e.g. conservation of mass.
                # here we consider the actuator 'ideal' in the sense that piston area on both sides is equal.
                # in reality this is not true, because there is typically one side with a rod which reduces
                # the piston area.
                sp.Eq(0, self.M1.s + self.M2.s),
            ]
        )

    def finalize(self):
        # velocity<->mass_flow relationships
        # the 'tracked' volume of fluid in the actuator increases when M1 is positive
        fluid = self.ports["port_a"].fluid
        self.add_eqs(
            [
                sp.Eq(self.dV.s, self.M1.s / fluid.density.s),
            ]
        )


class MassflowSensor(FluidTwoPort):
    """.
    Ideal massflow sensor in the incompressible fluid domain.
    """

    def __init__(
        self,
        ev,
        name=None,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        m_flow = self.declare_symbol(ev, "m_flow", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(m_flow.s, self.M1.s), kind=EqnKind.outp)
        self.add_eqs(
            [
                # does not accumulate mass
                sp.Eq(0, self.M1.s + self.M2.s),
                # no pressure drop
                sp.Eq(self.P1.s, self.P2.s),
            ]
        )


class MassflowSource(FluidOnePort):
    """
    Ideal massflow source in the incompressible fluid domain.
    """

    def __init__(self, ev, name=None, M=1.0, enable_massflow_port=False):
        raise NotImplementedError("Fluid MassflowSource not implemented.")


class Pipe(FluidTwoPort):
    """
    Simple pipe in the incompressible fluid domain.The characteristic equation is:
    P1(t) - P2(t) = massflow(t)*R.

    This is obviously overly simplified, but an acceptable starting point.
    """

    def __init__(self, ev, name=None, R=1.0):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

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
                # does not store mass
                sp.Eq(0, self.M1.s + self.M2.s),
                # pressure drop equation. use M1 due to sign convention
                sp.Eq(self.dP.s, R.s * self.M1.s),
            ]
        )


class PressureSensor(ComponentBase):
    """
    Ideal pressure sensor in the incompressible fluid domain.
    When port_b enabled, measures between port_a and port_b.
    When port_b disbaled, measures the absolute pressure.
    """

    def __init__(
        self,
        ev,
        name=None,
        enable_port_b=True,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        P1, M1 = self.declare_fluid_port(ev, "port_a")
        self.port_idx_to_name = {-1: "port_a"}
        if enable_port_b:
            P2, M2 = self.declare_fluid_port(ev, "port_b")
            self.port_idx_to_name = {1: "port_b"}

        p = self.declare_symbol(ev, "p", self.name, kind=SymKind.outp)

        if enable_port_b:
            self.declare_equation(sp.Eq(p.s, P1.s - P2.s), kind=EqnKind.outp)
            self.add_eqs([sp.Eq(M1.s, 0), sp.Eq(M1.s, 0)])
        else:
            self.declare_equation(sp.Eq(p.s, P1.s), kind=EqnKind.outp)
            self.add_eqs([sp.Eq(M1.s, 0)])


class PressureSource(FluidOnePort):
    """
    Ideal pressure source in the incompressible fluid domain.

    Args:
        pressure (number):
            Pressure value when enable_speed_port=False.
        enable_pressure_port (bool):
            When true, the pressure value is from a input signal. When false,
            pressure value is from 'pressure'.
    """

    def __init__(self, ev, name=None, pressure=0.1, enable_pressure_port=False):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        if enable_pressure_port:
            kind = SymKind.inp
            val = None
        else:
            kind = SymKind.param
            val = pressure

        # in this case we have to create an additional symbol since it is not OK
        # to change the kind of a potential/flow variable.
        pressure = self.declare_symbol(ev, "pressure", self.name, kind=kind, val=val)
        self.add_eqs([sp.Eq(self.P.s, pressure.s)])  # pressure source equality


class Pump(FluidTwoPort):
    """
    Ideal pump in the incompressible fluid domain.

    The pump has a maximum pressure differential, dP_max, which it can produce.
    Therefore, the max outlet pressure is dP_max - P_in.
    The pump has some input power, pwr, which is a causal input signal,
    that represents the external work done by the pump on the fluid
    system.
    The pump has some performance coefficient that converts dP*pwr to massflow,
    CoP, which has units of kg*s*MPa/Watt.
        massflow = pwr * CoP / ((dP_max - Pin) - P_out) [1]
        but dP = P_out - P_in -> P_out = dP + P_in [2]
    subing [2] into [1] and rearranging:
        massflow = pwr * CoP / (dP_max - dP)

    Note on sign convention.
    Pump component has 2 ports: p1=inlet and p2=outlet.
    massflow is defined as positive going into the component.
    Therefore the inlet massflow is positive when pump operating normally.

    """

    def __init__(
        self,
        ev,
        name=None,
        dPmax=1.0,
        CoP=1.0,
    ):
        self.name = self.__class__.__name__ if name is None else name

        super().__init__(ev, self.name)
        dPmax = self.declare_symbol(
            ev,
            "dPmax",
            self.name,
            kind=SymKind.param,
            val=dPmax,
            validator=lambda dPmax: dPmax > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have dPmax>0",
        )
        CoP = self.declare_symbol(
            ev,
            "CoP",
            self.name,
            kind=SymKind.param,
            val=CoP,
            validator=lambda CoP: CoP > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have CoP>0",
        )
        pwr = self.declare_symbol(ev, "pwr", self.name, kind=SymKind.inp)
        self.add_eqs(
            [
                # does not store mass
                sp.Eq(0, self.M1.s + self.M2.s),
                # pressure-massflow relationship.
                sp.Eq(self.M1.s, pwr.s * CoP.s / (dPmax.s - self.dP.s)),
            ]
        )
