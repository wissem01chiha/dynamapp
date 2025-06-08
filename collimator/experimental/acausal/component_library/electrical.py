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
from numbers import Number
from collimator.lazy_loader import LazyLoader
from .base import (
    SymKind,
    EqnKind,
)
from .component_base import ComponentBase
from ..error import AcausalModelError
from collimator.backend import numpy_api as cnp
from collimator.backend.typing import ArrayLike

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")

"""
Discussion on the design of the electrical components relative to Modelica Standard Library (MLS).

MSL Pin.mo defines the I(flow), and V(potential). We do the similar in ElecPort, but the symbols are passed in from
the component, to the port, so we get I1, I2, V1, V2.

MSL TwoPin.mo and OnePort.mo define the component I and V symbols. We do similar, but define V in ElecTwoPin,
and use the I1 from component for I.

In this way, the Resistor and Capacitor end up with essentially an analogous set of symbols, and
an equivalent set of equations relative to the MSL components.

Elecdtrical Domain variables:
flow:Units = current:Amps
potential:Units = voltage:Volts
"""


class ElecTwoPin(ComponentBase):
    """Partial component class for an electrical component with two
    pins, or two electrical terminals.
    """

    def __init__(
        self,
        ev,
        name,
        p1="p",
        p2="n",
        V_ic=None,
        V_ic_fixed=False,
        I_ic=None,
        I_ic_fixed=False,
    ):
        super().__init__()
        self.Vp, self.Ip = self.declare_electrical_port(
            ev, p1, I_ic=I_ic, I_ic_fixed=I_ic_fixed
        )
        self.Vn, self.In = self.declare_electrical_port(ev, p2)
        self.V = self.declare_symbol(
            ev, "V", name, kind=SymKind.var, ic=V_ic, ic_fixed=V_ic_fixed
        )

        self.add_eqs(
            [
                sp.Eq(self.Vp.s - self.Vn.s, self.V.s),
                sp.Eq(0, self.Ip.s + self.In.s),
            ]
        )
        # This defines that, in the WebApp, p1 is on the left and p2 on the right of the block.
        self.port_idx_to_name = {-1: p1, 1: p2}


class Battery(ElecTwoPin):
    """
    Component modeling battery cell using an Equivalent Circuit Model (ECM).
    The governing equations are:
        1. Vt = OCV(SOC) - I*R - Up(I), where:
            I is the current
            Vt is terminal voltage
            Up is polarization voltage
            OCV is open circuit voltage, a lookup table of SOC
            R is the internal resistance
        2. d(Up)/dt = I/Cp - Up/(Rp*Cp).
            Cp is the capacitance of the RC pair forming the polarization voltage.
            Rp is the resistance of the RC pair.
        3. d(SOC)dt = I/(AH*3600), this is known as "Coulomb Counting".
            AH is the Amp-Hours of the cell.

    Temperature effects are presently neglected. Or put another way, this only models the cell at one fixed
    temperature.
    OCV hysteresis observed in some Lithium chemistries, most commonly LiFePO4, is not yet modeled.

    Args
        AH (float):
            the Amp-Hours capacity of the cell.
        OCV_soc (array):
            The SOC break points of the OCV lookup table.
        OCV_volts (array):
            The voltage points of the OCV lookup table.
        R,Rp,Cp (float, array):
            These can be provided as either scalars, or an arrays. If as arrays,
            then their values are lookup tables of SOC, and SOC_v must be provided.
            See equations above for definitions of symbols.
        SOC_v (array):
            SOC break point for parameter looup tables. See above.
        initial_soc (float):
            The initial value of the SOC state.
        iniital_soc_fixed (bool):
            Whether the initial_soc condition is fixed or not.
        enable_<some>_port (bool):
            whether to expose a causal port that outputs <some> variable which can be used
            for control or debugging.
    """

    def __init__(
        self,
        ev,
        name=None,
        AH=1.0,
        OCV_soc=[0.0, 1.0],
        OCV_volts=[10.0, 15.0],
        R=0.01,
        Rp=1e-6,
        Cp=1e-6,
        SOC_v=None,
        initial_soc=0.5,
        initial_soc_fixed=False,
        enable_soc_port=False,
        enable_Up_port=False,
        enable_ocv_port=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        V_ic = cnp.interp(initial_soc, cnp.array(OCV_soc), cnp.array(OCV_volts))
        super().__init__(ev, self.name, V_ic=V_ic, I_ic=0.0)

        # declare symbols for the state variables
        SOC = self.declare_symbol(
            ev,
            "SOC",
            self.name,
            kind=SymKind.var,
            ic=initial_soc,
            ic_fixed=initial_soc_fixed,
        )
        dSOC = self.declare_symbol(
            ev,
            "dSOC",
            self.name,
            kind=SymKind.var,
            int_sym=SOC,
            ic=0.0,
        )
        SOC.der_sym = dSOC
        Up_ = self.declare_symbol(
            ev,
            "Up_",
            self.name,
            kind=SymKind.var,
            ic=0.0,  # assign string IC of 0.0
            ic_fixed=True,
        )
        dUp = self.declare_symbol(
            ev,
            "dUp",
            self.name,
            kind=SymKind.var,
            int_sym=Up_,
            ic=0.0,  # assign string IC of 0.0
            ic_fixed=True,
        )
        Up_.der_sym = dUp

        # declare symbols for the paramsters
        AH = self.declare_symbol(
            ev,
            "AH",
            self.name,
            kind=SymKind.param,
            val=AH,
            validator=lambda AH: AH > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have AH>0",
        )
        OCV_lut = self.declare_1D_lookup_table(
            ev, SOC.s, "OCV_soc", OCV_soc, "OCV_volts", OCV_volts, "OCV_lut"
        )

        scalar_types = (Number, ArrayLike)
        R_is_scalar = isinstance(R, scalar_types)
        Rp_is_scalar = isinstance(Rp, scalar_types)
        Cp_is_scalar = isinstance(Cp, scalar_types)
        params_are_scalars = [R_is_scalar, Rp_is_scalar, Cp_is_scalar]
        if any(params_are_scalars) and not all(params_are_scalars):
            raise AcausalModelError(
                message=f"Component {self.__class__.__name__} must have R,Rp,Cp all scalars, or all arrays."
            )
        if all(params_are_scalars):
            # declare the scalar symbols for the params
            R = self.declare_symbol(
                ev,
                "R",
                self.name,
                kind=SymKind.param,
                val=R,
                validator=lambda R: R > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have R>0",
            )
            Rp = self.declare_symbol(
                ev,
                "Rp",
                self.name,
                kind=SymKind.param,
                val=Rp,
                validator=lambda Rp: Rp > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Rp>0",
            )
            Cp = self.declare_symbol(
                ev,
                "Cp",
                self.name,
                kind=SymKind.param,
                val=Cp,
                validator=lambda Cp: Cp > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Cp>0",
            )
        else:
            # declare lookup tables for R,Rp, and Cp
            if not len(R) == len(Cp) == len(Cp) == len(SOC_v):
                raise AcausalModelError(
                    message=f"Component {self.__class__.__name__} must have R,Rp,Cp,SOC_v arrays all be same length."
                )

            R = self.declare_1D_lookup_table(
                ev, SOC.s, "SOC_v", SOC_v, "R_v", R, "R_lut"
            )
            Rp = self.declare_1D_lookup_table(
                ev, SOC.s, "SOC_v", SOC_v, "Rp_v", Rp, "Rp_lut"
            )
            Cp = self.declare_1D_lookup_table(
                ev, SOC.s, "SOC_v", SOC_v, "Cp_v", Cp, "Cp_lut"
            )

        self.add_eqs(
            [
                sp.Eq(dSOC.s, self.Ip.s / AH.s / 3600),  # eqn 3
                sp.Eq(
                    self.V.s,
                    OCV_lut.s + R.s * self.Ip.s + Up_.s,
                ),  # eqn 1
                sp.Eq(dUp.s, self.Ip.s / Cp.s - Up_.s / (Rp.s * Cp.s)),  # eqn 2
            ]
        )

        if enable_soc_port:
            soc = self.declare_symbol(ev, "soc", self.name, kind=SymKind.outp)
            self.declare_equation(sp.Eq(soc.s, SOC.s), kind=EqnKind.outp)
        if enable_Up_port:
            Up = self.declare_symbol(ev, "Up", self.name, kind=SymKind.outp)
            self.declare_equation(sp.Eq(Up_.s, Up.s), kind=EqnKind.outp)
        if enable_ocv_port:
            ocv = self.declare_symbol(ev, "ocv", self.name, kind=SymKind.outp)
            self.declare_equation(sp.Eq(ocv.s, OCV_lut.s), kind=EqnKind.outp)


class BLDC(ElecTwoPin):
    """
    Brushless Direct Current Motor (BLDC).
    Combined Motor-Inverter model for a 4 Quadrant BLDC motor. The governing equations:
        1. trq = trq_req_norm * peaktrq_lut(abs(speed))
        2. mech_pwr = trq * speed
        3. if mech_pwr >= 0:
                elec_pwr = mech_pwr / eff
           else:
                elec_pwr = mech_pwr * eff
        4. I = elec_pwr/V
        5. trq = J*alpha
        6. eff = eff_lut(abs(speed),abs(trq))
    optionally:
        6. heat = abs(elec_pwr - mech_pwr)

    This component requires the following lookup tables:
        peaktrq_lut: 1D lookup from speed[0:inf] to trq[0:inf]
        eff_lut: 2D lookup from (speed[0:inf],trq[0:inf]) to eff[0:1]
    where [a:b] means the range of the variable.
    The peaktrq_lut can be prvided as 2 vectors, or as 3 scalar parameters.
    The eff_lut can be provied as 2 vectors and a 2D matrix, or a single scalar.

    Inputs:
        trq_req: causal signal for torque request from external controller.

    From the rotational domain, the component is like a torque source, from the
    electrical domain, the component is like a current source. And optionally, from
    the thermal domain, the component is like a heat source.

    Note on sign convention.
        - positive electrical power flows into the component
        - positive mechanical power flows out of the component

    """

    def __init__(
        self,
        ev,
        name=None,
        J=0.05,
        peaktrq_spd=None,
        peaktrq_trq=None,
        peak_trq=None,
        peak_pwr=None,
        peak_spd=None,
        eff_spd=None,
        eff_trq=None,
        eff_eff=None,
        eff_k=None,
        enable_heat_port=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name, p1="pos", p2="neg")

        # process user provided parameters
        self.peaktrq_spd, self.peaktrq_trq = self._process_peak_trq(
            self.name,
            peaktrq_spd,
            peaktrq_trq,
            peak_trq,
            peak_pwr,
            peak_spd,
        )
        self.eff_k = self._process_eff(eff_spd, eff_trq, eff_eff, eff_k)
        # below is for testing the 2D LUT feature witho
        # eff_spd = cnp.linspace(0, cnp.max(self.peaktrq_spd), 20)
        # eff_trq = cnp.linspace(0, cnp.max(self.peaktrq_trq), 20)
        # eff_eff = cnp.ones((len(eff_spd), len(eff_trq))) * self.eff_k
        # self.eff_k = None

        # declare component ports
        trq, ang, w, alpha = self.declare_rotational_port(
            ev,
            "shaft",
            w_ic=0.0,
            ang_ic=0.0,
        )
        self.port_idx_to_name = {-2: "pos", -3: "neg", 1: "shaft"}

        # create component symbols
        J = self.declare_symbol(
            ev,
            "J",
            self.name,
            kind=SymKind.param,
            val=J,
            validator=lambda J: J > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have J>0",
        )
        trq_lut = self.declare_1D_lookup_table(
            ev,
            sp.Abs(w.s),
            "peaktrq_spd",
            self.peaktrq_spd,
            "peaktrq_trq",
            self.peaktrq_trq,
            "trq_lut",
        )
        if self.eff_k is None:
            eff_lut = self.declare_2D_lookup_table(
                ev,
                sp.Abs(w.s),
                "eff_spd",
                eff_spd,
                sp.Abs(trq.s),
                "eff_trq",
                eff_trq,
                "eff_eff",
                eff_eff,
                "eff_lut",
            )
        else:
            eff_lut = self.declare_symbol(
                ev,
                "eff_k",
                self.name,
                kind=SymKind.param,
                val=self.eff_k,
                validator=lambda eff_k: eff_k > 0.5,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have eff_k>0.5",
            )
        mech_pwr = self.declare_symbol(ev, "mech_pwr", self.name, kind=SymKind.var)
        trq_req_norm = self.declare_symbol(
            ev, "trq_req_norm", self.name, kind=SymKind.inp
        )
        elec_pwr = self.declare_conditional(
            ev,
            mech_pwr.s >= 0,
            mech_pwr.s / eff_lut.s,
            mech_pwr.s * eff_lut.s,
            "elec_pwr_cond",
        )  # eqn 3

        self.add_eqs(
            [
                sp.Eq(
                    0, trq.s + trq_req_norm.s * trq_lut.s - J.s * alpha.s
                ),  # eqn 1 & 5
                sp.Eq(mech_pwr.s, w.s * -trq.s),  # eqn 2
                sp.Eq(self.Ip.s, elec_pwr.s / (self.Vp.s - self.Vn.s)),  # eqn 4
            ]
        )

        if enable_heat_port:
            port_name = "heat"
            T, Q = self.declare_thermal_port(ev, port_name)
            # NOTE: flow vars are negative for flow going out of the component,
            # since heat flows out of the resistor, we
            # need the minus sign on 'Q'.
            self.add_eqs([sp.Eq(-Q.s, sp.Abs(elec_pwr.s - mech_pwr.s))])
            self.port_idx_to_name[2] = port_name

    def _process_peak_trq(
        self,
        name,
        peaktrq_spd,
        peaktrq_trq,
        peak_trq,
        peak_pwr,
        peak_spd,
    ):
        if peak_spd is None:
            peak_spd = 1000
        if peaktrq_spd is None:
            peaktrq_spd = cnp.arange(0, peak_spd, 50)
        if peaktrq_trq is None:
            if peak_pwr is None:
                peak_pwr = 100e3  # Watts
            if peak_trq is None:
                peak_trq = 200  # Nm
            peak_trq_v = cnp.ones_like(peaktrq_spd) * peak_trq  # Nm
            peak_pwrTrq_v = peak_pwr / cnp.maximum(peaktrq_spd, 1.0)  # Nm
            peaktrq_trq = cnp.minimum(peak_trq_v, peak_pwrTrq_v)

        if peaktrq_trq[-1] != 0.0:
            # zero torque at peak speed
            peaktrq_spd = cnp.append(peaktrq_spd, peaktrq_spd[-1] * 1.02)
            peaktrq_trq = cnp.append(peaktrq_trq, 0.0)

        if len(peaktrq_spd) != len(peaktrq_trq):
            raise ValueError(
                f"Component BLDC {self.name} peaktrq_spd and peaktrq_trq must be same length."
            )

        return peaktrq_spd, peaktrq_trq

    def _process_eff(self, eff_spd, eff_trq, eff_eff, eff_k):
        eff_params = [eff_spd, eff_trq, eff_eff]
        if any(eff_params) and not all(eff_params):
            raise ValueError(
                f"Component BLDC {self.name} eff_spd, eff_trq and eff_eff must be all defined, or all None."
            )
        if None in eff_params:
            if eff_k is None:
                eff_k = 0.9
        else:
            eff_k = None

        return eff_k


class IdealDiode(ElecTwoPin):
    """
    FIXME: This component behaves the same regardless of whether using the naive or modelica equations.
        see test_ideal_diode(). with voltage applied one way (dont know if forward or reverse bias), it
        doesn't have the behavior expected. with voltage the other way, the simulation seems to hang,
        maybe poor handling of zeno (see WC-420)?
    Ideal diode in electrical domain.
    Similar to the PWL model here: https://en.wikipedia.org/wiki/Diode_modelling#Diode_with_voltage_source_and_current-limiting_resistor
    The characteristic equations are:
        1.  if V >= Vknee:
                R = Ron
            else:
                R = Roff
        2. I = V / R

    Args:
        Vknee (number):
            Knee voltage in volts.
        Ron (number):
            Resistance in forward bias.
        Roff (number):
            Resistance in reverse bias.
    """

    def __init__(self, ev, name=None, Vknee=0.7, Ron=1e-6, Roff=1e9):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        Vknee = self.declare_symbol(
            ev,
            "Vknee",
            self.name,
            kind=SymKind.param,
            val=Vknee,
            validator=lambda Vknee: Vknee > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Vknee>0",
        )
        Ron = self.declare_symbol(
            ev,
            "Ron",
            self.name,
            kind=SymKind.param,
            val=Ron,
            validator=lambda Ron: Ron > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Ron>0",
        )
        Roff = self.declare_symbol(
            ev,
            "Roff",
            self.name,
            kind=SymKind.param,
            val=Roff,
            validator=lambda Roff: Roff > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Roff>0",
        )
        # both implememtations below are equivalent I think. Not sure why Modelica does it that way.
        # the naive attempt
        # R = self.declare_conditional(ev, self.V.s >= Vknee.s, Ron.s, Roff.s, "R_cond")
        # self.add_eqs([sp.Eq(self.V.s, self.Ip.s * R.s)])

        # copy modelica attempt. the equations from the modelica component:
        #   v = p.v - n.v # appears in ElecTwoPin
        #   0 = p.i + n.i # appears in ElecTwoPin
        #   i = p.i # we just use Ip.s, which is the equivalent to p.i in modelica
        #   v = (s*unitCurrent)*(if s < 0 then 1 else Ron) + Vknee
        #   i = (s*unitVoltage)*(if s < 0 then Goff else 1) + Goff*Vknee
        # the implementation in collimator:
        s = self.declare_symbol(ev, "s", self.name, kind=SymKind.var)
        Vcond = self.declare_conditional(ev, s.s < 0, 1.0, Ron.s, "Vcond")
        Icond = self.declare_conditional(ev, s.s < 0, 1.0 / Roff.s, 1.0, "Icond")
        self.add_eqs(
            [
                sp.Eq(self.V.s, s.s * Vcond.s + Vknee.s),
                sp.Eq(self.Ip.s, s.s * Icond.s + Vknee.s / Roff.s),
            ]
        )


class Diode(ElecTwoPin):
    """
    FIXME: This compoenent behaves as expected with constant voltage source placingin it
        in either forward or reverse bias. However, with sinusoidal voltage source, causing
        bias mode switching, the simulation hangs, not sure why.
    Shockley Diode with parallel resistance.
    https://en.wikipedia.org/wiki/Shockley_diode_equation

    Args:
        Vknee (number):
            Knee voltage in volts.
        Ron (number):
            Resistance in forward bias.
        Roff (number):
            Resistance in reverse bias.
    """

    def __init__(self, ev, name=None, Ids=1e-6, Rp=1e8, Vt=0.04):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        Ids = self.declare_symbol(
            ev,
            "Ids",
            self.name,
            kind=SymKind.param,
            val=Ids,
            validator=lambda Ids: Ids > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Ids>0",
        )
        Rp = self.declare_symbol(
            ev,
            "Rp",
            self.name,
            kind=SymKind.param,
            val=Rp,
            validator=lambda Rp: Rp > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Rp>0",
        )
        Vt = self.declare_symbol(
            ev,
            "Vt",
            self.name,
            kind=SymKind.param,
            val=Vt,
            validator=lambda Vt: Vt > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Vt>0",
        )
        id_ = self.declare_symbol(ev, "id", self.name, kind=SymKind.var)
        self.add_eqs(
            [
                sp.Eq(id_.s, Ids.s * (sp.exp(self.V.s / Vt.s) - 1.0)),
                sp.Eq(self.Ip.s, id_.s + self.V.s / Rp.s),
            ]
        )


class Capacitor(ElecTwoPin):
    """
    Ideal capacitor in electrical domain. The characteristic equation is:
    Derivative(v(t)) = i(t)*C, where C is the capacitance in Farads.

    Args:
        C (number):
            Capacitance in Farads.
        initial_voltage (number):
            Initial voltage of the capacitor.
    """

    def __init__(
        self, ev, name=None, C=1.0, initial_voltage=0.0, initial_voltage_fixed=False
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev, self.name, V_ic=initial_voltage, V_ic_fixed=initial_voltage_fixed
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
        dV = self.declare_symbol(
            ev,
            "dV",
            self.name,
            kind=SymKind.var,
            int_sym=self.V,
            ic=0.0,
        )
        self.V.der_sym = dV
        self.add_eqs([sp.Eq(self.Ip.s, C.s * dV.s)])


class CurrentSensor(ElecTwoPin):
    """
    Ideal current sensor in electrical domain.
    """

    def __init__(self, ev, name=None):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        i = self.declare_symbol(ev, "i", self.name, kind=SymKind.outp)
        # the current output is the current at either pin. chose Ip.s at pin 'p'
        # for sign convention. Use declare_equation() so we can set the EqnKind
        # as 'output'.
        self.declare_equation(sp.Eq(i.s, self.Ip.s), kind=EqnKind.outp)
        # current sensor has no effect on the system. it is inline, but has not
        # voltage across it, this is the 'ideal' part.
        self.declare_equation(sp.Eq(self.Vp.s, self.Vn.s))


class CurrentSource(ElecTwoPin):
    """
    Ideal current source in electrical domain.
    """

    def __init__(
        self,
        ev,
        name=None,
        i=1.0,  # noqa
        enable_current_port=False,
        **kwargs,
    ):
        # raise NotImplementedError("Electrical CurrentSource not implemented.")
        # Although this appears to be implemented, it has never been successful
        # in a test.

        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        if enable_current_port:
            i = self.declare_symbol(ev, "i", self.name, kind=SymKind.inp, val=i)
        else:
            i = self.declare_symbol(ev, "i", self.name, kind=SymKind.param, val=i)

        self.add_eqs([sp.Eq(self.Ip.s, i.s)])


class Ground(ComponentBase):
    """
    *ground* reference in electrical domain.

    Note: the only 'single' pin component in electrical domain.
    """

    def __init__(self, ev, name=None):
        super().__init__()
        self.name = self.__class__.__name__ if name is None else name
        v, i = self.declare_electrical_port(ev, "p")
        self.add_eqs([sp.Eq(0, v.s)])
        self.port_idx_to_name = {-1: "p"}


class IdealMotor(ComponentBase):
    """
    Ideal 4-quadrant motor with inertia, but no visous losses. The governing equations:
        trq = Kt*I
        backEMF = Ke*w
        V - backEMF = I*R
        heatflow = I*I*R

    Note on efficiency. In this simple case, power loss is I*I*R,
    but this is not explicitly computed, unless the heat_port is enabled.
    Because there are no electro-magnetic losses, this means mech power and
    BackEMF power are equal. Some algebra shows that Kt=Ke; therfore the
    component has only K as a parameter.

    Note on sign convention. I1 is used everywhere, to maintain consistent
    sign of the current in the equations.
    BackEFM and rotation velocity always have the same sign.
    Current and Torque always have the same sign.

    Args:
        R (number):
            Armateur resistance in Ohms.
        K (number):
            Torque constant with unit Nm/Amp, and Back ElectrocMotive Force
            (backEMF) constant with units Volts/(rad/s).
        L (number):
            Armateur inductance in Henry.
        J (number):
            Armateur inertia in Kg*m^2.
        enable_heat_port (bool):
            When true, exposes a thermal port which acts as a heatflow source.
        initial_angle (number):
            Initial angle of the armateur.
        initial_velocity (number):
            Initial velocity of the armateur.
    """

    def __init__(
        self,
        ev,
        name=None,
        R=1.0,
        K=1.0,
        L=1e-6,
        J=1.0,
        initial_velocity=0.0,
        initial_velocity_fixed=False,
        initial_angle=0.0,
        initial_angle_fixed=False,
        initial_current=0.0,
        initial_current_fixed=False,
        enable_heat_port=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()

        Vp, Ip = self.declare_electrical_port(
            ev,
            "pos",
            I_ic=initial_current,
            I_ic_fixed=initial_current_fixed,
        )
        Vn, In = self.declare_electrical_port(ev, "neg")
        trq, ang, w, alpha = self.declare_rotational_port(
            ev,
            "shaft",
            w_ic=initial_velocity,
            w_ic_fixed=initial_velocity_fixed,
            ang_ic=initial_angle,
            ang_ic_fixed=initial_angle_fixed,
        )
        self.port_idx_to_name = {-1: "pos", -2: "neg", 1: "shaft"}

        R = self.declare_symbol(
            ev,
            "R",
            self.name,
            kind=SymKind.param,
            val=R,
            validator=lambda R: R > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have R>0",
        )
        K = self.declare_symbol(
            ev,
            "K",
            self.name,
            kind=SymKind.param,
            val=K,
            validator=lambda K: K > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have K>0",
        )
        L = self.declare_symbol(
            ev,
            "L",
            self.name,
            kind=SymKind.param,
            val=L,
            validator=lambda L: L > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have L>0",
        )
        J = self.declare_symbol(
            ev,
            "J",
            self.name,
            kind=SymKind.param,
            val=J,
            validator=lambda J: J > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have J>0",
        )
        backEMF = self.declare_symbol(
            ev, "backEMF", self.name, kind=SymKind.var, ic=0.0
        )
        dI = self.declare_symbol(
            ev, "dI", self.name, kind=SymKind.var, int_sym=Ip, ic=0.0
        )

        self.add_eqs(
            [
                # NOTE: flow vars are negative for flow going out of the component,
                # since torque flows out of the motor when I1 is positive, we
                # need the minus sign on 't'.
                sp.Eq(-trq.s, K.s * Ip.s - J.s * alpha.s),  # torque balance
                sp.Eq(backEMF.s, K.s * w.s),
                sp.Eq(
                    Vp.s - Vn.s - backEMF.s, Ip.s * R.s + dI.s * L.s
                ),  # voltage balance
                sp.Eq(0, Ip.s + In.s),
            ]
        )

        if enable_heat_port:
            port_name = "heat"
            T, Q = self.declare_thermal_port(ev, port_name)
            # NOTE: flow vars are negative for flow going out of the component,
            # since heat flows out of the resistor, we
            # need the minus sign on 'Q'.
            self.add_eqs([sp.Eq(-Q.s, Ip.s * Ip.s * R.s)])
            self.port_idx_to_name[2] = port_name


class Inductor(ElecTwoPin):
    """
    Ideal inductor in electrical domain. The characteristic equation is:
    Derivative(i(t))*L = v(t), where L is the inductance in Henry.

    Args:
        L (number):
            Inductance in Henry.
        initial_current (number):
            Initial current flowing through inductor.
    """

    def __init__(
        self, ev, name=None, L=1.0, initial_current=0.0, initial_current_fixed=False
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev, self.name, I_ic=initial_current, I_ic_fixed=initial_current_fixed
        )
        L = self.declare_symbol(
            ev,
            "L",
            self.name,
            kind=SymKind.param,
            val=L,
            validator=lambda L: L > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have L>0",
        )
        dI = self.declare_symbol(
            ev, "dI", self.name, kind=SymKind.var, int_sym=self.Ip, ic=0.0
        )
        self.Ip.der_sym = dI
        self.add_eqs([sp.Eq(self.V.s, L.s * dI.s)])


class Resistor(ElecTwoPin):
    """
    Ideal resistor in electrical domain. The characteristic equation is:
    v(t) = i(t)*R, where R is the resistance in Ohms.

    When heat port is enabled, the thermal equation is:
    heatflow(t) = i(t)*i(t)*R.

    Args:
        R (number):
            Electrical resistance in Ohms.
        enable_heat_port (bool):
            When true, exposes a thermal port which acts as a heatflow source.
    """

    def __init__(self, ev, name=None, R=1.0, enable_heat_port=False):
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
        self.add_eqs([sp.Eq(self.V.s, self.Ip.s * R.s)])

        if enable_heat_port:
            port_name = "heat"
            T, Q = self.declare_thermal_port(ev, port_name)
            # NOTE: flow vars are negative for flow going out of the component,
            # since heat flows out of the resistor, we
            # need the minus sign on 'Q'.
            self.add_eqs([sp.Eq(-Q.s, self.Ip.s * self.Ip.s * R.s)])
            self.port_idx_to_name[2] = port_name


class VoltageSensor(ElecTwoPin):
    """
    Ideal voltage sensor in electrical domain.
    """

    def __init__(self, ev, name=None):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        v = self.declare_symbol(ev, "v", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(v.s, self.V.s), kind=EqnKind.outp)
        # ensure the sensor does not contribute to flow balance equations.
        self.add_eqs(
            [
                sp.Eq(self.Ip.s, 0),
                sp.Eq(self.In.s, 0),
            ]
        )


class VoltageSource(ElecTwoPin):
    """
    Ideal voltage source in electrical domain.

    Args:
        v (number):
            Voltage value when enable_voltage_port=False.
        enable_voltage_port (bool):
            When true, the voltage value is from a input signal. When false, voltage
            value is from 'v'.
    """

    def __init__(self, ev, name=None, v=1.0, enable_voltage_port=False, **kwargs):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        if enable_voltage_port:
            v = self.declare_symbol(ev, "v", self.name, kind=SymKind.inp)
        else:
            v = self.declare_symbol(ev, "v", self.name, kind=SymKind.param, val=v)

        self.add_eqs([sp.Eq(self.V.s, v.s)])
