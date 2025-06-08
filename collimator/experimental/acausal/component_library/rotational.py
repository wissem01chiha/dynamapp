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
import warnings

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")

"""
1D mechanical rotational components similar to Modelica Standard Library.

flow variable:Units = torque:Newton*meters
potential variable:Units = angular_velocity:radians/second
"""


class RotationalOnePort(ComponentBase):
    """Partial component class for a rotational component with one flange."""

    def __init__(
        self,
        ev,
        name,
        w_ic=0.0,
        w_ic_fixed=False,
        ang_ic=0.0,
        ang_ic_fixed=False,
        p="flange",
    ):
        super().__init__()
        self.t, self.ang, self.w, self.alpha = self.declare_rotational_port(
            ev,
            p,
            w_ic=w_ic,
            w_ic_fixed=w_ic_fixed,
            ang_ic=ang_ic,
            ang_ic_fixed=ang_ic_fixed,
        )
        self.port_idx_to_name = {-1: p}


class RotationalTwoPort(ComponentBase):
    """Partial component class for an rotational component with two
    flanges that can rotate relative to each other.
    """

    def __init__(
        self,
        ev,
        name,
        ang1_ic=0.0,
        ang1_ic_fixed=False,
        w1_ic=0.0,
        w1_ic_fixed=False,
        ang2_ic=0.0,
        ang2_ic_fixed=False,
        w2_ic=0.0,
        w2_ic_fixed=False,
        p1="flange_a",
        p2="flange_b",
        include_torque_equality=True,
    ):
        super().__init__()
        self.t1, self.ang1, self.w1, self.alpha1 = self.declare_rotational_port(
            ev,
            p1,
            w_ic=w1_ic,
            w_ic_fixed=w1_ic_fixed,
            ang_ic=ang1_ic,
            ang_ic_fixed=ang1_ic_fixed,
        )
        self.t2, self.ang2, self.w2, self.alpha2 = self.declare_rotational_port(
            ev,
            p2,
            w_ic=w2_ic,
            w_ic_fixed=w2_ic_fixed,
            ang_ic=ang2_ic,
            ang_ic_fixed=ang2_ic_fixed,
        )
        if include_torque_equality:
            self.add_eqs([sp.Eq(0, self.t1.s + self.t2.s)])

        self.port_idx_to_name = {-1: p1, 1: p2}


class BasicEngine(RotationalOnePort):
    """
    Internal Combustion Engine modeled as a torque source.
    The produced toqrue is a function of the 'throttle' input,
    and the peak toqrue curve, which is a lookup table as a function
    of speed.

    Phenomena neglected are listed below, maybe these will be aded in the future.
    - idling
    - inertia
    - friction
    - fuel consumption
    - pumping losses
    - heat rejection

    Args:
        peak_trq_w: list(float):
            peak toqrue curve speed break points
        peak_trq_t: list(float):
            peak toqrue curve torque data points

    """

    def __init__(
        self,
        ev,
        name=None,
        peak_trq_w=[0.0, 99, 100, 500, 600],
        peak_trq_t=[0.0, 0.0, 100, 100, 0.0],
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        thr = self.declare_symbol(ev, "thr", self.name, kind=SymKind.inp)

        # create component specific lookup table function symbols
        peak_trq_lut = self.declare_1D_lookup_table(
            ev,
            self.w.s,
            "peak_trq_w",
            peak_trq_w,
            "peak_trq_t",
            peak_trq_t,
            lut_name="peak_trq_f",
        )
        # negative in front of t.s because the RHS is torque going out
        # of the component.
        self.add_eqs([sp.Eq(-self.t.s, thr.s * peak_trq_lut.s)])


class Damper(RotationalTwoPort):
    """
    Ideal damper in rotational domain. The characteristic equation is:
    torque(t) = D*(w1(t) - w2(t)), where D is the damping coefficient in Nm/(rad/s).

    Agrs:
        D (number):
            The damping coefficient.
        initial_velocity_A (number);
            initial velocity of flange_a.
        initial_angle_A (number);
            initial angle of flange_a.
        initial_velocity_B (number);
            initial velocity of flange_b.
        initial_angle_B (number);
            initial angle of flange_b.
    """

    def __init__(
        self,
        ev,
        name=None,
        D=1.0,
        initial_velocity_A=0.0,
        initial_velocity_A_fixed=False,
        initial_angle_A=0.0,
        initial_angle_A_fixed=False,
        initial_velocity_B=0.0,
        initial_velocity_B_fixed=False,
        initial_angle_B=0.0,
        initial_angle_B_fixed=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev,
            self.name,
            w1_ic=initial_velocity_A,
            w1_ic_fixed=initial_velocity_A_fixed,
            ang1_ic=initial_angle_A,
            ang1_ic_fixed=initial_angle_A_fixed,
            w2_ic=initial_velocity_B,
            w2_ic_fixed=initial_velocity_B_fixed,
            ang2_ic=initial_angle_B,
            ang2_ic_fixed=initial_angle_B_fixed,
        )

        d = self.declare_symbol(
            ev,
            "D",
            self.name,
            kind=SymKind.param,
            val=D,
            validator=lambda D: D > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have D>0",
        )
        self.add_eqs([sp.Eq(self.t1.s, d.s * (self.w1.s - self.w2.s))])


class FixedAngle(RotationalOnePort):
    """
    Rigid(non-moving) reference in mechanical rotational domain.

    Agrs:
        initial_angle (number);
            initial angle of flange.
    """

    def __init__(self, ev, name=None, initial_angle=0.0):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev,
            self.name,
            w_ic=0,
            ang_ic=initial_angle,
            ang_ic_fixed=True,
        )
        # not ideal to create a dummy parameter symbol in order to set the x symbol
        # to this fixed value, but this works for now. using 'initial_position' in the
        # IC equation below was tried, this results in equations processing errors. not
        # that these cannot be over come, but it's not a priority when the dummy param
        # solution works.
        self.add_eqs([sp.Eq(0, self.alpha.s), sp.Eq(0, self.w.s)])
        ang_ic_param = self.declare_symbol(
            ev, "ang_ic_param", self.name, kind=SymKind.param, val=initial_angle
        )
        # IC equation
        self.add_eqs([sp.Eq(ang_ic_param.s, self.ang.s)])

        # this sucks. need this here because not all one port blocks
        # use the 'input' of the block.
        self.port_idx_to_name = {1: "flange"}


class Friction(RotationalTwoPort):
    """
    Coulomb, Viscous and Stribeck friction in rotational domain. Coulomb friction is always
    modeled, while Viscous and Stribeck can be optionally enabled as detailed below.

    In it's simplest form, Tbrk=None and C=None, the characteristic equation is:
        t = Tc*tanh(w/(Wbrk/10))

    With only Tbrk=None, the viscous term is included, and the characteristic equation is:
        t = Tc*tanh(w/(Wbrk/10)) +
            C*w

    With all friction sources enabled, the characteristic equation is:
        t = Tc*tanh(w/(Wbrk/10)) +
            C*w +
            sqrt(2*e)*(Tbrk - Tc)*exp(-(w/(Wbrk*sqrt(2)))**2)*(w/(Wbrk*sqrt(2)))

    where:
        t:w are relative torque and velocity across the component ports
        Tbrk is break away friction
        Tc is Coulomb friction
        Wbrk is the break away friction velocity threshold
        C is the viscous friction coefficient

    Agrs:
        <blah> (number):
            The <blah> coefficient.
    """

    def __init__(self, ev, name=None, Tc=1.0, Wbrk=0.1, C=None, Tbrk=None):
        self.name = self.__class__.__name__ if name is None else name
        if Tc is None:
            Tc = 1.0
        if Wbrk is None:
            Wbrk = 0.1
        if Tbrk is not None:
            warnings.warn(
                f"Component {self.__class__.__name__} {self.name} Stribeck friction not implemented. Tbrk ignored."
            )

        super().__init__(ev, self.name)

        Wbrk = self.declare_symbol(
            ev,
            "Wbrk",
            self.name,
            kind=SymKind.param,
            val=Wbrk,
            validator=lambda Wbrk: Wbrk > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Wbrk>0",
        )
        Tc = self.declare_symbol(
            ev,
            "Tc",
            self.name,
            kind=SymKind.param,
            val=Tc,
            validator=lambda Tc: Tc > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have Tc>0",
        )
        frc_coul = self.declare_symbol(ev, "frc_coul", self.name, kind=SymKind.var)
        Vrel = self.declare_symbol(ev, "Vrel", self.name, kind=SymKind.var)
        self.add_eqs(
            [
                sp.Eq(Vrel.s, self.w1.s - self.w2.s),
                sp.Eq(frc_coul.s, Tc.s * sp.tanh(Vrel.s / (Wbrk.s / 10.0))),
            ]
        )
        if C is None:
            # only Coulomb friction
            self.add_eqs([sp.Eq(self.t1.s, frc_coul.s)])
        else:
            # include Viscous friction
            C = self.declare_symbol(
                ev,
                "C",
                self.name,
                kind=SymKind.param,
                val=C,
                validator=lambda C: C > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have C>0",
            )
            self.add_eqs([sp.Eq(self.t1.s, frc_coul.s + C.s * Vrel.s)])


class Gear(RotationalTwoPort):
    """
    Gear ratio with efficiency map based on speed and torque.
    The characteristic equations are:
    w1(t) = r * w2(t)
    eff(t) = LUT2D(t1(t),w1(t))
    pwr1(t) = t1(t)*w1(t)
    if pwr1(t)>=0: power is flowing from flang_1 to flange_2
        0 = r * t1(t) * eff + t2(t)
    else:
        0 = r * t1(t) / eff + t2(t)

    Note: the eff equation implies that a four quadrant (trq spd plane) efficiency
    map has been provided by the user. We could allow user to provide only one
    quadrant, but the eff equation would then need to be LUT2D(abs(t1(t)),abs(w1(t))),
    so we would need a bool arg is_one_quad_eff_map for the user to indicate this.
    Coincidently, having abs() in the equation also means zero corssings are present.
    """

    def __init__(
        self,
        ev,
        name=None,
        r=1.0,
        spd_pts=[-1e6, 1e6],
        trq_pts=[-1e6, 1e6],
        eff=[[0.98, 0.98], [0.98, 0.98]],
    ):
        self.name = self.__class__.__name__ if name is None else name
        # NOTE: 'include_torque_equality=False' because 0=t1+t1 is not valid for a gear.
        super().__init__(ev, self.name, include_torque_equality=False)

        r = self.declare_symbol(
            ev,
            "r",
            self.name,
            kind=SymKind.param,
            val=r,
            validator=lambda r: r > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have r>0",
        )
        eff = self.declare_2D_lookup_table(
            ev,
            self.w1.s,
            "spd_pts",
            spd_pts,
            self.t1.s,
            "trq_pts",
            trq_pts,
            "eff_pts",
            eff,
            "eff_map",
        )
        pwr1 = self.declare_symbol(ev, "pwr1", self.name, kind=SymKind.var)
        trq_expr = self.declare_conditional(
            ev,
            pwr1.s >= 0,  # if
            r.s * self.t1.s * eff.s + self.t2.s,  # then
            r.s * self.t1.s / eff.s + self.t2.s,  # else
        )
        self.add_eqs(
            [
                sp.Eq(self.w1.s, r.s * self.w2.s),
                sp.Eq(pwr1.s, self.w1.s * self.t1.s),
                sp.Eq(0, trq_expr.s),
            ]
        )


class IdealGear(RotationalTwoPort):
    """
    Ideal gear ratio. The characteristic equations are:
    w1(t) = r * w2(t)
    0 = r * t1(t) + t2(t)

    Note: for r>1.0, IdealGear is a 'reducer'.
    """

    def __init__(self, ev, name=None, r=1.0):
        self.name = self.__class__.__name__ if name is None else name
        # NOTE: 'include_torque_equality=False' because 0=t1+t1 is not valid for a gear.
        super().__init__(ev, self.name, include_torque_equality=False)

        r = self.declare_symbol(
            ev,
            "r",
            self.name,
            kind=SymKind.param,
            val=r,
            validator=lambda r: r > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have r>0",
        )
        self.add_eqs(
            [
                sp.Eq(self.w1.s, r.s * self.w2.s),
                sp.Eq(0, r.s * self.t1.s + self.t2.s),
            ]
        )


class IdealPlanetary(ComponentBase):
    """
    Ideal planetary gear.
    Ratio parameter 'r' is the ratio of ring teeth to sun teeth.
    This fixes the number of planet teeth due to np = (nr - ns) / 2.

    The characteristic equations are:
    ring_t(t) = r*sun_t(t).
    carrier_t(t) = -(1-r)*sun_t(t).
    (1+r)*carrier_w = sun_w(t) + r*ring_w(t).
    """

    def __init__(self, ev, name=None, r=2.0):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()

        tc, angc, wc, alphac = self.declare_rotational_port(ev, "carrier")
        ts, angs, ws, alphas = self.declare_rotational_port(ev, "sun")
        tr, angr, wr, alphar = self.declare_rotational_port(ev, "ring")
        r = self.declare_symbol(
            ev,
            "r",
            self.name,
            kind=SymKind.param,
            val=r,
            validator=lambda r: r > 1.5,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have r>1.5",
        )
        self.add_eqs(
            [
                sp.Eq((1 + r.s) * wc.s, ws.s + r.s * wr.s),
                sp.Eq(tr.s, r.s * ts.s),
                sp.Eq(tc.s, -(1 + r.s) * ts.s),
            ]
        )

        self.port_idx_to_name = {-1: "carrier", -2: "sun", 1: "ring"}


class IdealWheel(ComponentBase):
    """
    Ideal wheel between rotational and translational.
    velocity = radius*angular_velocity
    force = torque/radius
    """

    def __init__(self, ev, name=None, r=1.0):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()

        f, x, v, a = self.declare_translational_port(ev, "flange")
        t, ang, w, alpha = self.declare_rotational_port(ev, "shaft")

        r = self.declare_symbol(
            ev,
            "r",
            self.name,
            kind=SymKind.param,
            val=r,
            validator=lambda r: r > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have r>0",
        )
        self.add_eqs(
            [
                sp.Eq(v.s, r.s * w.s),
                sp.Eq(-f.s, t.s / r.s),
            ]
        )
        self.port_idx_to_name = {-1: "shaft", 1: "flange"}


class Inertia(RotationalOnePort):
    """
    Ideal inertia in rotational domain. The characteristic equation is:
    torque(t) = I*alpha(t), where alpha=derivative(w(t)) and I is moment of
    inertia in Nm/(rad/s^s).
    Agrs:
        I (number):
            The moment of inertia.
        initial_velocity (number);
            initial velocity of flange.
        initial_angle (number);
            initial angle of flange.
    """

    def __init__(
        self,
        ev,
        name=None,
        I=1.0,  # noqa
        initial_velocity=0.0,
        initial_velocity_fixed=False,
        initial_angle=0.0,
        initial_angle_fixed=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev,
            self.name,
            w_ic=initial_velocity,
            w_ic_fixed=initial_velocity_fixed,
            ang_ic=initial_angle,
            ang_ic_fixed=initial_angle_fixed,
        )

        I = self.declare_symbol(  # noqa
            ev,
            "I",
            self.name,
            kind=SymKind.param,
            val=I,
            validator=lambda I: I > 0.0,  # noqa
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have I>0",
        )
        self.add_eqs([sp.Eq(self.t.s, I.s * self.alpha.s)])


class MotionSensor(ComponentBase):
    """
    Ideal speed sensor in the rotational domain.

    Agrs:
        enable_flange_b(bool):
            When flange_b enabled, measures between flange_a and flange_b.
            When flange_b disbaled, measures the absolute speed.
    """

    def __init__(
        self,
        ev,
        name=None,
        enable_flange_b=True,
        enable_angle_port=False,
        enable_velocity_port=True,
        enable_acceleration_port=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        if not (enable_angle_port or enable_velocity_port or enable_acceleration_port):
            raise ValueError(
                f"SpeedSensor {self.name} must have one causal outport enabled."
            )

        super().__init__()
        t1, ang1, w1, alpha1 = self.declare_rotational_port(ev, "flange_a")
        self.port_idx_to_name = {-1: "flange_a"}

        if enable_angle_port:
            ang_rel = self.declare_symbol(ev, "ang_rel", self.name, kind=SymKind.outp)
        if enable_velocity_port:
            w_rel = self.declare_symbol(ev, "w_rel", self.name, kind=SymKind.outp)
        if enable_acceleration_port:
            alpha_rel = self.declare_symbol(
                ev, "alpha_rel", self.name, kind=SymKind.outp
            )

        if enable_flange_b:
            t2, ang2, w2, alpha2 = self.declare_rotational_port(ev, "flange_b")
            self.port_idx_to_name[1] = "flange_b"
            self.add_eqs([sp.Eq(t1.s, 0), sp.Eq(t2.s, 0)])
            if enable_angle_port:
                self.declare_equation(
                    sp.Eq(ang_rel.s, ang1.s - ang2.s), kind=EqnKind.outp
                )
            if enable_velocity_port:
                self.declare_equation(sp.Eq(w_rel.s, w1.s - w2.s), kind=EqnKind.outp)
            if enable_acceleration_port:
                self.declare_equation(
                    sp.Eq(alpha_rel.s, alpha1.s - alpha2.s), kind=EqnKind.outp
                )
        else:
            self.add_eqs([sp.Eq(t1.s, 0)])
            if enable_angle_port:
                self.declare_equation(sp.Eq(ang_rel.s, ang1.s), kind=EqnKind.outp)
            if enable_velocity_port:
                self.declare_equation(sp.Eq(w_rel.s, w1.s), kind=EqnKind.outp)
            if enable_acceleration_port:
                self.declare_equation(sp.Eq(alpha_rel.s, alpha1.s), kind=EqnKind.outp)


class SpeedSource(ComponentBase):
    """
    Ideal speed source in rotational domain.

    Args:
        w_ref (number):
            Speed value when enable_speed_port=False.
        enable_speed_port (bool):
            When true, the speed value is from a input signal. When false, speed
            value is from w_ref.
        enable_flange_b (bool):
            When flange_b enabled, applies speed between flange_a and flange_b.
            When flange_b disbaled, applies absolute speed to flange_a.
    """

    def __init__(
        self,
        ev,
        name=None,
        w_ref=0.0,
        enable_speed_port=False,
        enable_flange_b=True,
    ):
        raise NotImplementedError("Rotational SpeedSource not implemented.")


class Spring(RotationalTwoPort):
    """
    Ideal spring in rotational domain. The characteristic equation is:
    torque(t) = K*(ang1(t) - ang2(t)), where K is the spring constant in Nm/rad.

    Agrs:
        K (number):
            The stiffness of the spring.
        initial_velocity_A (number);
            initial velocity of flange_a.
        initial_angle_A (number);
            initial angle of flange_a.
        initial_velocity_B (number);
            initial velocity of flange_b.
        initial_angle_B (number);
            initial angle of flange_b.

    """

    def __init__(
        self,
        ev,
        name=None,
        K=1.0,
        initial_velocity_A=0.0,
        initial_velocity_A_fixed=False,
        initial_angle_A=0.0,
        initial_angle_A_fixed=False,
        initial_velocity_B=0.0,
        initial_velocity_B_fixed=False,
        initial_angle_B=0.0,
        initial_angle_B_fixed=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(
            ev,
            self.name,
            w1_ic=initial_velocity_A,
            w1_ic_fixed=initial_velocity_A_fixed,
            ang1_ic=initial_angle_A,
            ang1_ic_fixed=initial_angle_A_fixed,
            w2_ic=initial_velocity_B,
            w2_ic_fixed=initial_velocity_B_fixed,
            ang2_ic=initial_angle_B,
            ang2_ic_fixed=initial_angle_B_fixed,
        )

        k = self.declare_symbol(
            ev,
            "K",
            self.name,
            kind=SymKind.param,
            val=K,
            validator=lambda K: K > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have K>0",
        )
        self.add_eqs([sp.Eq(self.t1.s, k.s * (self.ang1.s - self.ang2.s))])


class TorqueSensor(RotationalTwoPort):
    """
    Ideal torque sensor in rotational domain.
    Measures torque between flange_a and flange_b.
    """

    def __init__(
        self,
        ev,
        name=None,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        tau = self.declare_symbol(ev, "tau", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(tau.s, self.t1.s), kind=EqnKind.outp)
        self.add_eqs(
            [
                sp.Eq(self.ang1.s, self.ang2.s),
                sp.Eq(self.w1.s, self.w2.s),
                sp.Eq(self.alpha1.s, self.alpha2.s),
            ]
        )


class TorqueSource(ComponentBase):
    """
    Ideal torque source in rotational domain.

    Args:
        tau (number):
            Torque value when enable_torque_port=False.
        enable_torque_port (bool):
            When true, the torque value is from a input signal. When false,
            torque value is from 'tau'.
        enable_flange_b (bool):
            When flange_b enabled, applies torque between flange_a and flange_b.
            when flange_b disbaled, applies absolute torque to flange_a.
    """

    def __init__(
        self,
        ev,
        name=None,
        tau=0.0,
        enable_torque_port=False,
        enable_flange_b=True,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        t1, ang1, w1, alpha1 = self.declare_rotational_port(ev, "flange_a")
        self.port_idx_to_name = {-1: "flange_a"}

        if enable_torque_port:
            tau = self.declare_symbol(ev, "tau", self.name, kind=SymKind.inp)
        else:
            tau = self.declare_symbol(ev, "tau", self.name, kind=SymKind.param, val=tau)

        self.add_eqs([sp.Eq(tau.s, -t1.s)])
        if enable_flange_b:
            t2, ang2, w2, alpha2 = self.declare_rotational_port(ev, "flange_b")
            self.port_idx_to_name[1] = "flange_b"
            self.add_eqs([sp.Eq(tau.s, t2.s)])
