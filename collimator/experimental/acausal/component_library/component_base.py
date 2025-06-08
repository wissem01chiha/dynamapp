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

from typing import Tuple, List, Dict, Set, TYPE_CHECKING, Callable

from .base import EqnEnv, Domain, SymKind, Sym, EqnKind, Eqn
from .fluid_media import Fluid
from ..error import AcausalModelError
from collimator.backend import numpy_api as cnp
from collimator.backend.typing import ArrayLike

from collimator.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")


class PortBase:
    """Base class for the port of acausal components.
    The flow and pot attributes are used by DiagramProcessing when constructing
    node conserving and constraint equations, to identify these relative to other
    component/port symbols.

    Attributes:
        domain (Domain):
            The physical domain of the port.
        flow (Sym):
            The 'flow' symbol associated with the port.
        pot (Sym):
            The 'potential' symbol associated with the port.
    """

    def validate(self):
        assert isinstance(self.domain, Domain)
        assert self.flow.kind == SymKind.flow
        assert self.pot.kind == SymKind.pot

    def __repr__(self):
        return str(
            self.__class__.__name__
            + " pot:"
            + str(self.pot)
            + " flow:"
            + str(self.flow)
        )


class ElecPort(PortBase):
    """Class for acausal port in the electrical domain."""

    def __init__(self, name, V_sym=None, I_sym=None):
        self.domain = Domain.electrical
        self.name = name
        self.pot = V_sym
        self.flow = I_sym
        self.validate()


class FluidPort(PortBase):
    """Class for acausal port in the fluid_i domain.
    The fluid attribute is assigned after the constructor. the reason is
    that the constructor is called at component creation. However, the
    fluid at the port is not known until DiagramProcessing has performed
    some graph analysis to know which FluidPropertiesd conponent the port
    is associated with (connected to). Once the association with a FluidProperties
    components is made, the 'fluid' attribute is assigned.
    """

    fluid: Fluid = None

    def __init__(self, name, P_sym=None, M_sym=None):
        self.domain = Domain.fluid
        self.name = name
        self.pot = P_sym
        self.flow = M_sym
        self.validate()

    def assign_fluid(self, fluid):
        self.fluid = fluid


class RotationalPort(PortBase):
    """Class for acausal port in the rotational domain."""

    def __init__(self, name, t_sym: Sym, w_sym: Sym):
        self.domain = Domain.rotational
        self.name = name
        self.flow = t_sym
        self.pot = w_sym
        self.validate()


class ThermalPort(PortBase):
    """Class for acausal port in the thernal domain."""

    def __init__(self, name, T_sym=None, Q_sym=None):
        self.domain = Domain.thermal
        self.name = name
        self.pot = T_sym
        self.flow = Q_sym
        self.validate()


class TranslationalPort(PortBase):
    """Class for acausal port in the translational domain."""

    def __init__(self, name, f_sym=None, v_sym=None):
        self.domain = Domain.translational
        self.name = name
        self.flow = f_sym
        self.pot = v_sym
        self.validate()


class ComponentBase:
    """Base class for acausal components.

    Attributes:
        ports (dict{port_name:port}):
            A dictionary from port_name to port object.
        syms:
            The set of all Sym objects related to this component.
        eqs:
            The set of all Eqn objects related to this component.
        port_idx_to_name (dict{int:port_id}):
            This is temporary, and only used for integration with the UI.
            It is a map between a numberical id for component ports and the port_id which is a string.
            the numerical id  is negative for ports on the left side of the block in teh UI,
            and positive for ports on the right. -1 and 1 are thr top, increasing values go down.
        zc (dict{zc_idx:(zc_expr,direction)}):
            A dictionary tracking any zero crossings thta this component adds to the AcusalSystem.
            <more details please>

    """

    def __init__(self):
        self.ports: Dict = {}
        self.syms: Set = set()
        self.eqs: Set = set()
        self.port_idx_to_name: Dict = {}
        self.zc: Dict = {}
        self.cond_name_suffix: int = 0
        self.next_zc_idx: int = 0

    def __repr__(self):
        return str(self.__class__.__name__ + "_" + self.name)

    def declare_symbol(
        self,
        ev: EqnEnv,
        sym_name: str,
        base_name: str,
        val: float = None,
        der_sym: Sym = None,
        int_sym: Sym = None,
        kind: SymKind = None,
        ic: float = None,
        ic_fixed: bool = False,
        sym: "sp.Symbol" = None,
        validator: Callable = None,
        invalid_msg: str = None,
    ):
        # declare a symbol in the components system of equations.
        sym = Sym(
            ev,
            sym_name=sym_name,
            base_name=base_name,
            val=val,
            der_sym=der_sym,
            int_sym=int_sym,
            kind=kind,
            ic=ic,
            ic_fixed=ic_fixed,
            sym=sym,
            validator=validator,
            invalid_msg=invalid_msg,
        )

        if sym in self.syms:
            raise ValueError(f"declare_symbol() Sym {sym} already exists.")
        self.syms.add(sym)

        return sym

    def _check_port_name(self, name: str):
        if name in self.ports.keys():
            raise ValueError(f"_check_port_name() port name {name} already exists.")

    """
    Note on the declare_<domain>_port() methods below.
    These methods should return a tuple of all Sym objects related to the port.
    For example:
        electrical port: (V, I) = declare_port(ev,'p')
        rotatioanl port: (torque, angle,velocity,alpha) = declare_port(ev,'flange')
    This way, in the component constructor where declare_<domain>_port() is called,
    it is possible to have convenient access to the port symbols, with terse names
    relevant to the component context.
    For example, a resistor ports can be defined as follows:
        (Vp, Ip) = declare_port(ev,'p')
        (Vn, In) = declare_port(ev,'n')
    Then, with convenient access to the Sym objects, which in Sympy have globally unique
    names, we can still define component equations as follows:
        Eq(Vp.s - Vn.s, Ip.s*R.s)
        Eq(0,Ip.s + In.s)
    As opposed to:
        Eq(self.ports['p'].pot.s - self.ports['n'].pot.s, self.ports['p'].flow.s, R.s)
        Eq(0,self.ports['p'].flows.s + self.ports['n'].flows.s)
    Note that is not exactly how a resistor is defined, it's just an example to illustrate
    why returning the port symbols as a tuple, such that they are locally renamed is the
    convenient way to handle component port related symbols inside the component constructor.
    """

    def declare_electrical_port(
        self,
        ev: EqnEnv,
        port_name: str,
        I_ic: float = None,
        I_ic_fixed: bool = False,
    ) -> Tuple[Sym, Sym]:
        self._check_port_name(port_name)
        sym_base_name = self.name + "_" + port_name
        v = self.declare_symbol(ev, "V", sym_base_name, kind=SymKind.pot)

        i = self.declare_symbol(
            ev,
            "I",
            sym_base_name,
            kind=SymKind.flow,
            ic=I_ic,
            ic_fixed=I_ic_fixed,
        )
        self.ports[port_name] = ElecPort(port_name, V_sym=v, I_sym=i)
        return v, i

    def declare_fluid_port(
        self,
        ev: EqnEnv,
        port_name: str,
        P_ic: float = None,
        P_ic_fixed: bool = False,
    ) -> Tuple[Sym, Sym]:
        self._check_port_name(port_name)
        sym_base_name = self.name + "_" + port_name
        P = self.declare_symbol(
            ev, "P", sym_base_name, kind=SymKind.pot, ic=P_ic, ic_fixed=P_ic_fixed
        )
        M = self.declare_symbol(ev, "M", sym_base_name, kind=SymKind.flow, ic=0.0)
        self.ports[port_name] = FluidPort(port_name, P_sym=P, M_sym=M)
        return P, M

    def declare_rotational_port(
        self,
        ev: EqnEnv,
        port_name: str,
        w_ic: float = None,
        w_ic_fixed: bool = False,
        ang_ic: float = None,
        ang_ic_fixed: bool = False,
    ) -> Tuple[Sym, Sym, Sym, Sym]:
        self._check_port_name(port_name)
        sym_base_name = self.name + "_" + port_name
        t = self.declare_symbol(ev, "t", sym_base_name, kind=SymKind.flow, ic=0.0)
        ang = self.declare_symbol(
            ev,
            "ang",
            sym_base_name,
            kind=SymKind.var,
            ic=ang_ic,
            ic_fixed=ang_ic_fixed,
        )
        w = self.declare_symbol(
            ev,
            "w",
            sym_base_name,
            kind=SymKind.pot,
            int_sym=ang,
            ic=w_ic,
            ic_fixed=w_ic_fixed,
        )
        alpha = self.declare_symbol(
            ev, "alpha", sym_base_name, kind=SymKind.var, int_sym=w
        )
        # encode the derivative relationships
        ang.der_sym = w
        w.der_sym = alpha
        self.ports[port_name] = RotationalPort(port_name, t_sym=t, w_sym=w)
        return t, ang, w, alpha

    def declare_thermal_port(
        self,
        ev: EqnEnv,
        port_name: str,
        T_ic: Sym = None,
        T_ic_fixed: bool = False,
    ) -> Tuple[Sym, Sym]:
        self._check_port_name(port_name)
        sym_base_name = self.name + "_" + port_name
        T = self.declare_symbol(
            ev, "T", sym_base_name, kind=SymKind.pot, ic=T_ic, ic_fixed=T_ic_fixed
        )
        Q = self.declare_symbol(
            ev, "Q", self.name + port_name, kind=SymKind.flow, ic=0.0
        )
        self.ports[port_name] = ThermalPort(port_name, T_sym=T, Q_sym=Q)
        return T, Q

    def declare_translational_port(
        self,
        ev: EqnEnv,
        port_name: str,
        v_ic: float = None,
        v_ic_fixed: bool = False,
        x_ic: float = None,
        x_ic_fixed: bool = False,
    ) -> Tuple[Sym, Sym, Sym, Sym]:
        self._check_port_name(port_name)
        sym_base_name = self.name + "_" + port_name
        f = self.declare_symbol(ev, "f", sym_base_name, kind=SymKind.flow, ic=0.0)
        x = self.declare_symbol(
            ev,
            "x",
            sym_base_name,
            kind=SymKind.var,
            ic=x_ic,
            ic_fixed=x_ic_fixed,
        )
        v = self.declare_symbol(
            ev,
            "v",
            sym_base_name,
            kind=SymKind.pot,
            int_sym=x,
            ic=v_ic,
            ic_fixed=v_ic_fixed,
        )
        a = self.declare_symbol(ev, "a", sym_base_name, kind=SymKind.var, int_sym=v)
        # encode the derivative relationships
        x.der_sym = v
        v.der_sym = a
        self.ports[port_name] = TranslationalPort(port_name, f_sym=f, v_sym=v)
        return f, x, v, a

    def _err_prefix(self, lut_name):
        return f"For component {self.name} and lookup table {lut_name}, the "

    def _val_lut_param_type(self, lut_name, p, p_name):
        prefix_str = self._err_prefix(lut_name)
        if not isinstance(p, (ArrayLike, List)):
            raise AcausalModelError(
                message=prefix_str
                + f"{p_name} param must be an array or list. Type found: {type(p)=}"
            )

    def declare_1D_lookup_table(
        self,
        ev: EqnEnv,
        x_expr,  # sympy expression
        xp_name: str,
        xp: ArrayLike,
        yp_name: str,
        yp: ArrayLike,
        lut_name: str = None,
    ):
        if lut_name is None:
            lut_name = "lut_" + xp_name + "_" + yp_name

        # validate input
        self._val_lut_param_type(lut_name, xp, "xp")
        self._val_lut_param_type(lut_name, yp, "yp")

        xp_array = cnp.array(xp)
        yp_array = cnp.array(yp)

        prefix_str = self._err_prefix(lut_name)
        if len(xp_array) < 2:
            raise AcausalModelError(
                message=prefix_str
                + f"xp param be an array or list of length >= 2. Length: {len(xp_array)}"
            )
        if len(yp_array) < 2:
            raise AcausalModelError(
                message=prefix_str
                + f"xp param be an array or list of length >= 2. Length: {len(yp_array)}"
            )

        if len(yp_array) != len(xp_array):
            raise AcausalModelError(
                message=prefix_str
                + f"xp and yp param be same length. Length xp: {len(xp_array)}. Length yp: {len(yp_array)}"
            )

        # create lookup table symbols
        xp_sym = self.declare_symbol(
            ev,
            xp_name,
            self.name,
            kind=SymKind.param,
            val=xp_array,
        )
        yp_sym = self.declare_symbol(
            ev,
            yp_name,
            self.name,
            kind=SymKind.param,
            val=yp_array,
        )
        lut_f = sp.Function("jax.numpy.interp")(
            x_expr,
            xp_sym.s,
            yp_sym.s,
        )
        lut_sym = self.declare_symbol(
            ev,
            lut_name,
            self.name,
            sym=lut_f,
            kind=SymKind.lut,
        )

        return lut_sym

    def declare_2D_lookup_table(
        self,
        ev: EqnEnv,
        x_expr,  # sympy expression
        xp_name: str,
        xp: ArrayLike,
        y_expr,  # sympy expression
        yp_name: str,
        yp: ArrayLike,
        zp_name: str,
        zp: ArrayLike,
        lut_name: str = None,
        ic: float = None,
    ):
        if lut_name is None:
            lut_name = "lut_" + xp_name + "_" + yp_name

        # validate input
        self._val_lut_param_type(lut_name, xp, "xp")
        self._val_lut_param_type(lut_name, yp, "yp")
        self._val_lut_param_type(lut_name, zp, "zp")

        xp_array = cnp.array(xp)
        yp_array = cnp.array(yp)
        zp_array = cnp.array(zp)

        prefix_str = self._err_prefix(lut_name)
        if not len(zp_array.shape) == 2:
            raise AcausalModelError(
                message=prefix_str
                + f"zp param be a 2 dimensional array or list. Shape: {zp_array.shape}"
            )
        xdim, ydim = zp_array.shape
        if len(xp_array) != xdim:
            raise AcausalModelError(
                message=prefix_str
                + f"xp param must have same length as zp.shape[0]. len(xp)={len(xp_array)}. zp.shape[0]={zp_array.shape[0]}"
            )
        if len(yp_array) != ydim:
            raise AcausalModelError(
                message=prefix_str
                + f"yp param must have same length as zp.shape[1]. len(yp)={len(yp_array)}. zp.shape[1]={zp_array.shape[1]}"
            )

        # create lookup table symbols
        xp_sym = self.declare_symbol(
            ev,
            xp_name,
            self.name,
            kind=SymKind.param,
            val=xp_array,
        )
        yp_sym = self.declare_symbol(
            ev,
            yp_name,
            self.name,
            kind=SymKind.param,
            val=yp_array,
        )
        zp_sym = self.declare_symbol(
            ev,
            zp_name,
            self.name,
            kind=SymKind.param,
            val=zp_array,
        )
        lut_f = sp.Function("cnp.interp2d")(
            xp_sym.s,
            yp_sym.s,
            zp_sym.s,
            x_expr,
            y_expr,
        )
        lut_sym = self.declare_symbol(
            ev,
            lut_name,
            self.name,
            sym=lut_f,
            kind=SymKind.lut,
        )

        return lut_sym

    def declare_conditional(
        self,
        ev: EqnEnv,
        if_expr,
        then_expr,
        else_expr,
        cond_name: str = None,
    ):
        if cond_name is None:
            cond_name = "cond_" + self.name + "_" + str(self.cond_name_suffix)
            self.cond_name_suffix += 1

        cond_f = sp.Piecewise(
            (then_expr, if_expr),
            (else_expr, True),
        )
        cond_sym = self.declare_symbol(
            ev,
            "cond_name",
            self.name,
            sym=cond_f,
            kind=SymKind.cond,
        )
        self.delcare_zc(if_expr, "crosses_zero", True)

        return cond_sym

    def delcare_zc(self, zc_expr, direction: str, is_bool_expr: bool):
        self.zc[self.next_zc_idx] = (zc_expr, direction, is_bool_expr)
        self.next_zc_idx += 1

    def declare_equation(self, e: "sp.Eq", kind=EqnKind.comp):
        eqn = Eqn(e=e, kind=kind)
        if eqn in self.eqs:
            raise ValueError(f"declare_equation() Eqn {eqn} already exists.")
        self.eqs.add(eqn)

    def add_eqs(self, eqs: List["sp.Eq"]):
        for e in eqs:
            self.declare_equation(e)

    def get_syms_by_kind(self, kind: SymKind):
        syms = []
        for sym in self.syms:
            if sym.kind == kind:
                syms.append(sym)
        return syms

    def get_sym_by_port_name(self, port_name: str):
        """
        Mainly used for creating maps between the port indices of the
        acausal components in the orginal diagram and the port indices
        of the AcausalSystem that replaces the acausal components.
        """
        for sym in self.syms:
            # by definition, if the sym is related to a port, it MUST either inp or outp
            if sym.kind in [SymKind.inp, SymKind.outp]:
                if sym.sym_name == port_name:
                    return sym
        return None

    def finalize(self):
        """
        This is an abstract method that is called after some diagram processing
        has been completed. Some components cannot be fully defined until after
        some information about what else they are conencted to has been identified.
        Only after this can the component definition be 'finalized'.
        This method is called on all components, even if most components do nothing.

        Example: Fluid components may have equations whihc depend on fluid properties.
        Since fluid properties are assigned to a set of fluid components using a
        FluidProperties component connected to the network, components who need fluid
        properties intheir equations can only define those equations once the fluid
        properties have been assigned to their ports. this last thing can only happen
        after some diagram processing has been completed, e.g. we need to know which
        FluidProperties component a fluid components port(s) are conencted to.
        """
        pass
