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

from enum import Enum
from typing import TYPE_CHECKING

from collimator.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")

"""
Framework classes for Symbols, Equations, and Systems of Equations.
"""


class EqnEnv:
    """Equation Environment object containing any object which is used
    in equations or expressions and that is common throughout a given
    Acausal model.

    Attributes:
        t (Sympy Symbol):
            The symbol for time which must be used by all equations
            which need time.
    """

    def __init__(self):
        self.t = sp.symbols("t")


class Domain(Enum):
    """Enumeration for the Acausal domains"""

    electrical = 0
    magnetic = 1
    thermal = 2
    rotational = 3
    translational = 4
    fluid = 6


class SymKind(Enum):
    """Enumeration for the 'kind' of a 'Sym' object. 'kind' qualifies the symbol
    w.r.t. how it gets treated when rearanging/simplifying equations, or determining
    initial conditions.

     - 'flow' means this symbol is a flow variable. it must appear in one and only one
        conservation equation for a given 'node'. e.g. all forces sum to 0.
     - 'pot' means this symbol is a potential variable. it must appear in at least one
        equality constraint. e.g. all velocities of ports connected to a 'node' must be equal.
     - 'param' means this symbol is a parameter of the system, e.g. mass, resistance, etc.
     - 'in' means this symbol is an input, similar to param, but the value is read in
        periodically. e.g. force signal for controlled_ideal_force_source.
     - 'out' means this symbol is a output and therefore must appear on the LHS of an
        expression. e.g. force value for ideal force sensor.
     - 'var' means this symbol is a variable of the system that does not match any description
        above. e.g. voltage_across_capacitor = capacitor_pot_positive - capacitor_pot_negative, here
        voltage_across_capacitor is a 'var', while the other two are 'pot'.
        similarly, the time derivaive of voltage_across_capacitor would be a 'var'.
     - 'node_pot' means this symbol is in the potential derivative index of the node. See
        AcausalCompiler.add_node_potential_eqs() for details.
     - 'lut' means the symbol represents a lookup table function
     - 'cond' means conditional, whihc presently is Sympy.Piecewise()
    """

    flow = 0
    pot = 1
    param = 2
    inp = 3
    outp = 4
    var = 6
    node_pot = 7
    lut = 8
    cond = 9


class Sym:
    """
    A class for a symbol that can have an association with the symbols that are its
    integral or derivative w.r.t. time.
    Although declaring derivatives using Sympy.Symbol.diff method is possible, this
    only creates the symbolic meaning of the derivative relationship, it doesn't
    simultaneously track the relationship in a way that it is possible to 'search
    through the graph' of derivative relationships.

    Attributes:
        der_relation (Sympy.Eq):
            Equation relating the self.s symbol to the Sympy derivative it represents.
            For example, if this symbol is acceleration(t), the der_relation would be:
            acceleration(t) = Derivative(velocity(t))
        s_int (Sympy.Derivative):
            The sympy.Derivative object that is the derivative of the int_sym passed in.
            For example, if this symbol is acceleration(t), int_sym is velocity(t), and
            s_int is Derivative(velocity(t)).
        name (string):
            the string used to uniquely identify the symbol in Sympy.
        s (Sympy.Symbol):
            The symbol that this object is a containter for.
        val (number):
            A literal numerical value associated with this symbol.
        kind (SymKind):
            see SymKind documentation.
        ic (number):
            If this symbol could represent the state of a system, this is the initial
            conditions for the state, as assigned by the user.
        ic_fixed (bool):
            If true, the 'ic' is considered a fixed requirement, and must be respected.
            If false, the 'ic' is considered a suggestion, which can be used to fill in
            gaps of system initial conditions.
        int_sym (Sympy.Symbol);
            If this symbol appears as a derivative of time in the equations, then this
            symbol is equal to Derivative(int_sym).
        der_sym (Sympy.Symbol):
            Derivative(this symbol) = der_sym.
        validator (callable):
            if the symbol is a parameter, a function that validates the parameter val. e.g.
            lambda val: val >=0.0.
        invalid_msg (str):
            an error message to be passed to error constructor in the event the parameter
            validator fails. e.g. "<param_name> must be > 0".
    """

    def __init__(
        self,
        eqn_env,
        sym_name=None,  # the symbol name, e.g. alpha for angular acceleration
        base_name=None,  # comp_name or comp_name+port_name
        name=None,  # the full name. used for autogenerated names like node_pots
        val=None,
        der_sym=None,
        int_sym=None,
        kind=None,
        ic=None,
        ic_fixed=False,
        sym=None,
        validator=None,
        invalid_msg=None,
    ):
        if name is None:
            # for retaining the symbol base if the symbol was created that way
            self.sym_name = sym_name
            if base_name is not None:
                self.name = base_name + "_" + sym_name
            else:
                self.name = sym_name
        else:
            self.name = self.sym_name = name

        if kind not in SymKind:
            raise Exception(f"kind:{kind} of symbol:{self.name} not one of {SymKind}")

        if kind == SymKind.param and val is None:
            raise Exception(f"symbol:{self.name} has kind param, val cannot be None")

        if kind in [SymKind.param, SymKind.inp, SymKind.outp]:
            if ic is not None:
                raise Warning(f"assigning ic to symbol if kind:{kind} has no effect.")

        # to capture all the conditions which would make the symbol a function of time.
        is_fcn = kind != SymKind.param

        self.der_relation = None
        self.s_int = None  # the sympy.Derivative object that is the derivative of the int_sym passed in
        if sym is not None:
            # this symbol has been defined externally
            self.s = sym
        elif int_sym is not None:
            # when this symbol is the derivative w.r.t. time of another symbol,
            # we have to define it as such.
            self.s = sp.Function(self.name)(eqn_env.t)
            self.s_int = int_sym.s.diff(eqn_env.t)
            # debatable whether this should happen in here or in the components.
            # it's more of a components equation, but if done there, it would
            # mean lots of repeated code, where as here it's automatic.
            # when done here, it means that DiagramProcessing needs to collect
            # equations from Sym objects as well.
            self.der_relation = Eqn(
                e=sp.Eq(self.s, self.s_int), kind=EqnKind.der_relation
            )

            # NOTE: For some domains, all der_relation equations come from node potential
            # variables. In these cases, it would be cleaner to add the der_relations in the
            # compiler. However, some domains have der_relations for non-node_pot variables,
            # so at least for now, it seems best to keep declaration of der_relations here.
        elif is_fcn:
            # when this symbol is a function of time, we have to define it as such.
            self.s = sp.Function(self.name)(eqn_env.t)
        else:
            # otherwise, the symbol is just a plain symbol with no other relations.
            self.s = sp.Symbol(self.name)
        self.val = val
        self.kind = kind
        self.ic = ic
        self.ic_fixed = ic_fixed
        self.int_sym = int_sym
        self.der_sym = der_sym
        self.validator = validator
        self.invalid_msg = invalid_msg

    def __repr__(self):
        return str(self.name)

    def subs(self, e1, e2):
        self.s = self.s.subs(e1, e2)
        return self


class EqnKind(Enum):
    """Enumeration for the 'kind' of an 'Eqn' object. 'kind' qualifies the equation
    w.r.t. how it gets treated when rearanging/simplifying equations, or determining
    initial conditions.

     - 'pot' this is a 'potential variable' constraint equation.
     - 'flow' this is a 'flow variable' summation equation.
     - 'der_relation' this is a deerivative relation equation,
        e.g. accel = Derivative(vel).
     - 'comp' this equation define component behavior.
     - 'outp' this equation relates an output symbol (a symbol who value is written
        to a causal outport) to an expression.
    """

    pot = 0
    flow = 1
    der_relation = 2
    comp = 3
    outp = 4


class Eqn:
    """Class for holding Sympy.Eq objects with additional qualification data.

    Attributes;
        e (Sympy.Eq):
            The equation.
        kind (EqnKind):
            See documentation for EqnKind.
        node_id:
            The node of the AcausalDiagram that this equation belongs to.
            Not user settable. Assigned by AcausalCompiler.
    """

    def __init__(self, e, kind=None, node_id=None):
        self.e = e
        self.kind = kind
        self.node_id = node_id  #

    def subs(self, e1, e2):
        self.e = self.e.subs(e1, e2)
        return self

    def __repr__(self):
        if self.kind is None:
            return str(self.e)
        else:
            return (
                str(self.e) + ":\t\t\t" + str(self.kind) + "\tnid:" + str(self.node_id)
            )

    @property
    def expr(self):
        """
        Return the equation, re-arranged as an expression equal to zero.
        """
        return self.e.lhs - self.e.rhs
