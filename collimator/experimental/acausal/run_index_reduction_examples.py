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

import sympy as sp

# from index_reduction import IndexReduction
from acausal_compiler import AcausalCompiler
from component_library.base import eqn_env


def pendulum():
    # Define the symbol for time
    ev = eqn_env()
    t = ev.t

    # Define parameters
    L = sp.symbols("L")
    g = sp.Function("g")(t)  # make g time-varying for testing a time-varying parameter

    # Define functions of time
    x = sp.Function("x")(t)
    y = sp.Function("y")(t)
    w = sp.Function("w")(t)
    z = sp.Function("z")(t)
    T = sp.Function("T")(t)

    # knowns
    knowns = [L, g]

    # Define the derivatives of these functions with respect to time
    xdot = x.diff(t)
    ydot = y.diff(t)
    wdot = w.diff(t)
    zdot = z.diff(t)

    # Define equations
    expr0 = xdot - w
    expr1 = ydot - z
    expr2 = wdot - T * x
    expr3 = zdot - T * y + g
    expr4 = x**2 + y**2 - L**2

    # Equations list
    exprs = [expr0, expr1, expr2, expr3, expr4]

    # ir = IndexReduction(t, eqs, knowns)
    compiler = AcausalCompiler(eqn_env=ev)
    compiler.ir_test_inputs(exprs, knowns)
    compiler.index_reduction()
    # ir()


if __name__ == "__main__":
    pendulum()
