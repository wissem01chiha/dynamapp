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
from collimator.experimental.acausal.index_reduction import IndexReduction

# Define the symbol for time
t = sp.symbols("t")

# Define parameters
u1 = sp.Function("u1")(t)
u2 = sp.Function("u2")(t)
u3 = sp.Function("u3")(t)
u4 = sp.Function("u4")(t)

# Define functions of time
x1 = sp.Function("x_1")(t)
x2 = sp.Function("x_2")(t)
x3 = sp.Function("x_3")(t)
x4 = sp.Function("x_4")(t)
x5 = sp.Function("x_5")(t)  # der x1
x6 = sp.Function("x_6")(t)  # der x2
x7 = sp.Function("x_7")(t)  # der x3


# knowns
knowns = {
    u1: 1.0,
    u1.diff(t): 0.0,
    u1.diff(t, 2): 0,
    u2: 1.0,
    u2.diff(t): 0.0,
    u2.diff(t, 2): 0,
    u3: 1.0,
    u3.diff(t): 0.0,
    u4: 1.0,
    u4.diff(t): 0.0,
}

# Define the derivatives of these functions with respect to time
x1_dot = x1.diff(t)
x2_dot = x2.diff(t)

x3_dot = x3.diff(t)
x4_dot = x4.diff(t)

x5_dot = x5.diff(t)
x6_dot = x6.diff(t)
x7_dot = x7.diff(t)

# Define equations
eq0 = x1 + x2 + u1
eq1 = x1 + x2 + x3 + u2
eq2 = x1 + x3_dot + x4 + u3
eq3 = 2 * x5_dot + x6_dot + x7_dot + x4_dot + u4
eq4 = x1_dot - x5
eq5 = x2_dot - x6
eq6 = x3_dot - x7

# Equations list
eqs = [eq0, eq1, eq2, eq3, eq4, eq5, eq6]

# ics = None  # infeasible; error should tell 2 conditions are needed

# ics = {
#     x1: 1.0,
#     x2: 2.0,
# }  # infeasible, error should tell to choose different combination

ics = {x1: 1.0, x5: 2.0}  # feasible
ics_weak = {
    x2: -0.5,
    x3: -0.5,
    x4: -0.5,
    x6: -0.5,
    x7: -0.5,
    x1_dot: -0.5,
    x2_dot: -0.5,
    x3_dot: -0.5,
    x4_dot: -0.5,
    x5_dot: -0.5,
    x6_dot: -0.5,
    x7_dot: -0.5,
}

ir = IndexReduction(t, eqs, knowns, ics, ics_weak, verbose=True)
ir.run_dev()
