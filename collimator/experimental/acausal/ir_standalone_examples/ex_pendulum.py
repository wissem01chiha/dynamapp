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
L = sp.symbols("L")
g = sp.symbols("g")

# Define functions of time
x = sp.Function("x")(t)
y = sp.Function("y")(t)
w = sp.Function("w")(t)
z = sp.Function("z")(t)
T = sp.Function("T")(t)

# knowns
knowns = {L: 1.0, g: 9.8}

# Define the derivatives of these functions with respect to time
xdot = x.diff(t)
ydot = y.diff(t)
wdot = w.diff(t)
zdot = z.diff(t)

# Define equations
eq0 = xdot - w
eq1 = ydot - z
eq2 = wdot - T * x
eq3 = zdot - T * y + g
eq4 = x**2 + y**2 - L**2

# Equations list
eqs = [eq0, eq1, eq2, eq3, eq4]

ics = {x: 3.14 / 20, ydot: 1.0}
ics_weak = {y: -0.5, z: -0.5, w: -0.5, T: -0.5, xdot: -0.5, zdot: -0.5, wdot: -0.5}

ir = IndexReduction(t, eqs, knowns, ics, ics_weak, verbose=True)
ir.run_dev()
