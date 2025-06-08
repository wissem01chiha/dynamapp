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
K1 = sp.symbols("K_1")
K2 = sp.symbols("K_2")
K3 = sp.symbols("K_3")
K4 = sp.symbols("K_4")
C0 = sp.symbols("C_0")
T0 = sp.symbols("T_0")

# Define functions of time
C = sp.Function("C")(t)
T = sp.Function("T")(t)
R = sp.Function("R")(t)
Tc = sp.Function("T_c")(t)

u = sp.Function("u")(t)  # known

# knowns
knowns = {
    K1: 1.0,
    K2: 1.0,
    K3: 1.0,
    K4: 1.0,
    C0: 1.0,
    T0: 1.0,
    u: 1.0,
    u.diff(t): 0.0,
    u.diff(t, 2): 0,
}

# Define the derivatives of these functions with respect to time
Cdot = C.diff(t)
Tdot = T.diff(t)

# Define equations
eq0 = Cdot - K1 * (C0 - C) + R
eq1 = Tdot - K1 * (T0 - T) - K2 * R + K3 * (T - Tc)
eq2 = R - K3 * sp.exp(-K4 / T) * C
eq3 = C - u

# Equations list
eqs = [eq0, eq1, eq2, eq3]

ics = None
ics_weak = {C: -0.5, T: -0.5, R: -0.5, Tc: -0.5, Cdot: -0.5, Tdot: -0.5}

ir = IndexReduction(t, eqs, knowns, ics, ics_weak, verbose=True)
ir.run_dev()
