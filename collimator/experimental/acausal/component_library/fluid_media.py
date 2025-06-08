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

from dataclasses import dataclass
from enum import Enum
from .base import Sym, SymKind


"""
Fluid media specification classes.
These appear here alone to avoid circular import issues.
If they are located in fluid_i.py, they need to be imported into base.py, but
fluid_i.py imports from base.py.
If they are located in base.py, this cullters that file.
"""


class FluidName(Enum):
    """Enumeration class for the names of fluids for which property sets are
    provided.
    """

    water = 0
    hydraulic_fluid = 1


@dataclass
class Water:
    name = "water"
    density = 1000  # kg/m**3
    viscosity_dyn = 0.89e3  # Pa*s
    viscosity_kin = viscosity_dyn / density  # m^2/s


@dataclass
class HydraulicFluid:
    name = "hydraulic_fluid"
    density = 950  # kg/m**3
    viscosity_kin = 40e-6  # m^2/s
    viscosity_dyn = viscosity_kin * density  # Pa*s


class Fluid:
    """Class for holding the symbols for fluid properties."""

    def __init__(self, fluid="water"):
        if fluid == FluidName.water:
            fp = Water()
        elif fluid == FluidName.hydraulic_fluid:
            fp = HydraulicFluid()
        else:
            raise ValueError(
                f"Fluid class, {fluid} is incorrect input for arg 'fluid'."
            )

        self.density = Sym(
            None,  # eqn_env is not needed
            name=fp.name + "_density",
            kind=SymKind.param,
            val=fp.density,
        )
        self.viscosity_kin = Sym(
            None,  # eqn_env is not needed
            name=fp.name + "_viscosity",
            kind=SymKind.param,
            val=fp.viscosity_kin,
        )
        self.viscosity_dyn = Sym(
            None,  # eqn_env is not needed
            name=fp.name + "_viscosity",
            kind=SymKind.param,
            val=fp.viscosity_dyn,
        )
        self.syms = [self.density, self.viscosity_kin, self.viscosity_dyn]
