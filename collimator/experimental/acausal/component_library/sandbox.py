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
from .base import SymKind
from .rotational import RotationalOnePort

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")

"""
components here are just sandbox development. they will either go away,
or be moved to a proper domain library.
"""


class TorqueSwitch(RotationalOnePort):
    """
    Dev component to develop conditionals in acausal components.
    """

    def __init__(
        self,
        ev,
        name=None,
        timeThr=2.123456789,
        onTrq=-10.0,
        offTrq=10.0,
    ):
        self.name = self.__class__.__name__ if name is None else name
        # name the port 'flange_a' so the component is interchangebale with a
        # normal torque source with enable_flange_b=False. this makes for easier
        # debugging.
        super().__init__(ev, self.name, p="flange_a")

        timeThr = self.declare_symbol(
            ev, "timeThr", self.name, kind=SymKind.param, val=timeThr
        )
        onTrq = self.declare_symbol(
            ev, "onTrq", self.name, kind=SymKind.param, val=onTrq
        )
        offTrq = self.declare_symbol(
            ev, "offTrq", self.name, kind=SymKind.param, val=offTrq
        )
        cond_sym = self.declare_conditional(
            ev, ev.t <= timeThr.s, onTrq.s, offTrq.s, "trqSwitch"
        )
        self.add_eqs([sp.Eq(self.t.s, cond_sym.s)])
