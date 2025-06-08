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

from collimator.experimental.acausal.component_library.base import SymKind


class AcausalDiagram:
    """
    collection of components and connections representing a network of acausal components.
    """

    def __init__(self, name=None, comp_list=None, cnctn_list=None):
        self.name = "acausal_diagram" if name is None else name
        self.comps = set() if comp_list is None else set(comp_list)
        self.connections = [] if cnctn_list is None else cnctn_list

        # these will get populated in DiagramProcessing:finalize_diagram().
        # the reason for this is that for fluid domain, fluid property syms
        # are only appropriately identified after some graph analysis of the
        # AcausalDiagram.
        self.syms = set()  # the Sym objects
        self.syms_sp = set()  # the sympy symbols
        self.eqs = set()  # accumulate equations, and remove
        # dict[sym:cmp] needed to dereference syms to their source compnent
        self.sym_to_cmp = {}

    def connect(self, cmp_a, port_a, cmp_b, port_b):
        self.comps.update(set([cmp_a, cmp_b]))
        self.connections.append(((cmp_a, port_a), (cmp_b, port_b)))

    def add_cmp_sympy_syms(self, cmp):
        if cmp not in self.sym_to_cmp.values():
            # check for symbol uniqueness
            cmp_syms_sp = [s.s for s in cmp.syms]
            dupes = self.syms_sp.intersection(cmp_syms_sp)
            if dupes:
                # @am. this error occurred in early development, that's why this error exists.
                # that said, it is untested because it's not possible to make 2 components with
                # same name in same AcausalDiagram, which means it's not possible to have two symbols
                # with same name. So maybe this error is obsolete.
                msg = (
                    f"AcausalDiagram. {cmp.name} has symbols which already appear in the acausal model. {dupes}. "
                    "Try giving the component a different name."
                )
                # this doesn't use AcausalModelError to avoid circular imports
                raise ValueError(msg=msg)
            self.syms_sp.update(cmp_syms_sp)
            for sym_ in list(cmp.syms):
                self.sym_to_cmp[sym_] = cmp

    @property
    def input_syms(self):
        syms = []
        for comp in self.comps:
            syms += comp.get_syms_by_kind(kind=SymKind.inp)
        return syms

    @property
    def num_inputs(self):
        return len(self.input_syms)

    @property
    def has_inputs(self):
        return self.num_inputs > 0

    @property
    def output_syms(self):
        syms = []
        for comp in self.comps:
            syms += comp.get_syms_by_kind(kind=SymKind.outp)
        return syms

    @property
    def num_outputs(self):
        return len(self.output_syms)

    @property
    def has_outputs(self):
        return self.num_outputs > 0
