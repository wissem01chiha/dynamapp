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

from typing import List, Optional, TYPE_CHECKING
from .component_library.component_base import Sym, SymKind
from .types import DiagramProcessingData

if TYPE_CHECKING:
    import sympy as sp


class AcausalCompilerError(Exception):
    def __init__(
        self,
        message: str,
        dpd: Optional[DiagramProcessingData] = None,
    ):
        super().__init__(message)
        self.message = message
        self.dpd = dpd

    def __str__(self):
        message = self.message or None
        return f"Compilation of AcaualDiagram failed, please contact support.\n{self._context_info()}\n{message}"

    def _context_info(self) -> str:
        strbuf = []
        if self.dpd:
            strbuf.append(f"Related AcausalDiagram: {self.dpd.ad.name}.")

        return "".join(strbuf)


class AcausalModelError(Exception):
    """An error class for raising errors related to invalid model construction.
    If possible, these should be 'collected' so that multiple errors can be reported
    simultaneously, so as to reduce the number of compilation attempts needed to fix
    all errors."""

    def __init__(
        self,
        message: str,
        components=None,  # : Optional[List[ComponentBase]] cant use this type hint without circular import
        ports=None,  # : Optional[List[Tuple[ComponentBase, PortBase]]]  cant use this type hint without circular import
        include_port_domain: bool = False,
        dpd: Optional[DiagramProcessingData] = None,
        variables: Optional[List["sp.Symbol"]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.components = components
        self.ports = ports
        self.include_port_domain = include_port_domain
        self.dpd = dpd
        self.vars = variables
        # this is just for debugging this method.
        ctx_str = self._context_info()
        print(ctx_str)

    def _deref_vars(self):
        def _append(self, sym, var2ref):
            print(f"\n{sym=}\n\n{self.dpd.ad.sym_to_cmp=}\n\n")
            cmp = self.dpd.ad.sym_to_cmp[sym]
            if cmp not in var2ref.keys():
                var2ref[cmp] = []
            var2ref[cmp].append(sym.sym_name)

            return var2ref

        var2ref = {}  # dict{var:{comp:[params]}}
        for var in self.vars:
            print(f"\n\nnext in self.vars: {var}")
            if var in self.dpd.syms_map_original.keys():
                # if var is a sp.Symbol, get the parent Sym object
                sym = self.dpd.syms_map_original[var]
            elif isinstance(var, Sym):
                sym = var
            else:
                raise ValueError(f"{var=} cannot be dereferenced")

            if sym.kind == SymKind.node_pot:
                print(f"{sym} is node_pot")
                # node potential variable smean nothing for user, so de-alias to the
                # component potential variables.
                aliasees = self.dpd.aliaser_map[sym]
                print(f"{aliasees=}")
                for aliasee_tuple in aliasees:
                    (aliasee, aliasee_expr) = aliasee_tuple
                    print(f"next aliasee: {aliasee}")
                    if aliasee == SymKind.node_pot:
                        raise ValueError(
                            f"node_pot {sym} aliased another node pot {aliasee} which error formatting does not support"
                        )
                    _append(self, aliasee, var2ref)
            else:
                _append(self, sym, var2ref)

        return var2ref

    def __str__(self):
        message = self.message or self.default_message
        return f"{self._context_info()}\n{message}"

    def _context_info(self) -> str:
        strbuf = []
        if self.dpd:
            strbuf.append(f"\nRelated AcausalDiagram:\n\t{self.dpd.ad.name}.")
        if self.components:
            components_str = "\n\t".join([c.name for c in self.components])
            strbuf.append(f"\nRelated components:\n\t{components_str}")
        elif self.ports:
            portnames = []
            for c, port_name in self.ports:
                port_str = c.name + ":" + port_name
                if self.include_port_domain:
                    port = c.ports[port_name]
                    port_str = port_str + f"[{port.domain}]"
                portnames.append(port_str)
            ports_str = "\n\t".join(portnames)
            strbuf.append(f"\nRelated ports:\n\t{ports_str}")
        elif self.vars:
            deref_vars = self._deref_vars()
            strbuf.append(f"\nRelated parameters:\n\t{deref_vars}")

        return "".join(strbuf)

    @property
    def default_message(self):
        return type(self).__name__
