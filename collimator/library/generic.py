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

from __future__ import annotations
from typing import Callable

from ..framework import LeafSystem, DependencyTicket

__all__ = [
    "SourceBlock",
    "FeedthroughBlock",
    "ReduceBlock",
]


class SourceBlock(LeafSystem):
    """Simple blocks with a single time-dependent output"""

    def __init__(self, func: Callable, **kwargs):
        """Create a source block with a time-dependent output.

        Args:
            func (Callable):
                A function of time and parameters that returns a single value.
                Signature should be `func(time, **parameters) -> Array`.
        """
        super().__init__(**kwargs)
        self._output_port_idx = self.declare_output_port(
            None,
            name="out_0",
            prerequisites_of_calc=[DependencyTicket.time],
            requires_inputs=False,
        )
        self.replace_op(func)

    def replace_op(self, func):
        def _callback(time, state, *inputs, **parameters):
            return func(time, **parameters)

        self.configure_output_port(
            self._output_port_idx,
            _callback,
            prerequisites_of_calc=[DependencyTicket.time],
            requires_inputs=False,
        )


class FeedthroughBlock(LeafSystem):
    """Simple feedthrough blocks with a function of a single input"""

    def __init__(self, func, parameters={}, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )
        self.replace_op(func)

    def replace_op(self, func):
        def _callback(time, state, *inputs, **parameters):
            return func(*inputs, **parameters)

        self.configure_output_port(
            self._output_port_idx,
            _callback,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )


class ReduceBlock(LeafSystem):
    def __init__(self, n_in, op, parameters={}, **kwargs):
        super().__init__(**kwargs)

        for i in range(n_in):
            self.declare_input_port()

        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )

        self.replace_op(op)

    def replace_op(self, op):
        def _compute_output(time, state, *inputs, **parameters):
            return op(inputs, **parameters)

        self.configure_output_port(
            self._output_port_idx,
            _compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            requires_inputs=True,
        )
