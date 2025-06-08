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

"""Classes for input/output ports of systems.

These are simple wrappers around SystemCallback that add a few extra properties
and methods for convenience.

TODO: This file is small enough now that it should just be merged into system_base.py
"""

from __future__ import annotations
from abc import abstractmethod

from typing import TYPE_CHECKING, Tuple, Union
import dataclasses

from .dependency_graph import DependencyTicket
from .cache import SystemCallback

if TYPE_CHECKING:
    from .system_base import SystemBase
    from .context import ContextBase
    from ..backend.typing import Array

    # These are hashable objects that can be used to locate a port within a system.
    # Used primarily for tracking connectivity in diagrams.
    InputPortLocator = Tuple[SystemBase, int]  # (system, port_index)
    OutputPortLocator = Tuple[SystemBase, int]  # (system, port_index)
    EitherPortLocator = Union[InputPortLocator, OutputPortLocator]

    # This one is used for error formatting
    DirectedPortLocator = Tuple[SystemBase, str, int]  # (system, direction, port_index)


__all__ = [
    "InputPort",
    "OutputPort",
    "InputPortLocator",
    "OutputPortLocator",
    "EitherPortLocator",
    "DirectedPortLocator",
]


@dataclasses.dataclass
class PortBase(SystemCallback):
    """Base class for input and output ports."""

    # Index of the port within the owning System.  For a Diagram-level port, this
    # will be the index within the Diagram, not within the LeafSystem
    # whose port was forwarded. This should not be confused with callback_index, which
    # will index into the system callbacks, rather than (input/output)_ports.
    # This value should be automatically created by the system when the port is added.
    #
    # Should not be None, but cannot be declared as a required argument because of
    # dataclass __init__ argument ordering, so this is checked in __post_init__.
    index: int = None

    def __post_init__(self, callback):
        super().__post_init__(callback)
        assert self.index is not None, "Port index cannot be None"

    @property
    @abstractmethod
    def direction(self) -> str: ...

    @property
    def locator(self) -> EitherPortLocator:
        return (self.system, self.index)

    @property
    def directed_locator(self) -> DirectedPortLocator:
        return (self.system, self.direction, self.index)

    def __hash__(self) -> int:
        return hash(self.locator)


class OutputPort(PortBase):
    """Output port of a system."""

    @property
    def direction(self) -> str:
        return "out"


class FixedPortManager:
    """Context manager for temporarily fixing a port to a constant value.

    This helps prevent accidentally storing a traced value in a port. See also
    the documentation for `InputPort.fixed`.  Note that this should not need
    to be used directly in most cases.  Instead, use `InputPort.fixed` to access
    a properly created instance of this class.
    """

    def __init__(self, port: InputPort, value):
        self.port = port
        self.value = value

    def __enter__(self):
        self.port.fix_value(self.value)

    def __exit__(self, _exc_type, _exc_value, _exc_traceback):
        self.port.unfix()


class InputPort(PortBase):
    """Input port of a system."""

    is_fixed = False  # Whether the value of this port is fixed to a constant value

    def __post_init__(self, callback):
        super().__post_init__(callback)

    @property
    def direction(self) -> str:
        return "in"

    def fixed(self, value: Array) -> FixedPortManager:
        """Temporarily fix the value of this port to a constant.

        This should be preferred over `InputPort.fix_value` when the value will only
        be fixed for a specific calculation.  This helps prevent accidentally storing
        JAX tracers in the port value, which can lead to errors.

        Example usage:
        ```python
        with system.input_ports[0].fixed(u):
            xcdot = system.eval_time_derivatives(context)
        ```

        Args:
            value (Array): The value to fix the port to.
        """
        return FixedPortManager(self, value)

    def fix_value(self, value: Array):
        """Set the value of this port to a constant.

        This can be used for testing subsystems with inputs in isolation, for instance.
        In order to undo the fixed value, the system can be re-created or added to
        a Diagram and the input can be connected to a different port.

        Args:
            value (Array): The value to fix the port to.

        Notes:
            This should not be called in any "traced" code, for example during an
            optimization or differentiation pass.  This is because the value will be
            a JAX tracer that is then stored in the system, creating a side effect
            and possible a leaked tracer error.  Instead, use `InputPort.fixed` to
            temporarily fix the value.
        """
        self._callback = lambda _: value
        self.default_value = value
        self.prerequisites_of_calc = [DependencyTicket.nothing]
        self.is_fixed = True

    def unfix(self):
        """Restore the port to be "unfixed" and have its value calculated normally."""
        # assert self.is_fixed, (
        #     f"Attempting to unfix port {self.system.name}[{self.name}] which is not "
        #     "fixed.  Not enough information to restore the original callback."
        # )

        # Restore the definition of the callback from the original port creation.
        def _callback(context: ContextBase) -> Array:
            return self.system._eval_input_port(context, self.index)

        self._callback = _callback
        self.default_value = None
        self.prerequisites_of_calc = []  # Default - unknown

        # Add the port into the system's dependency graph, in case the graph was
        # created while the port was fixed.
        tracker = self.system.dependency_graph[DependencyTicket.u]
        tracker.add_prerequisite(self.system.dependency_graph[self.ticket])

        self.is_fixed = False
