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

"""Builder class for constructing block diagrams."""

from __future__ import annotations

from typing import Tuple, List, Mapping, TYPE_CHECKING, overload

from ..logging import logger

from . import build_recorder
from .system_base import SystemBase
from .diagram import Diagram
from .error import InputNotConnectedError, StaticError
from .parameter import Parameter

if TYPE_CHECKING:
    from .port import (
        InputPort,
        OutputPort,
        InputPortLocator,
        OutputPortLocator,
    )

    ExportedInputData = Tuple[InputPortLocator, str]  # (locator, port_name)

__all__ = [
    "DiagramBuilder",
]


class BuilderError(StaticError):
    """Errors related to constructing diagrams."""

    pass


class SystemNameNotUniqueError(BuilderError):
    def __init__(self, system: SystemBase):
        super().__init__(f"System name {system.name} is not unique", system=system)


class DisconnectedInputError(InputNotConnectedError, BuilderError):
    def __init__(self, input_port_locator: InputPortLocator):
        system, port_index = input_port_locator
        super().__init__(
            f"Input port {system.name}[{port_index}] is not connected",
            system=system,
            port_index=port_index,
            port_direction="in",
        )


class EmptyDiagramError(BuilderError):
    def __init__(self, name: str):
        super().__init__(f"Cannot compile an empty diagram: {name}")


class DiagramBuilder:
    """Class for constructing block diagram systems.

    The `DiagramBuilder` class is responsible for building a diagram by adding systems, connecting ports,
    and exporting inputs and outputs. It keeps track of the registered systems, input and output ports,
    and the connection map between input and output ports of the child systems.
    """

    def __init__(self):
        # Child input ports that are exported as diagram-level inputs
        self._input_port_ids: List[InputPortLocator] = []
        self._input_port_names: List[str] = []
        # Child output ports that are exported as diagram-level outputs
        self._output_port_ids: List[OutputPortLocator] = []
        self._output_port_names: List[str] = []

        # Connection map between input and output ports of the child systems
        self._connection_map: Mapping[InputPortLocator, OutputPortLocator] = {}

        # List of registered systems
        self._registered_systems: List[SystemBase] = []

        # Name lookup for input ports
        self._diagram_input_indices: Mapping[str, InputPortLocator] = {}

        # All input ports of child systems (for use in ensuring proper connectivity)
        self._all_input_ports: List[InputPortLocator] = []

        # Each DiagramBuilder can only be used to build a single diagram.  This is to
        # avoid creating multiple diagrams that reference the same LeafSystem. Doing so
        # may or may not actually lead to problems, since the LeafSystems themselves
        # should act like a collection of pure functions, but best practice is to have
        # each leaf system be fully unique.
        self._already_built = False

    @overload
    def add(self, system: SystemBase) -> SystemBase: ...

    @overload
    def add(self, system: SystemBase, *systems: SystemBase) -> List[SystemBase]: ...

    def add(self, *systems: SystemBase) -> List[SystemBase] | SystemBase:
        """Add one or more systems to the diagram.

        Args:
            *systems SystemBase:
                System(s) to add to the diagram.

        Returns:
            List[SystemBase] | SystemBase:
                The added system(s). Will return a single system if there is only
                a single system in the argument list.

        Raises:
            BuilderError: If the diagram has already been built.
            BuilderError: If the system is already registered.
            BuilderError: If the system name is not unique.
        """
        for system in systems:
            self._check_not_already_built()
            self._check_system_not_registered(system)
            self._check_system_name_is_unique(system)
            self._registered_systems.append(system)

            # Add the system's input ports to the list of all input ports
            # So that we can make sure they're all connected before building.
            self._all_input_ports.extend([port.locator for port in system.input_ports])

            logger.debug("Added system %s to DiagramBuilder", system.name)
            logger.debug(
                "    Registered systems: %s",
                [s.name for s in self._registered_systems],
            )
        build_recorder.add_block(self, systems)

        return systems[0] if len(systems) == 1 else systems

    def connect(self, src: OutputPort, dest: InputPort):
        """Connect an output port to an input port.

        The input port and output port must both belong to systems that have
        already been added to the diagram.  The input port must not already be
        connected to another output port.

        Args:
            src (OutputPort): The output port to connect.
            dest (InputPort): The input port to connect.

        Raises:
            BuilderError: If the diagram has already been built.
            BuilderError: If the source system is not registered.
            BuilderError: If the destination system is not registered.
            BuilderError: If the input port is already connected.
        """
        self._check_not_already_built()
        self._check_system_is_registered(src.system)
        self._check_system_is_registered(dest.system)
        self._check_input_not_connected(dest.locator)

        build_recorder.connect_ports(self, src, dest)

        self._connection_map[dest.locator] = src.locator

        logger.debug(
            f"Connected port {src.name} of system {src.system.name} to port {dest.name} of system {dest.system.name}"
        )
        logger.debug(f"Connection map so far: {self._connection_map}")

    def export_input(self, port: InputPort, name: str = None) -> int:
        """Export an input port of a child system as a diagram-level input.

        The input port must belong to a system that has already been added to the
        diagram. The input port must not already be connected to another output port.

        Args:
            port (InputPort): The input port to export.
            name (str, optional):
                The name to assign to the exported input port. If not provided, a
                unique name will be generated.

        Returns:
            int: The index (in the to-be-built diagram) of the exported input port.

        Raises:
            BuilderError: If the diagram has already been built.
            BuilderError: If the system is not registered.
            BuilderError: If the input port is already connected.
            BuilderError: If the input port name is not unique.
        """
        self._check_not_already_built()
        self._check_system_is_registered(port.system)
        self._check_input_not_connected(port.locator)

        if name is None:
            # Since the system names are unique, auto-generated port names are also unique
            # at the level of _this_ diagram (subsystems can have ports with the same name)
            name = f"{port.system.name}_{port.name}"
        elif name in self._diagram_input_indices:
            raise BuilderError(
                f"Input port name {name} is not unique",
                system=port.system,
                port_index=port.index,
                port_direction="in",
            )

        # Index at the diagram (not subsystem) level
        port_index = len(self._input_port_ids)
        self._input_port_ids.append(port.locator)
        self._input_port_names.append(name)

        self._diagram_input_indices[name] = port_index

        build_recorder.export_port(self, port.system, "input", port.index, name)

        return port_index

    def export_output(self, port: OutputPort, name: str = None) -> int:
        """Export an output port of a child system as a diagram-level output.

        The output port must belong to a system that has already been added to the
        diagram.

        Args:
            port (OutputPort): The output port to export.
            name (str, optional):
                The name to assign to the exported output port. If not provided, a
                unique name will be generated.

        Returns:
            int: The index (in the to-be-built diagram) of the exported output port.

        Raises:
            BuilderError: If the diagram has already been built.
            BuilderError: If the system is not registered.
            BuilderError: If the output port name is not unique.
        """
        self._check_not_already_built()
        self._check_system_is_registered(port.system)

        if name is None:
            # Since the system names are unique, auto-generated port names are also unique
            # at the level of _this_ diagram (subsystems can have ports with the same name)
            name = f"{port.system.name}_{port.name}"
        elif name in self._output_port_names:
            raise BuilderError(
                f"Output port name {name} is not unique",
                system=port.system,
                port_index=port.index,
                port_direction="out",
            )

        # Index at the diagram (not subsystem) level
        port_index = len(self._output_port_ids)
        self._output_port_ids.append(port.locator)
        self._output_port_names.append(name)

        build_recorder.export_port(self, port.system, "output", port.index, name)

        return port_index

    def _check_not_already_built(self):
        if self._already_built:
            raise BuilderError(
                "DiagramBuilder: build has already been called to "
                "create a diagram; this DiagramBuilder may no longer be used."
            )

    def _check_system_name_is_unique(self, system: SystemBase):
        if system.name in map(lambda s: s.name, self._registered_systems):
            raise SystemNameNotUniqueError(system)

    def _system_is_registered(self, system: SystemBase) -> bool:
        # return (system is not None) and (system in self._registered_systems)
        if system.system_id is None:  # system.__init__ is not done yet
            return False
        return system.system_id in map(lambda s: s.system_id, self._registered_systems)

    def _check_system_not_registered(self, system: SystemBase):
        if self._system_is_registered(system):
            raise BuilderError(
                f"System {system.name} is already registered",
                system=system,
            )

    def _check_system_is_registered(self, system: SystemBase):
        if not self._system_is_registered(system):
            raise BuilderError(
                f"System {system.name} is not registered",
                system=system,
            )

    def _check_input_not_connected(self, input_port_locator: InputPortLocator):
        if not (
            (input_port_locator not in self._input_port_ids)
            and (input_port_locator not in self._connection_map)
        ):
            system, port_index = input_port_locator
            raise BuilderError(
                f"Input port {port_index} for {system} is already connected",
                system=system,
                port_index=port_index,
                port_direction="in",
            )

    def _check_input_is_connected(self, input_port_locator: InputPortLocator):
        if not (
            (input_port_locator in self._input_port_ids)
            or (input_port_locator in self._connection_map)
        ):
            raise DisconnectedInputError(input_port_locator)

    def _check_contents_are_complete(self):
        # Make sure all the systems referenced in the builder attributes are registered

        # Check that systems and registered_systems have the same elements
        for system in self._registered_systems:
            self._check_system_is_registered(system)

        # Check that connection_map only refers to registered systems
        for (
            input_port_locator,
            output_port_locator,
        ) in self._connection_map.items():
            self._check_system_is_registered(input_port_locator[0])
            self._check_system_is_registered(output_port_locator[0])

        # Check that input_port_ids and output_port_ids only refer to registered systems
        for port_locator in [*self._input_port_ids, *self._output_port_ids]:
            self._check_system_is_registered(port_locator[0])

    def _check_ports_are_valid(self):
        for dst, src in self._connection_map.items():
            dst_sys, dst_idx = dst
            if (dst_idx < 0) or (dst_idx >= dst_sys.num_input_ports):
                raise BuilderError(
                    f"Input port index {dst_idx} is out of range "
                    f"(0-{dst_sys.num_input_ports-1})",
                    system=dst_sys,
                    port_index=dst_idx,
                    port_direction="in",
                )
            src_sys, src_idx = src
            if (src_idx < 0) or (src_idx >= src_sys.num_output_ports):
                raise BuilderError(
                    f"Output port index {src_idx} is out of range "
                    f"(0-{src_sys.num_output_ports-1})",
                    system=src_sys,
                    port_index=src_idx,
                    port_direction="out",
                )

    def build(
        self,
        name: str = "root",
        ui_id: str = None,
        parameters: dict[str, Parameter] = None,
    ) -> Diagram:
        """Builds a Diagram system with the specified name and system ID.

        Args:
            name (str, optional): The name of the diagram. Defaults to "root".
            ui_id (str, optional): The unique identifier for the diagram.
            parameters (dict[str, Parameter], optional):
                A dictionary of dynamic parameters to declare for the diagram.

        Returns:
            Diagram: The newly constructed diagram.

        Raises:
            EmptyDiagramError: If no systems are registered in the diagram.
            BuilderError: If the diagram has already been built.
            AlgebraicLoopError: If an algebraic loop is detected in the diagram.
            DisconnectedInputError: If an input port is not connected.
        """
        self._check_not_already_built()
        self._check_contents_are_complete()
        self._check_ports_are_valid()

        # Check that all internal input ports are connected
        for input_port_locator in self._input_port_ids:
            self._check_input_is_connected(input_port_locator)

        if len(self._registered_systems) == 0:
            raise EmptyDiagramError(name)

        diagram = Diagram(
            nodes=self._registered_systems,
            name=name,
            connection_map=self._connection_map,
            ui_id=ui_id,
        )

        build_recorder.build_diagram(self, diagram, parameters)

        if parameters:
            for name, parameter in parameters.items():
                diagram.declare_dynamic_parameter(name, parameter)
                diagram.instance_parameters.add(name)

        # Export diagram-level inputs
        for locator, port_name in zip(self._input_port_ids, self._input_port_names):
            diagram.export_input(locator, port_name)

        # Export diagram-level outputs
        assert len(self._output_port_ids) == len(self._output_port_names)
        for locator, port_name in zip(self._output_port_ids, self._output_port_names):
            diagram.export_output(locator, port_name)

        self._already_built = True  # Prevent further use of this builder
        return diagram
