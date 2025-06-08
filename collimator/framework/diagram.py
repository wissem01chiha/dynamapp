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

"""Composite block-diagram representation of a dynamical system.

A Diagram is a collection of Systems connected together to form a larger hybrid
dynamical system. Diagrams can be nested to any depth, creating a tree-structured
block diagram.

As with the LeafSystem, the Diagram comprises a a collection of pure functions that
can be evaluated given a corresponding Context object containing the composite
states, parameters, etc.
"""

from __future__ import annotations
import dataclasses
from typing import (
    TYPE_CHECKING,
    List,
    Mapping,
    Set,
    Tuple,
    Hashable,
    Iterator,
)
from collections import OrderedDict

from ..logging import logger
from .event import DiagramEventCollection, FlatEventCollection
from .error import InputNotConnectedError, StaticError
from .system_base import SystemBase, UpstreamEvalError, next_system_id
from .context_factory import DiagramContextFactory
from .dependency_graph import DependencyTicket, DiagramDependencyGraphFactory
from .parameter import Parameter
from .pprint import pprint_fancy

if TYPE_CHECKING:
    from .error import ErrorCollector
    from .port import (
        PortBase,
        InputPortLocator,
        OutputPortLocator,
        DirectedPortLocator,
    )

__all__ = [
    "AlgebraicLoopError",
    "Diagram",
]

if TYPE_CHECKING:
    from .cache import SystemCallback
    from .port import InputPortLocator, OutputPortLocator
    from .state import LeafState
    from .leaf_system import LeafSystem
    from .context import DiagramContext
    from ..backend.typing import Array


class AlgebraicLoopError(StaticError):
    def __init__(self, name: str, loop: list[DirectedPortLocator]):
        # `loop` is a list of ports that form a cycle. From there, we can recover
        # system, direction and index as port[0], port[1], port[2] respectively.
        str_loop = " \u2192 ".join(
            f"{port[0].name_path_str}.{port[1]}[{port[2]}]" for port in loop
        )
        super().__init__(
            f"Algebraic loop detected in {name}: {str_loop}",
            loop=loop,
            system=None,  # Not applicable since this involves multiple systems
        )


@dataclasses.dataclass
class Diagram(SystemBase):
    """Composite block-diagram representation of a dynamical system.

    A Diagram is a collection of Systems connected together to form a larger hybrid
    dynamical system. Diagrams can be nested to any depth, creating a tree-structured
    block diagram.

    NOTE: The Diagram class is not intended to be constructed directly.  Instead,
    use the `DiagramBuilder` to construct a Diagram, which will pass the appropriate
    information to this constructor.
    """

    # Redefine here to make pylint happy
    system_id: Hashable = dataclasses.field(default_factory=next_system_id, init=False)
    name: str = None  # Human-readable name for this system (optional)
    ui_id: str = None  # UUID of the block when loaded from JSON (optional)

    # None of these attributes are intended to be modified or accessed directly after
    # construction.  Instead, use the interface defined by `SystemBase`.

    # Direct children of this Diagram
    nodes: List[SystemBase] = dataclasses.field(default_factory=list)

    # Mapping from input ports to output ports of child subsystems
    connection_map: Mapping[InputPortLocator, OutputPortLocator] = dataclasses.field(
        default_factory=dict,
    )

    # Optional identifier for "reference diagrams"
    ref_id: str = None

    # for serialization
    instance_parameters: set[str] = dataclasses.field(default_factory=set)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}, {len(self.nodes)} nodes)"

    def _pprint(self, prefix="", fancy=True) -> str:
        if fancy:
            return pprint_fancy(prefix, self)
        return f"{prefix}|-- {self.name}\n"

    def _pprint_helper(self, prefix="", fancy=True) -> str:
        s = self._pprint(prefix=prefix, fancy=fancy)
        for _, substate in enumerate(self.nodes):
            s += substate._pprint_helper(prefix=f"{prefix}    ", fancy=fancy)
        return s

    def __hash__(self) -> Hashable:
        return hash(self.system_id)

    def __getitem__(self, name: str) -> SystemBase:
        # Access by name - for convenient user interface only.  Programmatic
        #  access should use the nodes directly, e.g. `self.nodes[idx]`
        lookup = {node.name: node for node in self.nodes}
        return lookup[name]

    def __iter__(self) -> Iterator[SystemBase]:
        return iter(self.nodes)

    def __post_init__(self):
        super().__post_init__()

        # Set parent for all the immediate child systems
        for node in self.nodes:
            node.parent = self

        # The map of subsystem inputs/outputs to inputs/outputs of this Diagram.
        self._input_port_map: Mapping[InputPortLocator, int] = {}
        self._output_port_map: Mapping[OutputPortLocator, int] = {}

        # Also need the inverse output map, for determining feedthrough paths.
        self._inv_output_port_map: Mapping[int, OutputPortLocator] = {}

        # Leaves of the system tree (not necessarily the same as the direct
        # children of this Diagram, which may themselves be Diagrams)
        self.leaf_systems: List[LeafSystem] = []
        for sys in self.nodes:
            if isinstance(sys, Diagram):
                # FIXME: In case of 'param estimation' optimization run, we end
                # up here and sys.leaf_systems is now None. Using or [] fixes
                # the crash but something is a bit fishy.
                self.leaf_systems.extend(sys.leaf_systems or [])
                # No longer need the child leaf systems, since methods using this
                # should only be called from the top level.
                sys.leaf_systems = None
            else:
                self.leaf_systems.append(sys)

    def post_simulation_finalize(self) -> None:
        """Perform any post-simulation cleanup for this system."""
        for system in self.nodes:
            system.post_simulation_finalize()

    # Inherits docstrings from SystemBase
    @property
    def has_feedthrough_side_effects(self) -> bool:
        # See explanation in `SystemBase.has_feedthrough_side_effects`.
        return any(sys.has_feedthrough_side_effects for sys in self.nodes)

    # Inherits docstrings from SystemBase
    @property
    def has_ode_side_effects(self) -> bool:
        # Return true if either of the following are true:
        # 1. At least one subsystem has ODE side effects
        # 2. At least one subsystem has feedthrough side effects and the output
        #    ports of the diagram are used as ODE inputs.

        # If no subsystems have feedthrough side effects, we're done.
        if not self.has_feedthrough_side_effects:
            return False

        # If any subsystem is already known to have this property, we're done.
        if any(sys.has_ode_side_effects for sys in self.nodes):
            return True

        # If we get here, we need to actually test the dependency graph.
        for sys in self.nodes:
            if sys.has_feedthrough_side_effects:
                for port in sys.output_ports:
                    tracker = port.tracker
                    if tracker.is_prerequisite_of([DependencyTicket.xcdot]):
                        return True
        return False

    @property
    def has_continuous_state(self) -> bool:
        return any(sys.has_continuous_state for sys in self.nodes)

    @property
    def has_discrete_state(self) -> bool:
        return any(sys.has_discrete_state for sys in self.nodes)

    @property
    def has_zero_crossing_events(self) -> bool:
        return any(sys.has_zero_crossing_events for sys in self.nodes)

    @property
    def num_systems(self) -> int:
        # Number of subsystems _at this level_
        return len(self.nodes)

    def check_types(
        self,
        context: DiagramContext,
        error_collector: ErrorCollector = None,
    ) -> None:
        """Perform any system-specific static analysis."""
        for system in self.nodes:
            system.check_types(
                context,
                error_collector=error_collector,
            )

    #
    # Simulation interface
    #

    # Inherits docstrings from SystemBase
    def eval_time_derivatives(self, root_context: DiagramContext) -> List[Array]:
        leaf_systems = [
            subctx.owning_system for subctx in root_context.continuous_subcontexts
        ]
        return [sys.eval_time_derivatives(root_context) for sys in leaf_systems]

    @property
    def mass_matrix(self) -> List[Array]:
        return [sys.mass_matrix for sys in self.leaf_systems]

    @property
    def has_mass_matrix(self) -> bool:
        return any(sys.has_mass_matrix for sys in self.leaf_systems)

    #
    # Event handling
    #
    @property
    def state_update_events(self) -> FlatEventCollection:
        assert self.parent is None, (
            "Can only get periodic events from top-level Diagram, not "
            f"{self.system_id} with parent {self.parent.system_id}"
        )
        events = sum(
            [sys.state_update_events for sys in self.leaf_systems],
            start=FlatEventCollection(),
        )
        return events

    @property
    def zero_crossing_events(self) -> DiagramEventCollection:
        assert self.parent is None, (
            "Can only get zero-crossing events from top-level Diagram, not "
            f"{self.system_id} with parent {self.parent.system_id}"
        )
        return DiagramEventCollection(
            OrderedDict(
                {sys.system_id: sys.zero_crossing_events for sys in self.leaf_systems}
            )
        )

    # Inherits docstrings from SystemBase
    def determine_active_guards(
        self, root_context: DiagramContext
    ) -> DiagramEventCollection:
        assert self.parent is None, (
            "Can only get zero-crossing events from top-level Diagram, not "
            f"{self.system_id} with parent {self.parent.system_id}"
        )
        return DiagramEventCollection(
            OrderedDict(
                {
                    sys.system_id: sys.determine_active_guards(root_context)
                    for sys in self.leaf_systems
                }
            )
        )

    # Inherits docstrings from SystemBase
    def eval_zero_crossing_updates(
        self,
        root_context: DiagramContext,
        events: DiagramEventCollection,
    ) -> dict[Hashable, LeafState]:
        substates = OrderedDict()
        for system_id, subctx in root_context.subcontexts.items():
            sys = subctx.owning_system
            substates[system_id] = sys.eval_zero_crossing_updates(root_context, events)

        return substates

    #
    # I/O ports
    #
    @property
    def _flat_callbacks(self) -> List[SystemCallback]:
        """Return a flat list of all SystemCallbacks in the Diagram."""
        return [cb for sys in self.nodes for cb in sys._flat_callbacks]

    @property
    def exported_input_ports(self):
        return self._input_port_map

    @property
    def exported_output_ports(self):
        return self._output_port_map

    def eval_subsystem_input_port(
        self, context: DiagramContext, port_locator: InputPortLocator
    ) -> Array:
        """Evaluate the input port for a child of this system given the root context.

        Args:
            context (ContextBase): root context for this system
            port_locator (InputPortLocator): tuple of (system, port_index) identifying
                the input port to evaluate

        Returns:
            Array: Value returned from evaluating the subsystem port.

        Raises:
            InputNotConnectedError: if the input port is not connected
        """

        is_exported = port_locator in self._input_port_map
        if is_exported:
            # The upstream source is an input to this whole Diagram; evaluate that
            # input port and use the result as the value for this one.
            port_index = self._input_port_map[port_locator]  # Diagram-level index
            return self.input_ports[port_index].eval(context)  # Return upstream value

        is_connected = port_locator in self.connection_map
        if is_connected:
            # The upstream source is an output port of one of this Diagram's child
            # subsystems; evaluate the upstream output.
            upstream_locator = self.connection_map[port_locator]

            # This will return the value of the upstream port
            return self.eval_subsystem_output_port(context, upstream_locator)

        block, port_index = port_locator
        raise InputNotConnectedError(
            system=block,
            port_index=port_index,
            port_direction="in",
            message=f"Input port {block.name}[{port_index}] is not connected",
        )

    def eval_subsystem_output_port(
        self, context: DiagramContext, port_locator: OutputPortLocator
    ) -> Array:
        """ "Evaluate the output port for a child of this system given the root context.

        Args:
            context (ContextBase): root context for this system
            port_locator (OutputPortLocator): tuple of (system, port_index) identifying
                the output port to evaluate

        Returns:
            Array: Value returned from evaluating the subsystem port.
        """
        system, port_index = port_locator
        port = system.output_ports[port_index]

        # During simulation all we should need to do is evaluate the port.
        if context.is_initialized:
            return port.eval(context)

        # If the context is not initialized, we have to determine the signal data type.
        # In the easy case, the port has a default value, so we can just use that.
        if port.default_value is not None:
            logger.debug(
                "Using default output value of %s for %s",
                port.default_value,
                port_locator[0].name,
            )
            return port.default_value

        logger.debug(
            "Evaluating output port %s for system %s. Context initialized: %s",
            port_locator,
            port_locator[0].name,
            context.is_initialized,
        )

        # If there is no default value, try to evaluate the port to pull a "template"
        # value with an appropriate data type from upstream.  This will return None if
        # the port is not yet connected (e.g. if its upstream is an exported input of)
        # a Diagram, so we can defer evaluation.

        # Try again to evaluate the port
        val = port.eval(context)
        logger.debug(
            "  ---> %s returns %s", (port_locator[0].name, port_locator[1]), val
        )

        # If there is still no value, the port is not connected to anything.
        # Post-initialization this would be an error, but pre-initialization
        # it may be the case that the upstream is an exported input port of
        # the Diagram, so we can defer evaluation. Expect the block that is
        # doing this to handle the UpstreamEvalError appropriately.
        if val is None:
            system_name = system.name_path_str
            logger.debug(
                "Upstream evaluation of %s.out[%s] returned None. Deferring evaluation.",
                system_name,
                port_index,
            )
            raise UpstreamEvalError(port_locator=(system, "out", port_index))
        return val

    #
    # System-level declarations (should be done via DiagramBuilder)
    #
    def export_input(self, locator: InputPortLocator, port_name: str) -> int:
        """Export a subsystem input port as a diagram-level input.

        This should typically only be called during construction by DiagramBuilder.
        The standard workflow will be to call export_input on the _builder_ object,
        which will automatically call this method on the Diagram once created.

        Args:
            locator (InputPortLocator): tuple of (system, port_index) identifying
                the input port to export
            port_name (str): name of the new exported input port

        Returns:
            int: index of the exported input port in the diagram input_ports list
        """
        diagram_port_index = self.declare_input_port(name=port_name)
        self._input_port_map[locator] = diagram_port_index

        # Sometimes API calls will export ports manually (e.g. in the PID autotuning
        # workflow), so we need to make sure these dependencies are properly tracked.
        self.update_dependency_graph()

        return diagram_port_index

    def export_output(self, locator: OutputPortLocator, port_name: str) -> int:
        """Export a subsystem output port as a diagram-level output.

        This should typically only be called during construction by DiagramBuilder.
        The standard workflow will be to call export_input on the _builder_ object,
        which will automatically call this method on the Diagram once created.

        Args:
            locator (OutputPortLocator): tuple of (system, port_index) identifying
                the output port to export
            port_name (str): name of the new exported output port

        Returns:
            int: index of the exported output port in the diagram output_ports list
        """
        subsystem, subsystem_port_index = locator
        source_port = subsystem.output_ports[subsystem_port_index]
        diagram_port_index = self.declare_output_port(
            source_port.eval,
            name=port_name,
            prerequisites_of_calc=[source_port.ticket],
        )
        self._output_port_map[locator] = diagram_port_index
        self._inv_output_port_map[diagram_port_index] = locator

        # Sometimes API calls will export ports manually (e.g. in the PID autotuning
        # workflow), so we need to make sure these dependencies are properly tracked.
        self.update_dependency_graph()

        return diagram_port_index

    #
    # Initialization
    #
    @property
    def context_factory(self) -> DiagramContextFactory:
        return DiagramContextFactory(self)

    @property
    def dependency_graph_factory(self) -> DiagramDependencyGraphFactory:
        return DiagramDependencyGraphFactory(self)

    def initialize_static_data(self, context: DiagramContext) -> DiagramContext:
        """Perform any system-specific static analysis."""
        for system in self.nodes:
            context = system.initialize_static_data(context)
        return context

    def _has_feedthrough(self, input_port_index: int, output_port_index: int) -> bool:
        """Check if there is a direct-feedthrough path from the input port to the output port.

        Internal function used by `get_feedthrough`.  Should not typically need to
        be called directly.
        """
        # TODO: Would this be simpler if the input port map was inverted?
        input_ids = []
        for locator, index in self._input_port_map.items():
            if index == input_port_index:
                input_ids.append(locator)

        input_ids = set(input_ids)

        # Search graph for a direct-feedthrough connection from the output_port
        # to the input_port.  Maintain a set of the output port identifiers that
        # are known to have a direct-feedthrough path to the output_port
        active_set: Set[OutputPortLocator] = set()
        active_set.add(self._inv_output_port_map[output_port_index])

        while len(active_set) > 0:
            sys, sys_output = active_set.pop()
            for u, v in sys.get_feedthrough():
                if v == sys_output:
                    curr_input_id = (sys, u)
                    if curr_input_id in input_ids:
                        # Found a direct-feedthrough path to the input_port
                        return True
                    elif curr_input_id in self.connection_map:
                        # Intermediate input port has a direct-feedthrough path to
                        # output_port. Add the upstream output port (if there
                        # is one) to the active set.
                        active_set.add(self.connection_map[curr_input_id])

        # If there are no intermediate output ports with a direct-feedthrough path
        # to the output port, there is no direct feedthrough from the input port
        return False

    # Inherits docstring from SystemBase.get_feedthrough
    def get_feedthrough(self) -> List[Tuple[int, int]]:
        if self.feedthrough_pairs is not None:
            return self.feedthrough_pairs

        pairs = []
        for u in range(self.num_input_ports):
            for v in range(self.num_output_ports):
                if self._has_feedthrough(u, v):
                    pairs.append((u, v))

        self.feedthrough_pairs = pairs
        return self.feedthrough_pairs

    def find_system_with_path(self, path: str | list[str]) -> SystemBase:
        if isinstance(path, str):
            path = path.split(".")

        def _find_in_children():
            for child in self.nodes:
                if child.name == path[0]:
                    if len(path) == 1:
                        return child
                    if isinstance(child, Diagram):
                        return child.find_system_with_path(path[1:])
                    return None
            return None

        if self.parent is None:
            return _find_in_children()

        if self.name == path[0] and len(path) == 1:
            return self

        return _find_in_children()

    def declare_dynamic_parameter(self, name: str, parameter: Parameter) -> None:
        """Declare a parameter for this system.

        Parameters:
            name (str): The name of the parameter.
            parameter (Parameter): The parameter object.
        """
        # Force the parameter to have the correct name, all diagram parameters
        # should be named.
        parameter.name = name
        super().declare_dynamic_parameter(name, parameter)

    def check_no_algebraic_loops(self):
        """Check for algebraic loops in the diagram.

        This is a more or less direct port of the Drake method
        DiagramBuilder::ThrowIfAlgebraicLoopExists. Some comments are verbatim
        explanations of the algorithm implemented there.
        """

        # The nodes in the graph are the input/output ports defined as part of
        # the diagram's internal connections.  Ports that are not internally
        # connected cannot participate in a cycle at this level, so we don't include them
        # in the nodes set.
        nodes: Set[PortBase] = set()

        # For each `value` in `edges[key]`, the `key` directly influences `value`.
        edges: Mapping[PortBase, Set[PortBase]] = {}

        # Add the diagram's internal connections to the digraph nodes and edges
        for input_port_locator, output_port_locator in self.connection_map.items():
            # Directly using the port locator does not result in a unique identifier
            # since (sys, 0) represents both input port 0 and output port 0.  Instead,
            # use the port directly as a key, since it is a unique hashable object.
            input_system, input_index = input_port_locator
            input_port = input_system.input_ports[input_index]
            logger.debug(f"Adding locator {input_port} to nodes")
            nodes.add(input_port)

            output_system, output_index = output_port_locator
            output_port = output_system.output_ports[output_index]
            logger.debug(f"Adding locator {output_port} to nodes")
            nodes.add(output_port)

            if output_port not in edges:
                edges[output_port] = set()

            logger.debug(f"Adding edge[{output_port}] = {input_port}")
            edges[output_port].add(input_port)

        # Add more edges based on each System's direct feedthrough.
        # input -> output port iff there is direct feedthrough from input -> output
        # If a feedthrough edge refers to a port not in `nodes`, omit it because ports
        # that are not connected inside the diagram cannot participate in a cycle at
        # the level of this diagram (higher-level diagrams will test for cycles at
        # their level).
        for system in self.nodes:
            logger.debug(f"Checking feedthrough for system {system.name}")
            for input_index, output_index in system.get_feedthrough():
                input_port = system.input_ports[input_index]
                output_port = system.output_ports[output_index]
                logger.debug(f"Feedthrough from {input_port} to {output_port}")
                if input_port in nodes and output_port in nodes:
                    if input_port not in edges:
                        edges[input_port] = set()
                    edges[input_port].add(output_port)

        def _graph_has_cycle(
            node: PortBase,
            visited: Set[DirectedPortLocator],
            stack: List[DirectedPortLocator],
        ) -> bool:
            # Helper to do the algebraic loop test by depth-first search on the graph
            # to find cycles. Modifies `visited` and `stack` in place.

            logger.debug(f"Checking node {node}")

            assert node.directed_locator not in visited
            visited.add(node.directed_locator)

            if node in edges:
                assert node not in stack
                stack.append(node.directed_locator)
                edge_iter = edges[node]
                for target in edge_iter:
                    if target.directed_locator not in visited and _graph_has_cycle(
                        target, visited, stack
                    ):
                        logger.debug(f"Found cycle at {target}")
                        return True
                    elif target.directed_locator in stack:
                        logger.debug(f"Found target {target} in stack {stack}")
                        return True
                stack.pop()

            # If we get this far there is no cycle
            return False

        # Evaluate the graph for cycles
        visited: Set[DirectedPortLocator] = set()
        stack: List[DirectedPortLocator] = []
        for node in nodes:
            if node.directed_locator in visited:
                continue
            if _graph_has_cycle(node, visited, stack):
                raise AlgebraicLoopError(self.name, stack)
