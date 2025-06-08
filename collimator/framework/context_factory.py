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

"""Factory classes for generating context classes.

These should not be used directly by users.  Instead, use the `create_context`
method on `SystemBase` objects.  These classes are broken out because they contain
significant logic that is not relevant to the user-facing System API and is only used
for context creation.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from collections import OrderedDict

from .error import StaticError, ErrorCollector
from ..logging import logger
from .context import LeafContext, DiagramContext

if TYPE_CHECKING:
    from ..backend.typing import Scalar
    from .system_base import SystemBase
    from .context import ContextBase
    from .diagram import Diagram
    from .leaf_system import LeafSystem


class ContextFactory(metaclass=abc.ABCMeta):
    """Abstract base class for creating context objects."""

    def __init__(self, system: SystemBase):
        """Initialize the ContextFactory with the given system.

        To actually create the context, call the ContextFactory object as a function.
        This should happen automatically when calling `system.create_context`.

        Args:
            system (SystemBase):
                The system for which the context is being created.
        """
        self.system = system

    def __call__(
        self,
        time: Scalar = 0.0,
        check_types: bool = True,
        error_collector: ErrorCollector = None,
    ) -> ContextBase:
        """Create a context object for the system.

        Args:
            time (Scalar):
                The initial simulation time for the context.  Can also be used
                to determine the dtype of the time variable. Defaults to 0.0 (a float).
            check_types (bool):
                Whether to perform basic type checking.  Should only be set to True
                when creating the root context.

        Returns:
            The created context object.
        """
        logger.debug("Creating context for %s", self.system.name)
        ctx = self.create_node_context()
        ctx = ctx.with_time(time)

        # At this point the call should be "untraced", meaning we can change
        # attributes of the owning system based on the context.  This is useful
        # when the system has an internal state with dtype or shape that must be
        # determined based on what it's connected to (see `DerivativeDiscrete` block
        # for an example).
        ctx = self.initialize_static_data(ctx)

        # Optionally, perform basic type checking by evaluating all the cache sources.
        # Systems are allowed to define any implementation-specific type checking.
        # Since this won't make sense on partially-constructed digrams, it should only
        # be done for the root context.
        if check_types:
            self.check_types(ctx, error_collector=error_collector)

        ctx = ctx.mark_initialized()
        return ctx

    @abc.abstractmethod
    def create_node_context(self) -> ContextBase:
        """Create a generic context for this system

        This will not perform any type checking or static analysis, but does initialize
        state and parameters, cache, etc.

        Returns:
            The created node context.
        """
        pass

    def initialize_static_data(self, root_context: ContextBase) -> ContextBase:
        """Perform any system-specific static analysis.

        For example, this could be used to determine the dtype of a system's internal
        state based on the types of its inputs.  This is called after the context is
        created, but before any type checking is performed.
        """
        return self.system.initialize_static_data(root_context)

    def check_types(
        self, root_context: ContextBase, error_collector: ErrorCollector = None
    ):
        """Perform basic type checking by evaluating cache sources.

        All cache sources (ports, ODE right-hand-side functions, etc) are evaluated
        to ensure that there are no shape mismatches, etc.  Sources with default values
        are skipped.  If any exceptions arise, they are caught and re-raised as a
        StaticError.

        Args:
            root_context: The root context object.

        Raises:
            StaticError: If any exceptions are raised during type checking.
        """
        system = self.system
        logger.debug("Cache sources: %s", system.callbacks)

        # Do any system-specific type checking
        system.check_types(root_context, error_collector=error_collector)

        for source in system.callbacks:
            if source.default_value is None:
                value = None
                try:
                    value = source.eval(root_context)

                    if value is None:
                        logger.error(
                            "Cache source %s was not initialized by source.eval", source
                        )

                except StaticError as exc:
                    with ErrorCollector.context(error_collector):
                        raise exc
                except BaseException as exc:
                    with ErrorCollector.context(error_collector):
                        raise StaticError(system=system) from exc

                logger.debug(
                    "    ---> check_types for %s returns %s with value: %s",
                    source,
                    root_context,
                    value,
                )


class LeafContextFactory(ContextFactory):
    """ContextFactory for leaf systems."""

    # Inherit docstring from ContextFactory
    def create_node_context(self) -> LeafContext:
        system: LeafSystem = self.system

        try:
            system.initialize(**system.parameters)
        except BaseException as exc:
            raise StaticError(system=system) from exc

        # Make sure the child system has a dependency graph so that we can
        # subscribe trackers to its ports and callbacks.  If the system already
        # has a dependency graph, this will do nothing.
        system.create_dependency_graph()

        dynamic_parameters = {}
        for name, param in system.dynamic_parameters.items():
            dynamic_parameters[name] = param.get()

        state = system.create_state()

        logger.debug(
            "Created state %s and parameters %s for %s",
            state,
            system.parameters,
            system.name,
        )

        return LeafContext(
            owning_system=system,
            state=state,
            parameters=dynamic_parameters,
        )


class DiagramContextFactory(ContextFactory):
    """ContextFactory for diagram systems."""

    # Inherit docstring from ContextFactory
    def create_node_context(self) -> DiagramContext:
        system: Diagram = self.system

        subcontext_nodes = [
            sys.context_factory.create_node_context() for sys in system.nodes
        ]

        system.check_no_algebraic_loops()

        # Create the dependency graph for the diagram
        system.create_dependency_graph()

        subcontexts = OrderedDict()
        for ctx in subcontext_nodes:
            if isinstance(ctx, DiagramContext):
                subcontexts.update(ctx.subcontexts)
            else:
                subcontexts[ctx.owning_system.system_id] = ctx

        parameters = {}
        for name, param in system.parameters.items():
            parameters[name] = param.get()

        return DiagramContext(
            owning_system=system,
            subcontexts=subcontexts,
            parameters=parameters,
        )

    # Inherit docstring from ContextFactory
    def initialize_static_data(self, root_context: DiagramContext) -> DiagramContext:
        system: Diagram = self.system

        # First sort the systems according to whether they have cached computations.
        # These need to be done in the appropriate order.
        # Multiple events may refer to the same system, we need to keep the order
        # with regards to the last occuring event of a system.
        nodes = []
        for cb in reversed(system.sorted_event_callbacks):
            if cb.system not in nodes:
                nodes.append(cb.system)
        nodes = list(reversed(nodes))

        for sys in system.nodes:
            if sys not in nodes:
                nodes.append(sys)

        for sys in nodes:
            root_context = sys.context_factory.initialize_static_data(root_context)
        return super().initialize_static_data(root_context)

    # Inherit docstring from ContextFactory
    def check_types(
        self, root_context: DiagramContext, error_collector: ErrorCollector = None
    ):
        system: Diagram = self.system
        for sys in system.nodes:
            logger.debug("**** Initializing cache for %s ****", sys.name)
            sys.context_factory.check_types(
                root_context,
                error_collector=error_collector,
            )
        super().check_types(
            root_context,
            error_collector=error_collector,
        )
