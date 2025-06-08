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
from . import build_recorder
from .state import LeafState
from .context import (
    ContextBase,
    LeafContext,
    DiagramContext,
)
from .event import (
    IntegerTime,
    DiscreteUpdateEvent,
    PeriodicEventData,
    ZeroCrossingEvent,
    ZeroCrossingEventData,
    EventCollection,
    LeafEventCollection,
    DiagramEventCollection,
    is_event_data,
)

from .cache import (
    SystemCallback,
)

from .system_base import SystemBase, parameters
from .leaf_system import LeafSystem
from .diagram_builder import DiagramBuilder
from .diagram import Diagram
from .parameter import Parameter, ParameterCache
from .error import (
    CollimatorError,
    StaticError,
    ShapeMismatchError,
    DtypeMismatchError,
    BlockInitializationError,
    BlockParameterError,
    BlockRuntimeError,
    ErrorCollector,
)

from .dependency_graph import (
    DependencyTicket,
    next_dependency_ticket,
)

if TYPE_CHECKING:
    from .state import State


__all__ = [
    "SystemCallback",
    "LeafState",
    "State",
    "ContextBase",
    "LeafContext",
    "DiagramContext",
    "IntegerTime",
    "DiscreteUpdateEvent",
    "PeriodicEventData",
    "ZeroCrossingEvent",
    "ZeroCrossingEventData",
    "EventCollection",
    "LeafEventCollection",
    "DiagramEventCollection",
    "is_event_data",
    "SystemBase",
    "LeafSystem",
    "DiagramBuilder",
    "Diagram",
    "StaticError",
    "CollimatorError",
    "ShapeMismatchError",
    "DtypeMismatchError",
    "BlockInitializationError",
    "BlockParameterError",
    "BlockRuntimeError",
    "ErrorCollector",
    "Parameter",
    "ParameterCache",
    "DependencyTicket",
    "next_dependency_ticket",
    "parameters",
    "build_recorder",
]
