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

"""Container classes for state information.

The only actual class definition is for LeafState, which contains:
- Continuous state: the component of state that evolves in continuous time
- Discrete state: component of the state that does not change in continuous time. Note
    That this does not necessarily take on discrete _values_ (though it could).
- Mode: an integer value indicating the current "mode", "stage", or discrete-valued
    state component of the system.  Used for finite state machines or multi-stage
    hybrid systems.

Diagram-level state information is stored in containers of LeafStates as defined
in the type hints towards the end of the file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Hashable, Mapping, Tuple, NamedTuple
import dataclasses

from ..backend import reshape
from jax import tree_util

if TYPE_CHECKING:
    from ..backend.typing import Array

    # Both continuous and discrete states are of type "LeafStateComponent".
    # Commonly this will just be an array, but other immutable pytree-structured
    # containers are also allowed.  The most common case will be a discrete
    # state with multiple components, defined as a tuple or namedtuple
    LeafStateComponent = Union[Array, Tuple[Array, ...], NamedTuple[Array, ...]]

    # Type hint for return values from SystemBase, which can either be a
    # leaf state component (from a LeafSystem), or a list of leaf state
    # components (from a Diagram).
    StateComponent = Union[LeafStateComponent, List[LeafStateComponent]]

    # The "finite state machine mode" is an integer value for a LeafSystem,
    # or a list of integer values for a Diagram.
    Mode = Union[int, List[int]]

__all__ = [
    "State",
    "LeafState",
    "LeafStateComponent",
    "StateComponent",
    "Mode",
]


@dataclasses.dataclass(frozen=True)
class LeafState:
    """Container for state information for a leaf system.

    Attributes:
        name (str):
            Name of the leaf system that owns this state.
        continuous_state (LeafStateComponent):
            Continuous state of the system, i.e. the component of state that evolves in
            continuous time. If the system has no continuous state, this will be None.
        discrete_state (LeafStateComponent):
            Discrete state of the system, i.e. one or more components of state that do
            not change continuously with ime (not necessarily discrete-_valued_). If
            the system has no discrete state, this will be None.
        mode (int):
            An integer value indicating the current "mode", "stage", or discrete-valued
            state component of the system.  Used for finite state machines or
            multi-stage hybrid systems.  If the system has no mode, this will be None.
        cache (tuple[LeafStateComponent]):
            The current values of sample-and-hold outputs from the system.  In a pure
            discrete system these would not be state components (just results of
            feedthrough computations), but in a hybrid or multirate system they act as
            discrete state from the perspective of continuous or asynchronous discrete
            components of the system.  Hence, they are stored in the state, but are
            maintained separately from the normal internal state of the system.

    Notes:
        (1) This class is immutable.  To modify a LeafState, use the `with_*` methods.

        (2) The type annotations for state components are LeafStateComponent, which is
        a union of array, tuple, and named tuple. The most common case is arrays, but
        this allows for more flexibility in defining state components, e.g. a
        second-order system can define a named tuple of generalized coordinates and
        velocities rather than concatenating into a single array.
    """

    name: str = None
    continuous_state: LeafStateComponent = None
    discrete_state: LeafStateComponent = None
    mode: int = None
    cache: tuple[Array] = None

    def __repr__(self) -> str:
        states = []
        if self.continuous_state is not None:
            states.append(f"xc={self.continuous_state}")
        if self.discrete_state is not None:
            states.append(f"xd={self.discrete_state}")
        if self.mode is not None:
            states.append(f"s={self.mode}")
        return f"{type(self).__name__}({', '.join(states)})"

    def with_continuous_state(self, value: LeafStateComponent) -> LeafState:
        """Create a copy of this LeafState with the continuous state replaced."""
        if value is not None and self.continuous_state is not None:
            value = tree_util.tree_map(self._reshape_like, value, self.continuous_state)

        return dataclasses.replace(self, continuous_state=value)

    def _component_size(self, component: LeafStateComponent) -> int:
        if component is None:
            return 0
        if isinstance(component, tuple):
            # return sum(x.size for x in component)
            return len(component)
        return component.size

    def _reshape_like(self, new_value: Array, current_value: Array) -> Array:
        """Helper function for tree-mapped type conversions.

        Ensures that the new components are array-like and have the same shape as
        the existing state to preserve PyTree structure.
        """
        return reshape(new_value, current_value.shape)

    @property
    def num_continuous_states(self) -> int:
        return self._component_size(self.continuous_state)

    @property
    def has_continuous_state(self) -> bool:
        return self.num_continuous_states > 0

    def with_discrete_state(self, value: LeafStateComponent) -> LeafState:
        """Create a copy of this LeafState with the discrete state replaced."""
        if value is not None and self.discrete_state is not None:
            value = tree_util.tree_map(self._reshape_like, value, self.discrete_state)

        return dataclasses.replace(self, discrete_state=value)

    @property
    def num_discrete_states(self) -> int:
        return self._component_size(self.discrete_state)

    @property
    def has_discrete_state(self) -> bool:
        return self.num_discrete_states > 0

    def with_mode(self, value: int) -> LeafState:
        """Create a copy of this LeafState with the mode replaced."""
        return dataclasses.replace(self, mode=value)

    @property
    def has_mode(self) -> bool:
        return self.mode is not None

    def with_cached_value(self, index: int, value: Array) -> LeafState:
        """Create a copy of this LeafState with the specified cache value replaced."""
        cache = list(self.cache)
        cache[index] = value
        return dataclasses.replace(self, cache=tuple(cache))

    def has_cache(self) -> bool:
        return self.cache is not None

    def num_cached_values(self) -> int:
        return len(self.cache)


#
# Type hints for export
#
# This is the union of the leaf state types and what would be the corresponding
# diagram-level data types.  These should be used for type hints in base classes,
# but the concrete types should be preferred in implementations where possible.
if TYPE_CHECKING:
    State = Union[LeafState, Mapping[Hashable, LeafState]]


#
# Register as custom pytree nodes
#    https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
#
# Even though this is a JAX utility, it can also be useful for generally working
# with tree-structured data, so we can also use these with other math backends.
#
def _leaf_state_flatten(state: LeafState):
    children = (
        state.continuous_state,
        state.discrete_state,
        state.mode,
        state.cache,
    )
    aux_data = (state.name,)
    return children, aux_data


def _leaf_state_unflatten(aux_data, children):
    continuous_state, discrete_state, mode, cache = children
    return LeafState(
        name=aux_data[0],
        continuous_state=continuous_state,
        discrete_state=discrete_state,
        mode=mode,
        cache=cache,
    )


tree_util.register_pytree_node(
    LeafState,
    _leaf_state_flatten,
    _leaf_state_unflatten,
)
