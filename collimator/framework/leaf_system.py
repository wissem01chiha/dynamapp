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

"""Self-contained systems with no subsystems.

A LeafSystem is a minimal component of a system model in collimator, containing no
subsystems.  If it has inputs or outputs it can be connected to other LeafSystems
as "blocks" to form Diagrams.  If not, it is a self-contained dynamical system.

The LeafSystem class defines the interface for these blocks, specifying various
options for configuring the block.  For example, a LeafSystem can declare that
it has a continuous state and provide an ODE function governing the time evolution
of that state.  It can also declare discrete states, parameters, and update events.
The built-in blocks in collimator.library are all subclasses of LeafSystem, as are
any custom blocks defined by the user.

After declaring states, parameters, ODE, updates, etc., the LeafSystem comprises a set
of pure functions that can be evaluated given a Context, which contains the actual
numeric values for the states, parameters, etc.
"""

from __future__ import annotations
from abc import ABCMeta
from functools import partial, wraps
from typing import List, Set, Tuple, Callable, TYPE_CHECKING

import numpy as np

import jax
import jax.numpy as jnp
from jax import tree_util

from . import build_recorder

from ..logging import logger

# Import the switchable backend dispatcher as "collimator.numpy" or "cnp"
from ..backend import utils, cond, numpy_api as cnp, IS_JAXLITE

from .cache import SystemCallback, CallbackTracer
from .state import LeafState
from .port import OutputPort

from .system_base import SystemBase, UpstreamEvalError
from .context_factory import LeafContextFactory
from .dependency_graph import (
    mark_cache,
    LeafDependencyGraphFactory,
    DependencyTicket,
)

from .event import (
    FlatEventCollection,
    LeafEventCollection,
    PeriodicEventData,
    DiscreteUpdateEvent,
    ZeroCrossingEvent,
    ZeroCrossingEventData,
)
from .parameter import Parameter

if TYPE_CHECKING:
    from ..backend.typing import (
        Array,
        Scalar,
        ShapeLike,
        DTypeLike,
    )
    from .context import ContextBase, LeafContext
    from .state import LeafStateComponent

__all__ = ["LeafSystem"]


# Helper functons used in feedthrough determination
_mark_up_to_date = partial(mark_cache, is_out_of_date=False)
_mark_out_of_date = partial(mark_cache, is_out_of_date=True)


def _resolve_params(func):
    def wrapper(*args, **kwargs):
        args = [arg.get() if isinstance(arg, Parameter) else arg for arg in args]
        kwargs = {
            k: v.get() if isinstance(v, Parameter) else v for k, v in kwargs.items()
        }

        result = func(*args, **kwargs)
        return result

    return wrapper


class InitializeParameterResolver(ABCMeta):
    """Wrapper for the LeafSystem for proper handling of parameters.

    1) wraps initialize() method such that parameters are always resolved when
    the function is called.
    2) automatically call initialize() after __init__.
    """

    def __new__(cls, name, bases, dct):
        if "initialize" in dct:
            orig_initialize = dct["initialize"]
            dct["initialize"] = _resolve_params(orig_initialize)

            if "__init__" in dct:
                orig_init = dct["__init__"]

                @wraps(orig_init)
                def new_init(self, *args, **kwargs):
                    orig_init(self, *args, **kwargs)
                    build_recorder.create_block(self, orig_init, *args, **kwargs)

                    # Only call initialize on the actual class, not on the base class
                    if type(self).__name__ == name:
                        self.initialize(**self.parameters)

                dct["__init__"] = new_init

        return super().__new__(cls, name, bases, dct)


class LeafSystem(SystemBase, metaclass=InitializeParameterResolver):
    """Basic building block for dynamical systems.

    A LeafSystem is a minimal component of a system model in collimator, containing no
    subsystems.  Inputs, outputs, state, parameters, updates, etc. can be added to the
    block using the various `declare_*` methods.  The built-in blocks in
    collimator.library are all subclasses of LeafSystem, as are any custom blocks defined
    by the user."""

    # SystemBase is a dataclass, so we need to call __post_init__ explicitly
    def __post_init__(self):
        super().__post_init__()
        logger.debug(f"Initializing {self.name} [{self.system_id}]")

        # If not None, this defines the shape and data type of the continuous state
        # component.  This value will be used to initialize the context, so it will
        # also serve as the initial value unless explicitly overridden. It will
        # typically be an array, but it can be any PyTree-structured object (list,
        # dict, namedtuple, etc.), provided that the ODE function returns a PyTree
        # of the same structure.
        self._default_continuous_state: LeafStateComponent = None
        self._mass_matrix: Array = None
        self._continuous_state_output_port_idx: int = None

        # The SystemCallback associated with time derivatives of the continuous state.
        # This is initialized in the `declare_continuous_state` method.
        self.ode_callback: SystemCallback = None

        # If not empty, this defines the shape and data type of the discrete state.
        # This value will be used to initialize the context, so it will also serve
        # as the initial value unless explicitly overridden. This will often be an
        # array, but as for the continuous state it can be any PyTree-structured
        # object (list, dict, namedtuple, etc.), provided that the update functions
        # return a PyTree of the same structure.
        self._default_discrete_state: LeafStateComponent = None

        # If not None, the system has a "mode" or "stage" component of the state.
        # In a "state machine" paradigm, this represents the current state of the
        # system (although "state" is obviously used for other things in this case).
        # The mode is an integer value, and the system can declare transitions between
        # modes using the `declare_zero_crossing` method, which in addition to the
        # guard function and reset map also takes optional `start_mode` and `end_mode`
        # arguments.
        self._default_mode: int = None

        # Set of "template" values for the sample-and-hold output ports, if known.
        # If not known, these will be `None`, in which case an appropriate value is
        # inferred from upstream during static analysis.
        self._default_cache: List[LeafStateComponent] = []

        # Transition map from (start_mode -> [*end_modes]) indicating which
        # transition events are active in each mode.  This is not used by
        # any logic in the system, but can be useful for debugging.
        self.transition_map: dict[int, List[Tuple[int, ZeroCrossingEvent]]] = {}

        # Set of events that updates at a fixed rate.  Each event has its own period
        # and offset, so "fires" independently of the other events. These can be
        # created using the `declare_periodic_update` method.
        self._state_update_events: List[DiscreteUpdateEvent] = []

        # Set of events that update when a zero-crossing occurs.  Each event has its
        # own guard function and, optionally, reset map, start mode, and end mode.
        # These can be created using the `declare_zero_crossing` method.
        self._zero_crossing_events: List[ZeroCrossingEvent] = []

    def initialize(self, **parameters):
        """Hook for initializing a system. Called during context creation.

        If the parameters are instances of Parameter, they will be resolved.
        If implemented, the function signature should contain all the declared
        parameters.

        This function should not be called directly. It will be called implicitly
        after __init__ with the resolved parameters.
        """
        pass

    @property
    def has_feedthrough_side_effects(self) -> bool:
        # See explanation in `SystemBase.has_feedthrough_side_effects`.  This will
        # almost always be False, but can be overridden in special cases where a
        # feedthrough output is computed via use of `io_callback`.
        return False

    @property
    def has_ode_side_effects(self) -> bool:
        # This will almost always be False for a LeafSystem - Diagram systems
        # have some special logic to do this determination.
        return False

    @property
    def has_continuous_state(self) -> bool:
        return self._default_continuous_state is not None

    @property
    def has_discrete_state(self) -> bool:
        return self._default_discrete_state is not None

    @property
    def has_zero_crossing_events(self) -> bool:
        return len(self._zero_crossing_events) > 0

    #
    # Event handling
    #
    def wrap_callback(
        self, callback: Callable, collect_inputs: bool = True
    ) -> Callable:
        """Wrap an update function to unpack local variables and block inputs.

        The callback should have the signature
        `callback(time, state, *inputs, **params) -> result`
        and will be wrapped to have the signature `callback(context) -> result`,
        as expected by the event handling logic.

        This is used internally for declaration methods like
        `declare_periodic_update` so that users can write more intuitive
        block-level update functions without worrying about the "context", and have
        them automatically wrapped to have the right interface.  It can also be
        called directly by users to wrap their own update functions, for example to
        create a callback function for `declare_output_port`.

        The context and state are strictly immutable, so the callback should not
        attempt to change any values in the context or state.  Even in cases where
        it is impossible to _enforce_ this (e.g. a state component is a list, which
        is always mutable in Python), the callback should be careful to avoid direct
        modification of the context or state, which may lead to unexpected behavior
        or JAX tracer errors.

        Args:
            callback (Callable):
                The (pure) function to be wrapped. See above for expected signature.
            collect_inputs (bool):
                If True, the callback will eval input ports to gather input values.
                Normally this should be True, but it can be set to False if the
                return value depends only on the state but not inputs, for
                instance. This helps reduce the number of expressions that need to
                be JIT compiled. Default is True.

        Returns:
            Callable:
                The wrapped function, with signature `callback(context) -> result`.
        """

        def _wrapped_callback(context: ContextBase) -> LeafStateComponent:
            if collect_inputs:
                inputs = self.collect_inputs(context)
            else:
                inputs = ()
            leaf_context: LeafContext = context[self.system_id]

            leaf_state = leaf_context.state
            params = leaf_context.parameters
            return callback(context.time, leaf_state, *inputs, **params)

        return _wrapped_callback

    def _passthrough(self, context: ContextBase) -> LeafState:
        """Dummy callback for inactive events."""
        return context[self.system_id].state

    @property
    def state_update_events(self) -> FlatEventCollection:
        return FlatEventCollection(tuple(self._state_update_events))

    @property
    def zero_crossing_events(self) -> LeafEventCollection:
        # The default is for all to be active. Use the `determine_active_guards`
        # method to determine which are active conditioned on the current "mode"
        # or "stage" of the system.
        return LeafEventCollection(tuple(self._zero_crossing_events)).mark_all_active()

    # Inherits docstring from SystemBase
    def eval_zero_crossing_updates(
        self,
        context: ContextBase,
        events: LeafEventCollection,
    ) -> LeafState:
        local_events = events[self.system_id]
        state = context[self.system_id].state

        logger.debug(f"Eval update events for {self.name}")
        logger.debug(f"local events: {local_events}")

        for event in local_events:
            # This is evaluated conditionally on event_data.active
            state = event.handle(context)

            # Store the updated state in the context for this block
            leaf_context = context[self.system_id].with_state(state)

            # Update the context for this block in the overall context
            context = context.with_subcontext(self.system_id, leaf_context)

        # Now `context` contains the updated "plus" state for this block, but
        # this needs to be discarded so that other block updates can also be
        # processed using the "minus" state. This is done by simply returning the
        # "plus" state and discarding the rest of the updated context.
        return state

    # Inherits docstring from SystemBase
    def determine_active_guards(self, root_context: ContextBase) -> LeafEventCollection:
        mode = root_context[self.system_id].mode  # Current system mode

        def _conditionally_activate(
            event: ZeroCrossingEvent,
        ) -> ZeroCrossingEvent:
            # Check to see if the event corresponds to a mode transition
            # If not, just return the event unchanged (will be active)
            if event.active_mode is None:
                return event
            # If the event does correspond to a mode transition, check to see
            # if the event is active in the current mode
            return cond(
                mode == event.active_mode,
                lambda e: e.mark_active(),
                lambda e: e.mark_inactive(),
                event,
            )

        # Apply the conditional activation to all events
        zero_crossing_events = LeafEventCollection(
            tuple(_conditionally_activate(e) for e in self.zero_crossing_events)
        )

        logger.debug(f"Zero-crossing events for {self.name}: {zero_crossing_events}")
        return zero_crossing_events

    @property
    def _flat_callbacks(self) -> List[OutputPort]:
        """Return all of the sample-and-hold output ports in this system."""
        return self.callbacks

    def declare_cache(
        self,
        callback: Callable,
        period: float | Parameter = None,
        offset: float | Parameter = 0.0,
        name: str = None,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array = None,
        requires_inputs: bool = True,
    ) -> int:
        """Declare a stored computation for the system.

        This method accepts a callback function with the block-level signature
            `callback(time, state, *inputs, **parameters) -> value`
        and wraps it to have the signature
            `callback(context) -> value`

        This callback can optionally be used to define a periodic update event that
        refreshes the cached value.  Other calculations (e.g. sample-and-hold output
        ports) can then depend on the cached value.

        Args:
            callback (Callable):
                The callback function defining the cached computation.
            period (float, optional):
                If not None, the callback function will be used to define a periodic
                update event that refreshes the value. Defaults to None.
            offset (float, optional):
                The offset of the periodic update event. Defaults to 0.0.  Will be ignored
                unless `period` is not None.
            name (str, optional):
                The name of the cached value. Defaults to None.
            default_value (Array, optional):
                The default value of the result, if known. Defaults to None.
            requires_inputs (bool, optional):
                If True, the callback will eval input ports to gather input values.
                This will add a bit to compile time, so setting to False where possible
                is recommended. Defaults to True.
            prerequisites_of_calc (List[DependencyTicket], optional):
                The dependency tickets for the computation. Defaults to None, in which
                case the default is to assume dependency on either (inputs) if
                `requires_inputs` is True, or (nothing) otherwise.

        Returns:
            int: The index of the callback in `system.callbacks`.  The cache index can
                recovered from `system.callbacks[callback_index].cache_index`.
        """
        # The index in the list of system callbacks
        callback_index = len(self.callbacks)

        # This is the index that this cached value will have in state.cache
        cache_index = len(self._default_cache)
        self._default_cache.append(default_value)

        # To help avoid unnecessary flagging of algebraic loops, trim the inputs as a
        # default prereq if the update callback doesn't use them
        if prerequisites_of_calc is None:
            if requires_inputs:
                prerequisites_of_calc = [DependencyTicket.u]
            else:
                prerequisites_of_calc = [DependencyTicket.nothing]

        def _update_callback(
            time: Scalar, state: LeafState, *inputs, **parameters
        ) -> LeafState:
            output = callback(time, state, *inputs, **parameters)
            return state.with_cached_value(cache_index, output)

        _update_callback = self.wrap_callback(
            _update_callback, collect_inputs=requires_inputs
        )

        if period is None:
            event = None

        else:
            # The cache has a periodic event updating its value defined by the callback
            event = DiscreteUpdateEvent(
                system_id=self.system_id,
                event_data=PeriodicEventData(
                    period=period, offset=offset, active=False
                ),
                name=f"{self.name}:cache_update_{cache_index}_",
                callback=_update_callback,
                passthrough=self._passthrough,
            )

        if name is None:
            name = f"cache_{cache_index}"

        sys_callback = SystemCallback(
            callback=_update_callback,
            system=self,
            callback_index=callback_index,
            name=name,
            prerequisites_of_calc=prerequisites_of_calc,
            event=event,
            default_value=default_value,
            cache_index=cache_index,
        )
        self.callbacks.append(sys_callback)

        return callback_index

    def declare_continuous_state(
        self,
        shape: ShapeLike = None,
        default_value: Array = None,
        dtype: DTypeLike = None,
        ode: Callable = None,
        mass_matrix: Array = None,
        as_array: bool = True,
        requires_inputs: bool = True,
        prerequisites_of_calc: List[DependencyTicket] = None,
    ):
        """Declare a continuous state component for the system."""

        self.ode_callback = SystemCallback(
            callback=None,
            system=self,
            callback_index=len(self.callbacks),
            name=f"{self.name}_ode",
            prerequisites_of_calc=prerequisites_of_calc,
        )
        self.callbacks.append(self.ode_callback)
        callback_idx = len(self.callbacks) - 1

        # FIXME: this is to preserve some backward compatibility while we decouple
        # declaration from configuration. Declaration should not have to call
        # configuration.
        if default_value is not None or shape is not None:
            self.configure_continuous_state(
                callback_idx,
                shape=shape,
                default_value=default_value,
                dtype=dtype,
                ode=ode,
                mass_matrix=mass_matrix,
                as_array=as_array,
                requires_inputs=requires_inputs,
                prerequisites_of_calc=prerequisites_of_calc,
            )

        return callback_idx

    def configure_continuous_state(
        self,
        callback_idx: int,
        shape: ShapeLike = None,
        default_value: Array = None,
        dtype: DTypeLike = None,
        ode: Callable = None,
        mass_matrix: Array = None,
        as_array: bool = True,
        requires_inputs: bool = True,
        prerequisites_of_calc: List[DependencyTicket] = None,
    ):
        """Configure a continuous state component for the system.

        The `ode` callback computes the time derivative of the continuous state based on the
        current time, state, and any additional inputs. If `ode` is not provided, a default
        zero vector of the same size as the continuous state is used. If provided, the `ode`
        callback should have the signature `ode(time, state, *inputs, **params) -> xcdot`.

        Args:
            callback_idx (int):
                The index of the callback in the system's callback list.
            shape (ShapeLike, optional):
                The shape of the continuous state vector. Defaults to None.
            default_value (Array, optional):
                The initial value of the continuous state vector. Defaults to None.
            dtype (DTypeLike, optional):
                The data type of the continuous state vector. Defaults to None.
            ode (Callable, optional):
                The callback for computing the time derivative of the continuous state.
                Should have the signature:
                    `ode(time, state, *inputs, **parameters) -> xcdot`.
                Defaults to None.
            mass_matrix (Array, optional):
                The mass matrix for the continuous state. Defaults to None. If
                provided, must be a square matrix with the same shape as the
                continuous state.  Using a mass matrix different from the identity
                in any LeafSystem will require the use of a compatible continuous-time
                solver (currently only BDF is supported).  Currently mass matrices are
                also only supported for scalar- or vector-valued continuous states (
                i.e. no matrices or other PyTree-structured states).
            as_array (bool, optional):
                If True, treat the default_value as an array-like (cast if necessary).
                Otherwise, it will be stored as the default state without modification.
            requires_inputs (bool, optional):
                If True, indicates that the ODE computation requires inputs.
            prerequisites_of_calc (List[DependencyTicket], optional):
                The dependency tickets for the ODE computation. Defaults to None, in
                which case the assumption is a dependency on either (time, continuous
                state) if `requires_inputs` is False, otherwise (time, continuous state,
                inputs.

        Raises:
            AssertionError:
                If neither shape nor default_value is provided, or if the mass matrix
                is inconsistent with the continuous state.

        Notes:
            (1) Only one of `shape` and `default_value` should be provided. If `default_value`
            is provided, it will be used as the initial value of the continuous state. If
            `shape` is provided, the initial value will be a zero vector of the given shape
            and specified dtype.
        """

        if prerequisites_of_calc is None:
            prerequisites_of_calc = [DependencyTicket.time, DependencyTicket.xc]
            if requires_inputs:
                prerequisites_of_calc.append(DependencyTicket.u)

        if as_array:
            default_value = utils.make_array(default_value, dtype=dtype, shape=shape)

        logger.debug(f"In block {self.name} [{self.system_id}]: {default_value=}")

        # Tree-map the default value to ensure that it is an array-like with the
        # correct shape and dtype. This is necessary because the default value
        # may be a list, tuple, or other PyTree-structured object.
        default_value = tree_util.tree_map(cnp.asarray, default_value)

        self._default_continuous_state = default_value
        if self._continuous_state_output_port_idx is not None:
            port = self.output_ports[self._continuous_state_output_port_idx]
            port.default_value = default_value
            self._default_cache[port.cache_index] = default_value

        if ode is None:
            # If no ODE is specified, return a zero vector of the same size as the
            # continuous state. This will break if the continuous state is
            # a named tuple, in which case a custom ODE must be provided.
            assert as_array, "Must provide custom ODE for non-array continuous state"

            def ode(time, state, *inputs, **parameters):
                return cnp.zeros_like(default_value)

        # Wrap the ode function to accept a context and return the time derivatives.
        ode = self.wrap_callback(ode)

        # Declare the time derivative function as a system callback so that its
        # dependencies can be tracked in the system dependency graph
        self.ode_callback._callback = ode
        self.ode_callback.prerequisites_of_calc = prerequisites_of_calc

        # Override the default `eval_time_derivatives` to use the wrapped ODE function
        self.eval_time_derivatives = self.ode_callback.eval

        if mass_matrix is not None:
            # Check that the state is a vector or scalar
            assert as_array, "Mass matrix only supported for array-valued states"
            assert (
                len(default_value.shape) <= 1
            ), "Mass matrix only supported for scalar or vector continuous states"
            n = default_value.size
            assert mass_matrix.shape in ((n, n), (n,)), (
                "Mass matrix must be either a square matrix or vector of the same "
                f"size as the continuous state, but got {mass_matrix.shape} for "
                f"continuous state of shape {default_value.shape}."
            )
            if len(mass_matrix.shape) == 1:
                mass_matrix = np.diag(mass_matrix)
            else:
                mass_matrix = np.asarray(mass_matrix)

            # If we end up with an identity matrix, we can just ignore the mass
            # matrix and use the default mass matrix (which is None).  This will
            # allow us to continue using explicit ODE solvers.
            nontrivial_mass_matrix = not np.allclose(mass_matrix, np.eye(n))
            if not nontrivial_mass_matrix:
                mass_matrix = None

        self._mass_matrix = mass_matrix

    @property
    def mass_matrix(self) -> Array:
        # When this is called, an array return value is expected, so we can safely
        # return the mass matrix as an array, even if the internal value is None.
        if self._default_continuous_state is None:
            return None

        if self._mass_matrix is not None:
            return self._mass_matrix

        # Currently only scalar- or vector-valued continuous states are supported,
        # so check that the continuous state (or all tree leaves if tree-structured)
        # is a scalar or vector, and return corresponding identity matrices.
        xc_leaves = tree_util.tree_leaves(self._default_continuous_state)
        if not all(len(xc.shape) <= 1 for xc in xc_leaves):
            raise ValueError(
                "Mass matrix DAEs are only supported when the continuous state is "
                f"scalar- or vector-valued.  System {self.name} has non-vector "
                "continuous state with default value "
                f"{self._default_continuous_state}."
            )

        # Now we are guaranteed that the continuous state is a scalar or vector, so
        # we can return the corresponding (tree-structured) identity matrix.
        return jax.tree.map(lambda x: np.eye(x.size), self._default_continuous_state)

    @property
    def has_mass_matrix(self) -> bool:
        # Does the system have a nontrivial mass matrix?  This will return
        # False if the mass matrix is None or the identity matrix, since
        # the internal _mass_matrix attribute is set to None during
        # continuous state creation in the case where the mass matrix is
        # the identity.
        return self._mass_matrix is not None

    def declare_discrete_state(
        self,
        shape: ShapeLike = None,
        default_value: Array | Parameter = None,
        dtype: DTypeLike = None,
        as_array: bool = True,
    ):
        """Declare a new discrete state component for the system.

        The discrete state is a component of the system's state that can be updated
        at specific events, such as zero-crossings or periodic updates. Multiple
        discrete states can be declared, and each is associated with a unique index.
        The index is used to access and update the corresponding discrete state in
        the system's context during event handling.

        The declared discrete state is initialized with either the provided default
        value or zeros of the correct shape and dtype.

        Args:
            shape (ShapeLike, optional):
                The shape of the discrete state. Defaults to None.
            default_value (Array, optional):
                The initial value of the discrete state. Defaults to None.
            dtype (DTypeLike, optional):
                The data type of the discrete state. Defaults to None.
            as_array (bool, optional):
                If True, treat the default_value as an array-like (cast if necessary).
                Otherwise, it will be stored as the default state without modification.

        Raises:
            AssertionError:
                If as_array is True and neither shape nor default_value is provided.

        Notes:
            (1) Only one of `shape` and `default_value` should be provided. If
            `default_value` is provided, it will be used as the initial value of the
            continuous state. If `shape` is provided, the initial value will be a
            zero vector of the given shape and specified dtype.

            (2) Use `declare_periodic_update` to declare an update event that
            modifies the discrete state at a recurring interval.
        """
        if as_array:
            default_value = utils.make_array(default_value, dtype=dtype, shape=shape)

        # Tree-map the default value to ensure that it is an array-like with the
        # correct shape and dtype. This is necessary because the default value
        # may be a list, tuple, or other PyTree-structured object.
        default_value = tree_util.tree_map(cnp.asarray, default_value)

        self._default_discrete_state = default_value

    #
    # I/O declaration
    #
    def declare_output_port(
        self,
        callback: Callable = None,
        period: float | Parameter = None,
        offset: float | Parameter = 0.0,
        name: str = None,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array | Parameter = None,
        requires_inputs: bool = True,
    ) -> int:
        """Declare an output port in the LeafSystem.

        This method accepts a callback function with the block-level signature
            `callback(time, state, *inputs, **parameters) -> value`
        and wraps it to the signature expected by SystemBase.declare_output_port:
            `callback(context) -> value`

        Args:
            callback (Callable):
                The callback function defining the output port.
            period (float, optional):
                If not None, the port will act as a "sample-and-hold", with the
                callback function used to define a periodic update event that refreshes
                the value that will be returned by the port. Typically this should
                match the update period of some associated update event in the system.
                Defaults to None.
            offset (float, optional):
                The offset of the periodic update event. Defaults to 0.0.  Will be ignored
                unless `period` is not None.
            name (str, optional):
                The name of the output port. Defaults to None.
            default_value (Array, optional):
                The default value of the output port, if known. Defaults to None.
            requires_inputs (bool, optional):
                If True, the callback will eval input ports to gather input values.
                This will add a bit to compile time, so setting to False where possible
                is recommended. Defaults to True.
            prerequisites_of_calc (List[DependencyTicket], optional):
                The dependency tickets for the output port computation.  Defaults to
                None, in which case the assumption is a dependency on either (nothing)
                if `requires_inputs` is False otherwise (inputs).

        Returns:
            int: The index of the declared output port.
        """

        if default_value is not None:
            default_value = cnp.array(default_value)

        cache_index = None
        if period is not None:
            # The output port will be of "sample-and-hold" type, so we have to declare a
            # periodic event to update the value.  The callback will be used to define the
            # update event, and the output callback will simply return the stored value.

            # This is the index that this port value will have in state.cache
            cache_index = len(self._default_cache)
            self._default_cache.append(default_value)

        output_port_idx = super().declare_output_port(
            callback, name=name, cache_index=cache_index
        )

        self.configure_output_port(
            output_port_idx,
            callback,
            period=period,
            offset=offset,
            prerequisites_of_calc=prerequisites_of_calc,
            default_value=default_value,
            requires_inputs=requires_inputs,
        )

        return output_port_idx

    def configure_output_port(
        self,
        port_index: int,
        callback: Callable,
        period: float | Parameter = None,
        offset: float | Parameter = 0.0,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array | Parameter = None,
        requires_inputs: bool = True,
    ):
        """Configure an output port in the LeafSystem.

        See `declare_output_port` for a description of the arguments.

        Args:
            port_index (int):
                The index of the output port to configure.

        Returns:
            None
        """
        if default_value is not None:
            default_value = cnp.array(default_value)

        # To help avoid unnecessary flagging of algebraic loops, trim the inputs as a
        # default prereq if the output callback doesn't use them
        if prerequisites_of_calc is None:
            if requires_inputs:
                prerequisites_of_calc = [DependencyTicket.u]
            else:
                prerequisites_of_calc = [DependencyTicket.nothing]

        if period is None:
            event = None
            _output_callback = self.wrap_callback(
                callback, collect_inputs=requires_inputs
            )
            cache_index = None

        else:
            # The output port will be of "sample-and-hold" type, so we have to declare a
            # periodic event to update the value.  The callback will be used to define the
            # update event, and the output callback will simply return the stored value.

            # This is the index that this port value will have in state.cache
            cache_index = self.output_ports[port_index].cache_index
            if cache_index is None:
                cache_index = len(self._default_cache)
                self._default_cache.append(default_value)

            def _output_callback(context: ContextBase) -> Array:
                state = context[self.system_id].state
                return state.cache[cache_index]

            def _update_callback(
                time: Scalar, state: LeafState, *inputs, **parameters
            ) -> LeafState:
                output = callback(time, state, *inputs, **parameters)
                return state.with_cached_value(cache_index, output)

            _update_callback = self.wrap_callback(
                _update_callback, collect_inputs=requires_inputs
            )

            # Create the associated update event
            event = DiscreteUpdateEvent(
                system_id=self.system_id,
                event_data=PeriodicEventData(
                    period=period, offset=offset, active=False
                ),
                name=f"{self.name}:output_{cache_index}",
                callback=_update_callback,
                passthrough=self._passthrough,
            )

            # Note that in this case the "prerequisites of calc" will correspond to the
            # prerequisites of the update event, not the literal output callback itself.
            # However, these can be used to determine dependencies for the update event
            # via the output port.

        super().configure_output_port(
            port_index,
            _output_callback,
            prerequisites_of_calc=prerequisites_of_calc,
            default_value=default_value,
            event=event,
            cache_index=cache_index,
        )

    def declare_continuous_state_output(
        self,
        name: str = None,
    ) -> int:
        """Declare a continuous state output port in the system.

        This method creates a new block-level output port which returns the full
        continuous state of the system.

        Args:
            name (str, optional):
                The name of the output port. Defaults to None (autogenerate name).

        Returns:
            int: The index of the new output port.
        """
        if self._continuous_state_output_port_idx is not None:
            raise ValueError("Continuous state output port already declared")

        def _callback(time: Scalar, state: LeafState, *inputs, **parameters):
            return state.continuous_state

        self._continuous_state_output_port_idx = self.declare_output_port(
            _callback,
            name=name,
            prerequisites_of_calc=[DependencyTicket.xc],
            default_value=self._default_continuous_state,
            requires_inputs=False,
        )
        return self._continuous_state_output_port_idx

    def declare_mode_output(self, name: str = None) -> int:
        """Declare a mode output port in the system.

        This method creates a new block-level output port which returns the component
        of the system's state corresponding to the discrete "mode" or "stage".

        Args:
            name (str, optional):
                The name of the output port. Defaults to None.

        Returns:
            int:
                The index of the declared mode output port.
        """

        def _callback(time: Scalar, state: LeafState, *inputs, **parameters):
            return state.mode

        return self.declare_output_port(
            _callback,
            name=name,
            prerequisites_of_calc=[DependencyTicket.mode],
            default_value=self._default_mode,
            requires_inputs=False,
        )

    #
    # Event declaration
    #
    def declare_periodic_update(
        self,
        callback: Callable = None,
        period: Scalar | Parameter = None,
        offset: Scalar | Parameter = None,
        enable_tracing: bool = None,
    ):
        self._state_update_events.append(None)
        event_idx = len(self._state_update_events) - 1

        # FIXME: this is to preserve some backward compatibility while we decouple
        # declaration from configuration. Declaration should not have to call
        # configuration.
        if callback is not None:
            self.configure_periodic_update(
                event_idx,
                callback,
                period,
                offset,
                enable_tracing=enable_tracing,
            )
        return event_idx

    def configure_periodic_update(
        self,
        event_index: int,
        callback: Callable,
        period: Scalar | Parameter,
        offset: Scalar | Parameter,
        enable_tracing: bool = None,
    ):
        """Configure an existing periodic update event.

        The event will be triggered at regular intervals defined by the period and
        offset parameters. The callback should have the signature
        `callback(time, state, *inputs, **params) -> xd_plus`, where `xd_plus` is the
        updated value of the discrete state.

        This callback should be written to compute the "plus" value of the discrete
        state component given the "minus" values of all state components and inputs.

        Args:
            event_index (int):
                The index of the event to configure.
            callback (Callable):
                The callback function defining the update.
            period (Scalar):
                The period at which the update event occurs.
            offset (Scalar):
                The offset at which the first occurrence of the event is triggered.
            enable_tracing (bool, optional):
                If True, enable tracing for this event. Defaults to None.
        """
        _wrapped_callback = self.wrap_callback(callback)

        def _callback(context: ContextBase) -> LeafState:
            xd = _wrapped_callback(context)
            return context[self.system_id].state.with_discrete_state(xd)

        if enable_tracing is None:
            enable_tracing = True

        event = DiscreteUpdateEvent(
            system_id=self.system_id,
            name=f"{self.name}:periodic_update",
            event_data=PeriodicEventData(period=period, offset=offset, active=False),
            callback=_callback,
            passthrough=self._passthrough,
            enable_tracing=enable_tracing,
        )
        self._state_update_events[event_index] = event

    def declare_default_mode(self, mode: int):
        self._default_mode = mode

    def declare_zero_crossing(
        self,
        guard: Callable,
        reset_map: Callable = None,
        start_mode: int = None,
        end_mode: int = None,
        direction: str = "crosses_zero",
        terminal: bool = False,
        name: str = None,
        enable_tracing: bool = None,
    ):
        """Declare an event triggered by a zero-crossing of a guard function.

        Optionally, the system can also transition between discrete modes
        If `start_mode` and `end_mode` are specified, the system will transition
        from `start_mode` to `end_mode` when the event is triggered according to `guard`.
        This event will be active conditionally on `state.mode == start_mode` and when
        triggered will result in applying the reset map. In addition, the mode will be
        updated to `end_mode`.

        If `start_mode` and `end_mode` are not specified, the event will always be active
        and will not result in a mode transition.

        The guard function should have the signature:
            `guard(time, state, *inputs, **parameters) -> float`

        and the reset map should have the signature of an unrestricted update:
            `reset_map(time, state, *inputs, **parameters) -> state`

        Args:
            guard (Callable):
                The guard function which triggers updates on zero crossing.
            reset_map (Callable, optional):
                The reset map which is applied when the event is triggered. If None
                (default), no reset is applied.
            start_mode (int, optional):
                The mode or stage of the system in which the guard will be
                actively monitored. If None (default), the event will always be
                active.
            end_mode (int, optional):
                The mode or stage of the system to which the system will transition
                when the event is triggered. If start_mode is None, this is ignored.
                Otherwise it _must_ be specified, though it can be the same as
                start_mode.
            direction (str, optional):
                The direction of the zero crossing. Options are "crosses_zero"
                (default), "positive_then_non_positive", "negative_then_non_negative",
                and "edge_detection".  All except edge detection operate on continuous
                signals; edge detection operates on boolean signals and looks for a
                jump from False to True or vice versa.
            terminal (bool, optional):
                If True, the event will halt simulation if and when the zero-crossing
                occurs. If this event is triggered the reset map will still be applied
                as usual prior to termination. Defaults to False.
            name (str, optional):
                The name of the event. Defaults to None.
            enable_tracing (bool, optional):
                If True, enable tracing for this event. Defaults to None.

        Notes:
            By default the system state does not have a "mode" component, so in
            order to declare "state transitions" with non-null start and end modes,
            the user must first call `declare_default_mode` to set the default mode
            to be some integer (initial condition for the system).
        """

        logger.debug(
            f"Declaring transition for {self.name} with guard {guard} and reset map {reset_map}"
        )

        if enable_tracing is None:
            enable_tracing = True

        if start_mode is not None or end_mode is not None:
            assert (
                self._default_mode is not None
            ), "System has no mode: call `declare_default_mode` before transitions."
            assert isinstance(start_mode, int) and isinstance(end_mode, int)

        # Wrap the reset map with a mode update if necessary
        def _reset_and_update_mode(
            time: Scalar, state: LeafState, *inputs, **parameters
        ) -> LeafState:
            if reset_map is not None:
                state = reset_map(time, state, *inputs, **parameters)
            logger.debug(f"Updating mode from {state.mode} to {end_mode}")

            # If the start and end modes are declared, update the mode
            if start_mode is not None:
                logger.debug(f"Updating mode from {state.mode} to {end_mode}")
                state = state.with_mode(end_mode)

            return state

        _wrapped_guard = self.wrap_callback(guard)
        _wrapped_reset = _wrap_reset_map(
            self, _reset_and_update_mode, _wrapped_guard, terminal
        )

        event = ZeroCrossingEvent(
            system_id=self.system_id,
            guard=_wrapped_guard,
            reset_map=_wrapped_reset,
            passthrough=self._passthrough,
            direction=direction,
            is_terminal=terminal,
            name=name,
            event_data=ZeroCrossingEventData(active=True, triggered=False),
            enable_tracing=enable_tracing,
            active_mode=start_mode,
        )

        event_index = len(self._zero_crossing_events)
        self._zero_crossing_events.append(event)

        # Record the transition in the transition map (for debugging or analysis)
        if start_mode is not None:
            if start_mode not in self.transition_map:
                self.transition_map[start_mode] = []
            self.transition_map[start_mode].append((event_index, event))

    #
    # Initialization
    #
    @property
    def context_factory(self) -> LeafContextFactory:
        return LeafContextFactory(self)

    @property
    def dependency_graph_factory(self) -> LeafDependencyGraphFactory:
        return LeafDependencyGraphFactory(self)

    def create_state(self) -> LeafState:
        # Hook for context creation: get the default state for this system.
        # Users should not need to call this directly - the state will be created
        # as part of the context.  Generally, `system.create_context()` should
        # be all that's necessary for initialization.

        return LeafState(
            name=self.name,
            continuous_state=self._default_continuous_state,
            discrete_state=self._default_discrete_state,
            mode=self._default_mode,
            cache=tuple(self._default_cache),
        )

    def initialize_static_data(self, context: ContextBase):
        # Try to infer any missing default values for "sample-and-hold" output ports
        # and any other cached computations.
        cached_callbacks: list(SystemCallback) = [
            cb for cb in self.callbacks if cb.cache_index is not None
        ]

        for callback in cached_callbacks:
            i = callback.cache_index
            if self._default_cache[i] is None:
                try:
                    if isinstance(callback, OutputPort):
                        # Try to eval the callback for the _event_ (not the output
                        # port return function), which would return a value of the
                        # right data type for the output port, provided it is connected
                        _eval = callback.event.callback
                    else:
                        # If it's not an output port, the callback function evaluation
                        # should return the correct data type.
                        _eval = callback.eval

                    state: LeafState = _eval(context)
                    y = state.cache[i]
                    self._default_cache[i] = y
                    local_context = context[self.system_id].with_cached_value(i, y)
                    context = context.with_subcontext(self.system_id, local_context)
                except UpstreamEvalError:
                    logger.debug(
                        "%s.initialize_static_data: UpstreamEvalError. "
                        "Continuing without default value initialization.",
                        self.name,
                    )

        return context

    def _create_dependency_cache(self) -> dict[int, CallbackTracer]:
        cache = {}
        for source in self.callbacks:
            cache[source.callback_index] = CallbackTracer(ticket=source.ticket)
        return cache

    # Inherits docstring from SystemBase.get_feedthrough
    def get_feedthrough(self) -> List[Tuple[int, int]]:
        # NOTE: This implementation is basically a direct port of the Drake algorithm

        # If we already did this or it was set manually, return the stored value
        if self.feedthrough_pairs is not None:
            return self.feedthrough_pairs

        feedthrough = []  # Confirmed feedthrough pairs (input, output)

        # First collect all possible feedthrough pairs
        unknown: Set[Tuple[int, int]] = set()
        for iport in self.input_ports:
            for oport in self.output_ports:
                unknown.add((iport.index, oport.index))

        if len(unknown) == 0:
            return feedthrough

        # Create a local context and "cache".  The cache here just contains CallbackTracer
        # objects that can be used to trace dependencies through the system, but
        # otherwise don't store any actual values.  This is different from any "cached"
        # computations that might be stored in the state for reuse by multiple ports or
        # downstream calculations within the system.
        #
        # This cache will only contain local sources - this is fine since we're just
        # testing local input -> output paths for this system.
        cache = self._create_dependency_cache()

        original_unknown = unknown.copy()
        for pair in original_unknown:
            u, v = pair
            output_port = self.output_ports[v]
            input_port = self.input_ports[u]

            # If output prerequisites are unspecified, this tells us nothing
            if DependencyTicket.all_sources in output_port.prerequisites_of_calc:
                continue

            # Determine feedthrough dependency via cache invalidation
            cache = _mark_up_to_date(cache, output_port.callback_index)

            # Notify subscribers of a value change in the input, invalidating all
            # downstream cache values
            input_tracker = self.dependency_graph[input_port.ticket]
            cache = input_tracker.notify_subscribers(cache, self.dependency_graph)

            # If the output cache is now out of date, this is a feedthrough path
            if cache[output_port.callback_index].is_out_of_date:
                feedthrough.append(pair)

            # Regardless of the result of the caching, the pair is no longer unknown
            unknown.remove(pair)

            # Reset the output cache to out-of-date in case other inputs also
            # feed through to this output.
            cache = _mark_out_of_date(cache, output_port.callback_index)

        logger.debug(f"{self.name} feedthrough pairs: {feedthrough}")

        # Conservatively assume everything still unknown is feedthrough
        for pair in unknown:
            feedthrough.append(pair)

        self.feedthrough_pairs = feedthrough
        return self.feedthrough_pairs


#
# Zero-crossing event handling with custom adjoint definitions
#
def _wrap_reset_map(
    system: SystemBase, reset_map: Callable, guard: Callable, is_terminal: bool
) -> Callable:
    """Wrap the reset map with a custom adjoint definition.

    Ignoring "saltation" effects, we could just use the built-in autodiff.
    However, we need to correct for variations in the "time of impact", which
    has ramifications for the continuous state, time, and parameters in the
    _local_ context.  Only the local context is affected since we assume that the
    guard and reset maps can only see and act on the local state. We need to
    compute these corrections to the adjoints and then store them in an updated
    _root_ context (since this was the original input).

    The adjoint update to time is also overridden at the Simulator level
    because it is most naturally defined as t_adj = dot(xf_dot, vf) at the
    end of the `advance` call. However, we also need to update it here in case the
    adjoint time variable is used by other things farther in the graph.

    For a helpful tutorial related to this implementation, see:
    "Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems"
    https://arxiv.org/abs/2306.06862
    This is a bit more complicated because we allow for differentiation with respect to
    more than just the initial state, but it introduces the "saltation" idea.
    """

    system_id = system.system_id

    # Transform the signature from (time, state, *inputs, **params) -> state to
    # context -> state.
    reset_map = system.wrap_callback(reset_map)

    if IS_JAXLITE:
        return reset_map

    # Evaluate the ODE RHS function.  This is needed to get the time sensitivity
    # of the continuous state, which is used in the adjoint update to time.
    def _ode(x: LeafState, context: ContextBase) -> Array:
        local_context = context[system_id].with_state(x)
        context = context.with_subcontext(system_id, local_context)
        return system.eval_time_derivatives(context)

    def _reset_map_fwd(context: ContextBase) -> LeafState:
        """Compute the "forward pass" of the zero-crossing update function."""

        # This basically just wraps the event handler function, but it also computes various
        # "residual" information that will be necessary for the backwards pass. The way to
        # understand what residuals are needed is to start with the adjoint function and then
        # see what information used there can be more efficiently computed in the forward
        # pass.

        # Evaluate the _local_ system dynamics before the transition
        x_minus = context[system_id].state
        xdot_minus = _ode(x_minus, context)  # This is the _local_ xdot

        # Evaluate the _local_ gradients of the guard with x_minus
        root_context_adj: ContextBase = jax.grad(guard, allow_int=True)(context)
        context_adj = root_context_adj[system_id]
        dg_dx = context_adj.continuous_state
        dg_dp = context_adj.parameters
        dg_dt = root_context_adj.time  # time is only defined at the root

        # Reset map VJP, and compute the transition (primal values)
        # Returns the _local_ updated state x_plus, and the function to compute
        # the vjp of the reset map given the adjoint state. This will be used in the
        # backwards pass.
        x_plus, reset_vjp = jax.vjp(reset_map, context)

        # Recompute the _local_ ODE values after the transition.  This will be zero
        # if the event is terminal, since the state does not advance after the event.
        # Note that this can use standard Python control flow because it is known
        # at compile time whether an event is terminal or not.
        if not context.has_continuous_state:
            # Can't "zero out" the continuous state if it doesn't exist
            xdot_plus = None
        elif is_terminal:
            xdot_plus = 0 * xdot_minus
        else:
            xdot_plus = _ode(x_plus, context)

        # Combine all the "residuals" necessary for the backwards pass into a tuple.
        res = (dg_dx, dg_dt, dg_dp, xdot_minus, xdot_plus, reset_vjp)
        return x_plus, res

    def _reset_map_adj(res: tuple, state_adj: LeafState) -> ContextBase:
        """Compute the "backward pass" of the zero-crossing update function."""

        # Unpack the residuals from the forward pass
        dg_dx, dg_dt, dg_dp, xdot_minus, xdot_plus, reset_vjp = res

        # Compute vjp with reset Jacobian (return is a tuple of length 1). If we did not
        # account for saltation effects we could just return this directly.
        root_context_adj: ContextBase = reset_vjp(state_adj)[0]

        # Unpack the adjoint variables associated with the reset map
        context_adj = root_context_adj[system_id]
        vT_dR_dx = context_adj.continuous_state
        vT_dR_dp = context_adj.parameters
        vT_dR_dt = root_context_adj.time  # Time is only defined at the root

        context_adj = context_adj.with_state(state_adj)

        # Get the adjoint continuous state
        vc = state_adj.continuous_state

        # Compute vjp with rank-1 correction
        den = dg_dt + jnp.dot(dg_dx, xdot_minus)
        num = jnp.dot(xdot_plus, vc) - jnp.dot(vT_dR_dx, xdot_minus) - vT_dR_dt
        safe_den = jnp.where(den != 0, den, 1.0)
        gamma = jnp.where(den != 0, num / safe_den, 0.0 * num)
        vT_C = gamma * dg_dx  # Correction to vjp with the reset map

        # Full vjp with saltation matrix (corrected reset vjp)
        vT_Xi = vT_dR_dx + vT_C

        # Update context with the adjoint variables associated with the continuous state
        context_adj = context_adj.with_continuous_state(vT_Xi)

        # Adjoint update to the parameters: vT * (dR_dp + frac * dg_dp)
        vT_dp = jax.tree_util.tree_map(lambda x, y: x + gamma * y, vT_dR_dp, dg_dp)

        # Update the adjoint variables associated with the parameters
        context_adj = context_adj.with_parameters(vT_dp)

        # Update the adjoint variables associated with time
        t_adj = vT_dR_dt + gamma * dg_dt

        root_context_adj = root_context_adj.with_subcontext(system_id, context_adj)
        root_context_adj = root_context_adj.with_time(t_adj)

        return (root_context_adj,)

    _reset_map = jax.custom_vjp(reset_map)
    _reset_map.defvjp(_reset_map_fwd, _reset_map_adj)
    return _reset_map
