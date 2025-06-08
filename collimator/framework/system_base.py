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

"""Base class for System objects.

Systems are the basic building blocks of a model in the collimator framework.
In isolation they represent a hybrid dynamical system, but they can also be
combined into block diagrams to represent more complex models.  The interface
defined by SystemBase is intended for composability, so diagrams can themselves
be treated as "blocks" in larger diagrams.

Systems are intended to act as a collection of pure functions, with the
"dynamic" data (time, state, etc.) stored in a corresponding Context object.
For example, `system.eval_time_derivatives(context)` will return the time
derivatives of the continuous state for the system, given the root context of
the diagram.
"""

from __future__ import annotations

import abc
from functools import wraps
import inspect
import traceback
from typing import (
    List,
    Tuple,
    TYPE_CHECKING,
    Hashable,
    Callable,
    Union,
    ParamSpec,
    TypeVar,
)
import dataclasses

import numpy as np

from collimator.framework.error import BlockParameterError
from ..backend import utils
from ..logging import logger
from .cache import SystemCallback
from .event import FlatEventCollection
from .parameter import Parameter
from .port import InputPort, OutputPort
from .pprint import pprint_fancy
from .dependency_graph import DependencyTicket, sort_trackers
from .error import StaticError

__all__ = ["SystemBase", "parameters"]

if TYPE_CHECKING:
    # Array-like object, e.g. a JAX PyTree or a numpy ndarray.
    from ..backend.typing import (
        Array,
        ShapeLike,
        DTypeLike,
    )

    from .context import ContextBase
    from .context_factory import ContextFactory
    from .diagram import Diagram
    from .error import ErrorCollector
    from .event import DiscreteUpdateEvent, EventCollection
    from .port import DirectedPortLocator, PortBase
    from .dependency_graph import DependencyGraph, DependencyGraphFactory

    # See note on "state" type hints in the docstring of SystemBase.
    from .state import State, StateComponent


def next_system_id() -> int:
    """Get the next unique system ID."""
    return IDGenerator.get_next_id()


def reset_system_id():
    """Reset the system ID counter."""
    IDGenerator.reset()


def _get_arg_names(func):
    sig = inspect.signature(func)
    positional_arg_names = []
    all_args = []
    for name, param in sig.parameters.items():
        is_positional_or_keyword = param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )
        if is_positional_or_keyword and param.default == inspect.Parameter.empty:
            positional_arg_names.append(name)
        all_args.append(name)

    return all_args, positional_arg_names


def _get_default_value(init_func, param_name):
    sig = inspect.signature(init_func)
    param = sig.parameters[param_name]
    if param.default == inspect.Parameter.empty:
        return None
    return param.default


def _get_params(param_names, init_func, args, kwargs, use_defaults=True):
    all_args, args_names = _get_arg_names(init_func)
    all_args = all_args[1:]  # remove self
    args_names = args_names[1:]  # remove self

    params = {}
    for k in param_names:
        if k in kwargs:
            params[k] = kwargs[k]
        elif k in args_names:
            idx = args_names.index(k)
            if idx >= len(args):
                raise ValueError(f"Missing required argument {k}")
            params[k] = args[idx]
        elif k in all_args and all_args.index(k) < len(args):
            params[k] = args[all_args.index(k)]
        elif use_defaults:
            # kwarg's default value
            params[k] = _get_default_value(init_func, k)
    return params


P = ParamSpec("P")
T = TypeVar("T")


def parameters(static: list[str] = None, dynamic: list[str] = None):
    """Decorator to apply to the __init__ function of a system to declare
    static or dynamic parameters."""

    if static is None:
        static = []

    if dynamic is None:
        dynamic = []

    static_param_names = set(static)
    dynamic_param_names = set(dynamic)

    def decorator(init_func: Callable[P, T]) -> Callable[P, T]:
        @wraps(init_func)
        def wrapped_init(self, *args, **kwargs):
            resolved_args = [
                arg.get() if isinstance(arg, Parameter) else arg for arg in args
            ]
            resolved_kwargs = {
                k: kwarg.get() if isinstance(kwarg, Parameter) else kwarg
                for k, kwarg in kwargs.items()
            }
            init_func(self, *resolved_args, **resolved_kwargs)

            static_params = _get_params(static_param_names, init_func, args, kwargs)
            for param_name, value in static_params.items():
                self.declare_static_parameter(param_name, value)

            dyn_params = _get_params(dynamic_param_names, init_func, args, kwargs)
            for param_name, value in dyn_params.items():
                if value is not None:
                    self.declare_dynamic_parameter(param_name, value)

        return wrapped_init

    return decorator


class IDGenerator:
    """Singleton class for automatically managing unique system IDs.

    This can be ignored altogether if using manually determined system IDs,
    for example as determined by "block path" from the UI. In either case, it
    typically does not need to be used directly by either users or developers.
    """

    _id = 0

    @classmethod
    def get_next_id(cls):
        cls._id += 1
        return cls._id

    @classmethod
    def reset(cls):
        cls._id = 0


class UpstreamEvalError(StaticError):
    def __init__(self, port_locator: DirectedPortLocator):
        system, direction, port_index = port_locator
        msg = (
            f"Evaluation of {direction}put port {system.name_path_str}[{port_index}] failed, "
            f"it may be disconnected or part of a subsystem"
        )
        super().__init__(
            msg, system=system, port_index=port_index, port_direction=direction
        )


@dataclasses.dataclass
class SystemBase:
    """Basic building block for simulation in collimator.

    NOTE: Type hints in SystemBase indicate the union between what would be returned
    by a LeafSystem and a Diagram. See type hints of the subclasses for the specific
    argument and return types.
    """

    # Generated unique ID for this system
    system_id: Hashable = dataclasses.field(default_factory=next_system_id, init=False)
    name: str = None  # Human-readable name for this system (optional but never None)
    ui_id: str = None  # UUID of the block when loaded from JSON (optional, may be None)

    # Immediate parent of the current system (can only be a Diagram).
    # If None, _this_ is the root system.
    parent: Diagram = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"{type(self).__name__}_{self.system_id}_"

        # All "cache sources" for this system. Typically these will correspond to
        # input ports, output ports, time derivative calculations, and any custom
        # "cached data" declared by the user (e.g. see ModelicaFMU block).
        self.callbacks: List[SystemCallback] = []

        # Index into SystemCallbacks for each port. For instance, input port `i` can
        # be retrieved by `self.callbacks[self.input_port_indices[i]]`. The
        # `input_ports` and `output_ports` properties give more convenient access.
        self.input_port_indices: List[int] = []
        self.output_port_indices: List[int] = []

        # Override this or set manually to provide a custom characteristic time scale
        # for the system. At the moment this is only used for zero-crossing isolation
        # in the simulator.
        self.characteristic_time = 1.0

        # A dependency graph for the system, mapping prerequisites of each calculation.
        # `None` indicates that the dependency graph has not been constructed yet.  If
        # accessed via the `dependency_graph` property, it will be constructed
        # automatically as necessary.
        self._dependency_graph: DependencyGraph = None

        # Static parameters are not jax-traceable
        self._static_parameters: dict[str, Parameter] = {}

        # If not empty, this defines the shape and data type of the numeric parameters
        # in the LeafSystem. This value will be used to initialize the context, so it
        # will also serve as the initial value unless explicitly overridden. In the
        # simplest cases, parameters could be stored as attributes of the LeafSystem,
        # but declaring them has the advantage of moving the values to the context,
        # allowing them to be traced by JAX rather than stored as static data. This
        # means they can be differentiated, vmapped, or otherwise modified without
        # re-compiling the simulation.
        self._dynamic_parameters: dict[str, Array] = {}

        # Map from (input_port, output_port) if that pair is feedthrough
        # `None` indicates that the feedthrough is unknown for this system.
        # This will be computed automatically using the dependency graph
        # during algebraic loop detection unless it is set manually.
        # To manually set feedthrough, either declare this explicitly or
        # override `get_feedthrough`.
        self.feedthrough_pairs: List[Tuple[int, int]] = None

        # Pre-sorted list of all output update events for this system.  This will
        # be created when the associated property is first accessed.  This should
        # only need to be done for the root system.
        self._cache_update_events: EventCollection = None

    def __hash__(self) -> Hashable:
        return hash(self.system_id)

    def pprint(self, output=print, fancy=True) -> str:
        """Pretty-print the system and its hierarchy."""
        output(self._pprint_helper(fancy=fancy).strip())

    def _pprint_helper(self, prefix="", fancy=True) -> str:
        if fancy:
            return pprint_fancy(prefix, self)
        return f"{prefix}|-- {self.name}(id={self.system_id})\n"

    def post_simulation_finalize(self) -> None:
        """Finalize the system after simulation has completed.

        This is only intended for special blocks that need to clean up
        resources and close files."""

    @property
    def static_parameters(self) -> dict[str, Parameter]:
        return self._static_parameters

    @static_parameters.setter
    def static_parameters(self, value):
        self._static_parameters = value

    @property
    def dynamic_parameters(self) -> dict[str, Parameter]:
        return self._dynamic_parameters

    @property
    def parameters(self) -> dict[str, Parameter]:
        return {**self.static_parameters, **self.dynamic_parameters}

    #
    # Simulation interface
    #
    @abc.abstractproperty
    def has_feedthrough_side_effects(self) -> bool:
        """Check if the system includes any feedthrough calls to `io_callback`."""
        # This is a tricky one to explain and is almost always False except for a
        # PythonScript block that is not JAX traced.  Basically, if the output of
        # the system is used as an ODE right-hand-side, will it fail in the case where
        # the ODE solver defines a custom VJP?  This happens in diffrax, so for example
        # if a PythonScript block is used to compute the ODE right-hand-side, it will
        # fail with "Effects not supported in `custom_vjp`"
        pass

    @abc.abstractproperty
    def has_ode_side_effects(self) -> bool:
        """Check if the ODE RHS for the system includes any calls to `io_callback`."""
        # This flag indicates that the system `has_feedthrough_side_effects` AND that
        # signal is used as an ODE right-hand-side.  This is used to determine whether
        # a JAX ODE solver can be used to integrate the system.
        pass

    @abc.abstractproperty
    def has_continuous_state(self) -> bool:
        pass

    @abc.abstractproperty
    def has_discrete_state(self) -> bool:
        pass

    @abc.abstractproperty
    def has_zero_crossing_events(self) -> bool:
        pass

    def eval_time_derivatives(self, context: ContextBase) -> StateComponent:
        """Evaluate the continuous time derivatives for this system.

        Given the _root_ context, evaluate the continuous time derivatives,
        which must have the same PyTree structure as the continuous state.

        In principle, this can be overridden by custom implementations, but
        in general it is preferable to declare continuous states for LeafSystems
        using `declare_continuous_state`, which accepts a callback function
        that will be used to compute the derivatives. For Diagrams, the time
        derivatives are computed automatically using the callback functions for
        all child systems with continuous state.

        Args:
            context (ContextBase): root context of this system

        Returns:
            StateComponent:
                Continuous time derivatives for this system, or None if the system
                has no continuous state.
        """
        return None

    @property
    @abc.abstractmethod
    def mass_matrix(self) -> StateComponent:
        """Mass matrix for this system.

        Returns PyTree-structured data where each leaf is an (n, n) array.
        This is used for implicit integration methods (currently only BDF).
        """
        pass

    @property
    @abc.abstractmethod
    def has_mass_matrix(self) -> bool:
        """Returns True if any component of the system has a nontrivial mass matrix."""
        pass

    @abc.abstractmethod
    def eval_zero_crossing_updates(
        self,
        context: ContextBase,
        events: EventCollection,
    ) -> State:
        """Evaluate reset maps associated with zero-crossing events.

        Args:
            context (ContextBase):
                The context for the system, containing the current state and parameters.
            events (EventCollection):
                The collection of events to be evaluated (for example zero-crossing or
                periodic events for this system).

        Returns:
            State: The complete state with all updates applied.

        Notes:
            (1) Following the Drake definition, "unrestricted" updates are allowed to
            modify any component of the state: continuous, discrete, or mode.  These
            updates are evaluated in the order in which they were declared, so it is
            _possible_ (but should be strictly avoided) for multiple events to modify the
            same state component at the same time.

            Each update computes its results given the _current_ state of the system
            (the "minus" values) and returns the _updated_ state (the "plus" values).
            The update functions cannot access any information about the "plus" values of
            its own state or the state of any other block.  This could change in the future
            but for now it ensures consistency with Drake's discrete semantices:

            More specifically, since all unrestricted updates can modify the entire state,
            any time there are multiple unrestricted updates, the resulting states are
            ALWAYS in conflict.  For example, suppose a system has two unrestricted
            updates, `event1` and `event2`.  At time t_n, `event1` is active and `event2`
            is inactive.  First, `event1` is evaluated, and the state is updated.  Then
            `event2` is evaluated, but the state is not updated.  Which one is valid?
            Obviously, the `event1` return is valid, but how do we communicate this to JAX?
            The situation is more complicated if both `event1` and `event2` happen to be
            active.  In this case the states have to be "merged" somehow.  In the worst
            case, these two will modify the same components of the state in different ways.

            The implementation updates the state in a local copy of the context (since both
            are immutable).  This allows multiple unrestricted updates, but leaves open the
            possibility of multiple active updates modifying the state in conflicting ways.
            This should strictly be avoided by the implementer of the LeafSystem.  If it is
            at all unclear how to do this, it may be better to split the system into
            multiple blocks to be safe.

            (2) The events are evaluated conditionally on being marked "active"
            (indicating that their guard function triggered), so the entire event
            collection can be passed without filtering to active events. This is necessary
            to make the function calls work with JAX tracing, which do not allow for
            variable-sized arguments or returns.
        """
        pass

    def handle_discrete_update(
        self, events: EventCollection, context: ContextBase
    ) -> ContextBase:
        """Compute and apply active discrete updates.

        Given the _root_ context, evaluate the discrete updates, which must have the
        same PyTree structure as the discrete states of this system. This should be
        a pure function, so that it does not modify any aspect of the context in-place
        (even though it is difficult to strictly prevent this in Python).

        This will evaluate the set of events that result from declaring state or output
        update events on systems using `LeafSystem.declare_periodic_update` and
        `LeafSystem.declare_output_port` with an associated periodic update rate.

        This is intended for internal use by the simulator and should not normally need
        to be invoked directly by users. Events are evaluated conditionally on being
        marked "active", so the entire event collection can be passed without filtering
        to active events. This is necessary to make the function calls work with JAX
        tracing, which do not allow for variable-sized arguments or returns.

        For a discrete system updating at a particular rate, the update rule for a
        particular block is:

        ```
        x[n+1] = f(t[n], x[n], u[n])
        y[n]   = g(t[n], x[n], u[n])
        ```

        Additionally, the value y[n] is held constant until the next update from the
        point of view of other continuous-time or asynchronous discrete-time blocks.

        Because each output `y[n]` may in general depend on the input `u[n]` evaluated
        _at the same time_, the composite discrete update function represents a
        system of equations.  However, since algebraic loops are prohibited, the events
        can be ordered and executed sequentially to ensure that the updates are applied
        in the correct order.  This is implemented in
        `SystemBase.sorted_event_callbacks`.

        Multirate systems work in the same way, except that the events are evaluated
        conditionally on whether the current time corresponds to an update time for each
        event.

        Args:
            events (EventCollection): collection of discrete update events
            context (ContextBase): root context for this system

        Returns:
            ContextBase:
                updated context with all active updates applied to the discrete state
        """
        logger.debug(
            f"Handling {events.num_events} discrete update events at t={context.time}"
        )
        if events.has_events:
            # Sequentially apply all updates
            for event in events:
                system_id = event.system_id
                state = event.handle(context)
                local_context = context[system_id].with_state(state)
                context = context.with_subcontext(system_id, local_context)

        return context

    def handle_zero_crossings(
        self, events: EventCollection, context: ContextBase
    ) -> ContextBase:
        """Compute and apply active zero-crossing events.

        This is intended for internal use by the simulator and should not normally need
        to be invoked directly by users. Events are evaluated conditionally on being
        marked "active", so the entire event collection can be passed without filtering
        to active events. This is necessary to make the function calls work with JAX
        tracing, which do not allow for variable-sized arguments or returns.

        Args:
            events (EventCollection): collection of zero-crossing events
            context (ContextBase): root context for this system

        Returns:
            ContextBase: updated context with all active zero-crossing events applied
        """
        logger.debug(
            "Handling %d state transition events at t=%s",
            events.num_events,
            context.time,
        )
        if events.num_events > 0:
            # Compute the updated state for all active events
            state = self.eval_zero_crossing_updates(context, events)

            # Apply the updates
            context = context.with_state(state)
        return context

    # Events that happen at some regular interval
    @property
    def periodic_events(self) -> FlatEventCollection:
        return self.cache_update_events + self.state_update_events

    # Events that update the discrete state of the system
    @abc.abstractproperty
    def state_update_events(self) -> FlatEventCollection:
        pass

    # Events that refresh sample-and-hold outputs of the system
    @property
    def cache_update_events(self) -> FlatEventCollection:
        if self._cache_update_events is None:
            # Sort the output update events and store in the private attribute.
            self._cache_update_events = [
                cb.event for cb in self.sorted_event_callbacks if cb.event is not None
            ]

        return FlatEventCollection(tuple(self._cache_update_events))

    @abc.abstractproperty
    def _flat_callbacks(self) -> List[SystemCallback]:
        """Get a flat list of callbacks for this and all child systems."""
        pass

    @property
    def sorted_event_callbacks(self) -> List[SystemCallback]:
        """Sort and return the event-related callbacks for this system."""
        # Collect all of the callbacks associated with cache update events.  These are
        # SystemCallback objects, so they are all associated with trackers in the
        # dependency graph.  We can use these to sort the events in execution order.
        trackers = sort_trackers([cb.tracker for cb in self._flat_callbacks])

        # Retrieve the callback associated with each tracker.
        return [tracker.cache_source for tracker in trackers]

    # Events that are triggered by a "guard" function and may induce a "reset" map
    @abc.abstractproperty
    def zero_crossing_events(self) -> EventCollection:
        pass

    @abc.abstractmethod
    def determine_active_guards(self, context: ContextBase) -> EventCollection:
        """Determine active guards for zero-crossing events.

        This method is responsible for evaluating and determining which
        zero-crossing events are active based on the current system mode
        and other conditions.  This can be overridden to flag active/inactive
        guards on a block-specific basis, for instance in a StateMachine-type
        block. By default all guards are marked active at this point unless
        the zero-crossing event was declared with a non-default `start_mode`, in
        which case the guard is activated conditionally on the current mode.

        For example, in a system with finite state transitions, where a transition from
        mode A to mode B is triggered by a guard function g_AB and the inverse
        transition is triggered by a guard function g_BA, this function would activate
        g_AB if the system is in mode A and g_BA if the system is in mode B. The other
        guard function would be inactive.  If the zero-crossing event is not associated
        with a starting mode, it is considered to be always active.

        Args:
            context (ContextBase):
                The root context containing the overall state and parameters.

        Returns:
            EventCollection:
                A collection of zero-crossing events with active/inactive status
                updated based on the current system mode and other conditions.
        """
        pass

    #
    # I/O ports
    #
    @property
    def input_ports(self) -> List[InputPort]:
        return [self.callbacks[i] for i in self.input_port_indices]

    def get_input_port(self, name: str) -> tuple[InputPort, int]:
        """Retrieve a specific input port by name."""
        for i, port in enumerate(self.input_ports):
            if port.name == name:
                return port, i
        raise ValueError(
            f"System {self.name} has no input port named {name}. "
            f"Available ports: {list(map(lambda x: x.name,self.input_ports))}"
        )

    @property
    def num_input_ports(self) -> int:
        return len(self.input_port_indices)

    @property
    def output_ports(self) -> List[OutputPort]:
        return [self.callbacks[i] for i in self.output_port_indices]

    def get_output_port(self, name: str) -> OutputPort:
        """Retrieve a specific output port by name."""
        for port in self.output_ports:
            if port.name == name:
                return port
        raise ValueError(f"System {self.name} has no output port named {name}")

    @property
    def num_output_ports(self) -> int:
        return len(self.output_port_indices)

    def eval_input(self, context: ContextBase, port_index: int = 0) -> Array:
        """Get the input for a given port.

        This works by evaluating the callback function associated with the port, which
        will "pull" the upstream output port values.

        Args:
            context (ContextBase): root context for this system
            port_index (int, optional): index into `self.input_ports`, for example
                the value returned by `declare_input_port`. Defaults to 0.

        Returns:
            Array: current input values
        """
        return self.input_ports[port_index].eval(context)

    def collect_inputs(self, context: ContextBase) -> List[Array]:
        """Collect all current inputs for this system.

        Args:
            context (ContextBase): root context for this system

        Returns:
            List[Array]: list of all current input values
        """
        return [self.eval_input(context, i) for i in range(self.num_input_ports)]

    def _eval_input_port(self, context: ContextBase, port_index: int) -> Array:
        """Evaluate an upstream input port given the _root_ context.

        Intended for internal use as a callback function. Users and developers
        should typically call `eval_input` in order to get this information. That
        method will call the callback function associated with the input port,
        which will have a reference to this method.

        Args:
            context (ContextBase): root context for this system
            port_index (int): index of the input port to evaluate on the target system

        Returns:
            Array: current input values
        """
        # A helper function to evaluate an upstream input port given the _root_ context.

        port_locator = self.input_ports[port_index].locator

        if self.parent is None:
            # This is currently the root system.  Typically root input ports should not be evaluated,
            #  but we can get here during subsystem construction (e.g. type inference).  In that case,
            #  we should just defer evaluation and rely on the graph analysis to determine that
            #  everything is connected correctly.
            # See https://collimator.atlassian.net/browse/WC-51.
            # This should not happen during simulation or root context construction.
            logger.debug(
                "    ---> %s is the root system, deferring evaluation of %s[%s]",
                self.name,
                port_locator[0].name,
                port_locator[1],
            )
            raise UpstreamEvalError(port_locator=(self, "in", port_index))

        # The `eval_subsystem_input_port` method is only defined for Diagrams, but
        # the parent system is guaranteed to be a Diagram if this is not the root.
        # If it is the root, it should not have any (un-fixed) input ports.
        return self.parent.eval_subsystem_input_port(context, port_locator)

    #
    # Declaration utilities
    #
    def _next_input_port_name(self, name: str | None = None) -> str:
        """Automatically generate a unique name for the next input port."""
        if name is not None:
            assert name != ""
            return name
        return f"in_{self.num_input_ports}"

    def _next_output_port_name(self, name: str | None = None) -> str:
        """Automatically generate a unique name for the next output port."""
        if name is not None:
            assert name != ""
            return name
        return f"out_{self.num_output_ports}"

    def declare_input_port(
        self,
        name: str = None,
        prerequisites_of_calc: List[DependencyTicket] = None,
    ) -> int:
        """Add an input port to the system.

        Returns the corresponding index into the system input_port_indices list
        Note that this is different from the callbacks index - typically it
        will make more sense to retrieve via system.input_ports[port_index], but

        Args:
            name (str, optional): name of the new port. Defaults to None, which will
                use the default naming scheme for the system (e.g. "u_0")
            prerequisites_of_calc (List[DependencyTicket], optional): list of
                dependencies for the callback function. Defaults to None.

        Returns:
            int: port index of the newly created port in `input_ports`
        """
        port_index = self.num_input_ports
        port_name = self._next_input_port_name(name)

        for port in self.input_ports:
            assert (
                port.name != port_name
            ), f"System {self.name} already has an input port named {port.name}"

        def _callback(context: ContextBase) -> Array:
            # Given the root context, evaluate the input port using the helper function
            return self._eval_input_port(context, port_index)

        callback_index = len(self.callbacks)
        port = InputPort(
            _callback,
            system=self,
            callback_index=callback_index,
            name=port_name,
            index=port_index,
            prerequisites_of_calc=prerequisites_of_calc,
        )

        assert isinstance(port, InputPort)
        assert port.system is self
        assert port.name != ""

        # Check that name is unique
        for p in self.input_ports:
            assert (
                p.name != port.name
            ), f"System {self.name} already has an input port named {port.name}"

        self.input_port_indices.append(callback_index)
        self.callbacks.append(port)

        return port_index

    def declare_output_port(
        self,
        callback: Callable,
        name: str = None,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array = None,
        event: DiscreteUpdateEvent = None,
        cache_index: int = None,
    ) -> int:
        """Add an output port to the system.

        This output port could represent any function of the context available to
        the system, so a callback function is required.  This function should have
        the form
            `callback(context: ContextBase) -> Array`
        SystemBase implementations have some specific convenience wrappers, e.g.:
            `LeafSystem.declare_continuous_state_output`
            `Diagram.export_output`

        Common cases are:
        - Feedthrough blocks: gather inputs and return some function of the
            inputs (e.g. a gain)
        - Stateful blocks: use LeafSystem.declare_(...)_state_output_port to
            return the value of a particular state
        - Diagrams: create and export a diagram-level port to the parent system using
            the callback function associated with the system-level port

        Returns the corresponding index into the system output_port_indices list
        Note that this is different from the callbacks index - typically it
        will make more sense to retrieve via system.output_ports[port_index].

        Args:
            callback (Callable, optional): computes the value of the output port given
                the root context.
            name (str, optional): name of the new port. Defaults to None, which will
                use the default naming scheme for the system (e.g. "y_0")
            prerequisites_of_calc (List[DependencyTicket], optional): list of
                dependencies for the callback function. Defaults to None, which will
                use the default dependencies for the system (all sources).  This may
                conservatively flag the system as having algebraic loops, so it is
                better to be specific here when possible.  This is done automatically
                in the wrapper functions like `LeafSystem.declare_(...)_output_port`
            default_value (Array, optional): A default array-like value used to seed
                the context and perform type inference, when this is known up front.
                Defaults to None, which will use information propagation through the
                graph along with type promotion to determine an appropriate value.
            event (DiscreteUpdateEvent, optional): A discrete update event associated
                with this output port that will periodically refresh the value that
                will be returned by the callback function. This makes the port act as
                a sample-and-hold rather than a direct function evaluation.
            cache_index (int, optional): Index into the cache state component
                corresponding to the output port result, if the output port is of
                periodically-updated sample-and-hold type.

        Returns:
            int: port index of the newly created port
        """
        port_index = self.num_output_ports
        port_name = self._next_output_port_name(name)

        for port in self.output_ports:
            assert (
                port.name != port_name
            ), f"System {self.name} already has an output port named {port.name}"

        if prerequisites_of_calc is None:
            prerequisites_of_calc = [DependencyTicket.all_sources]

        callback_index = len(self.callbacks)
        port = OutputPort(
            callback,
            system=self,
            callback_index=callback_index,
            name=port_name,
            index=port_index,
            prerequisites_of_calc=prerequisites_of_calc,
            default_value=default_value,
            event=event,
            cache_index=cache_index,
        )

        assert isinstance(port, OutputPort)
        assert port.system is self
        assert port.name != ""

        # Check that name is unique
        for p in self.output_ports:
            assert (
                p.name != port.name
            ), f"System {self.name} already has an output port named {port.name}"

        logger.debug("Adding output port %s to %s", port, self.name)
        self.output_port_indices.append(callback_index)
        self.callbacks.append(port)
        logger.debug(
            "    ---> %s now has %s output ports: %s",
            self.name,
            len(self.output_ports),
            self.output_ports,
        )
        logger.debug(
            "    ---> %s now has %s cache sources: %s",
            self.name,
            len(self.callbacks),
            self.callbacks,
        )

        return port_index

    def configure_output_port(
        self,
        port_index: int,
        callback: Callable,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array = None,
        event: DiscreteUpdateEvent = None,
        cache_index: int = None,
    ):
        """Configure an output port of the system.

        See `declare_output_port` for a description of the arguments.

        Args:
            port_index (int): index of the output port to configure

        Returns:
            None
        """
        old_port = self.output_ports[port_index]

        if prerequisites_of_calc is None:
            prerequisites_of_calc = [DependencyTicket.all_sources]

        port = OutputPort(
            callback,
            system=self,
            callback_index=old_port.callback_index,
            name=old_port.name,
            index=port_index,
            prerequisites_of_calc=prerequisites_of_calc,
            default_value=default_value,
            event=event,
            cache_index=cache_index,
            ticket=old_port.ticket,
        )

        self.callbacks[old_port.callback_index] = port
        logger.debug(
            "    ---> %s now has %s output ports: %s",
            self.name,
            len(self.output_ports),
            self.output_ports,
        )
        logger.debug(
            "    ---> %s now has %s cache sources: %s",
            self.name,
            len(self.callbacks),
            self.callbacks,
        )

    @abc.abstractmethod
    def get_feedthrough(self) -> List[Tuple[int, int]]:
        """Determine pairs of direct feedthrough ports for this system.

        By default, the algorithm relies on the dependency tracking system to determine
        feedthrough, but this can be overridden by implementing this method directly
        in a subclass, for instance if the automatic dependency tracking is too
        conservative in determining feedthrough.

        Returns:
            List[Tuple[int, int]]:
                A list of tuples (u, v) indicating that output port v has a direct
                dependency on input port u, resulting in a feedthrough path in the system.
                The indices u and v correspond to the indices of the input and output
                ports in the system's input and output port lists.
        """
        pass

    #
    # Initialization
    #
    def create_context(self, **kwargs) -> ContextBase:
        """Create a new context for this system.

        The context will contain all variable information used in
        simulation/analysis/optimization, such as state and parameters.

        Returns:
            ContextBase: new context for this system
        """
        return self.context_factory(**kwargs)

    def check_types(self, context: ContextBase, error_collector: ErrorCollector = None):
        """Perform any system-specific static analysis."""
        pass

    @abc.abstractproperty
    def context_factory(self) -> ContextFactory:
        """Factory object for creating contexts for this system.

        Should not be called directly - use `system.create_context` instead.
        """
        pass

    @property
    def dependency_graph(self) -> DependencyGraph:
        """Retrieve (or create if necessary) the dependency graph for this system."""
        if self._dependency_graph is None:
            self.create_dependency_graph()
        return self._dependency_graph

    @abc.abstractproperty
    def dependency_graph_factory(self) -> DependencyGraphFactory:
        """Factory object for creating dependency graphs for this system.

        Should not be called directly - use `system.create_dependency_graph` instead.
        """
        pass

    def create_dependency_graph(self):
        """Create a dependency graph for this system."""
        if self._dependency_graph is None:
            self._dependency_graph = self.dependency_graph_factory()

    def update_dependency_graph(self):
        """Update the dependency graph for this system, if already created."""
        if self._dependency_graph is not None:
            self._dependency_graph = self.dependency_graph_factory()

    def initialize_static_data(self, context: ContextBase) -> ContextBase:
        """Initialize any context data that has to be done after context creation.

        Use this to define custom auxiliary data or type inference that doesn't
        get traced by JAX. See the `ZeroOrderHold` implementation for an example.
        Since this is only applied during context initialization, it is allowed to
        modify the context directly (or the system itself).

        Typically this should not be called outside of the ContextFactory.

        Args:
            context (ContextBase): partially initialized context for this system.
        """
        return context

    @property
    def ports(self) -> dict[str, PortBase]:
        """Dictionary of all ports in this system, indexed by name"""
        return {port.name: port for port in self.input_ports + self.output_ports}

    # Convenience functions for errors and UI logs

    @property
    def name_path(self) -> list[str]:
        """Get the human-readable path to this system. None if some names are not set."""
        if self.parent is None:
            return [self.name]  # Likely to be 'root'
        if self.parent.parent is None:
            return [self.name]  # top-level block
        return self.parent.name_path + [self.name]

    @property
    def name_path_str(self) -> str:
        """Get the human-readable path to this system as a string."""
        return ".".join(self.name_path)

    @property
    def ui_id_path(self) -> Union[list[str], None]:
        """Get the uuid node path to this system. None if some IDs are not set."""
        if self.ui_id is None:
            return None
        if self.parent is None:
            return [self.ui_id]
        if self.parent.parent is None:
            return [self.ui_id]  # top-level block
        parent_path = self.parent.ui_id_path
        if parent_path is None:
            return None
        return parent_path + [self.ui_id]

    def declare_static_parameters(self, **params):
        """Declare a set of static parameters for the system.

        These parameters are not JAX-traceable and therefore can't be optimized.

        Examples of static parameters include booleans, strings, parameters
        used in shapes, etc.

        The args should be a dict of name-value pairs, where the values are either
        strings, bool, arrays, or Parameters.

        Typical usage:

        ```python
        class MyBlock(LeafSystem):
            def __init__(self, param1=True, param2=1.0):
                super().__init__()
                self.declare_static_parameters(param1=param1, param2=param2)
        ```
        """
        for name, value in params.items():
            if name in self.dynamic_parameters:
                raise BlockParameterError(
                    "Parameter already declared as dynamic parameter",
                    system=self,
                    parameter_name=name,
                )
            if isinstance(value, list):
                self._static_parameters[name] = Parameter(
                    value=np.array(value), system=self, is_static=True
                )
            else:
                self._static_parameters[name] = Parameter(
                    value=value, system=self, is_static=True
                )

    def declare_static_parameter(self, name, value):
        """Declare a single static parameter for the system.

        This is a convenience function for declaring a single static parameter.

        Args:
            name (str): name of the parameter
            value (Union[Array, Parameter]): value of the parameter
        """
        self.declare_static_parameters(**{name: value})

    def declare_dynamic_parameter(
        self,
        name: str,
        default_value: Array | Parameter = None,
        shape: ShapeLike = None,
        dtype: DTypeLike = None,
        as_array: bool = True,
    ):
        """Declare a numeric parameter for the system.

        Parameters are declared in the system and accessed through the context to
        maintain separation of data ownership. This method creates an entry in the
        system's dynamic_parameters, recording the name, default value, and dependency
        ticket for later reference.

        The default value will be used to initialize the context, so it
        will also serve as the initial value unless explicitly overridden. In the
        simplest cases, parameters could be stored as attributes of the LeafSystem,
        but declaring them has the advantage of moving the values to the context,
        allowing them to be traced by JAX rather than stored as static data. This
        means they can be differentiated, vmapped, or otherwise modified without
        re-compiling the simulation.

        Args:
            name (str):
                The name of the parameter.
            default_value (Union[Array, Parameter], optional):
                The default value of the parameter. Parameters are used
                primarily internally for serialization and should not normally need
                to be used directly when implementing LeafSystems. Defaults to None.
            shape (ShapeLike, optional):
                The shape of the parameter. Defaults to None.
            dtype (DTypeLike, optional):
                The data type of the parameter. Defaults to None.
            as_array (bool, optional):
                If True, treat the default_value as an array-like (cast if necessary).
                Otherwise, it will be stored as the default state without modification.

        Raises:
            AssertionError:
                If the parameter with the given name is already declared.

        Notes:
            (1) Only one of `shape` and `default_value` should be provided. If
            `default_value` is provided, it will be used as the initial value of the
            continuous state. If `shape` is provided, the initial value will be a
            zero vector of the given shape and specified dtype.
        """
        # assert (
        #     name not in self._dynamic_parameters
        # ), f"Parameter {name} already declared"

        if name in self.static_parameters:
            raise BlockParameterError(
                "Parameter already declared as static parameter",
                system=self,
                parameter_name=name,
            )

        try:
            if isinstance(default_value, Parameter):
                # FIXME: we should create a new Parameter object here for consistency
                # but this messes up the serialization here.
                self._dynamic_parameters[name] = default_value
                self._dynamic_parameters[name].as_array = as_array
            else:
                if as_array:
                    default_value = utils.make_array(
                        default_value, dtype=dtype, shape=shape
                    )
                self._dynamic_parameters[name] = Parameter(
                    value=default_value,
                    dtype=dtype,
                    shape=shape,
                    system=self,
                )

            logger.debug(
                "Adding parameter %s to %s with default: %s",
                name,
                self.name,
                default_value,
            )

        except Exception as e:
            traceback.print_exc()
            raise BlockParameterError(
                "Error declaring parameter",
                system=self,
                parameter_name=name,
            ) from e
