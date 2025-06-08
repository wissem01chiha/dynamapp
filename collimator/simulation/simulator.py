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

"""Functionality for simulating hybrid dynamical systems.

This module provides the `simulate` function, which is the primary entry point
for running simulations.  It also defines the `Simulator` class used by `simulate`,
which provides more fine-grained control over the simulation process.
"""

from __future__ import annotations
from functools import partial
import dataclasses
from typing import TYPE_CHECKING, Callable, Any

import numpy as np
import jax
import jax.numpy as jnp
from ..logging import logger
from ..profiling import Profiler
from ..lazy_loader import LazyLoader

from .types import (
    StepEndReason,
    GuardIsolationData,
    ContinuousIntervalData,
    SimulatorOptions,
    SimulatorState,
    SimulationResults,
    ResultsOptions,
    ResultsMode,
)

from ..backend import (
    ODESolver,
    ResultsData,
    numpy_api as cnp,
    io_callback,
    cond,
)

from ..framework import (
    IntegerTime,
    is_event_data,
    ZeroCrossingEvent,
)
from collimator import backend

if TYPE_CHECKING:
    import equinox as eqx
    from ..backend.ode_solver import ODESolverBase, ODESolverState
    from ..framework import ContextBase, SystemBase
    from ..framework.port import OutputPort
    from ..framework.event import PeriodicEventData, EventCollection
else:
    eqx = LazyLoader("eqx", globals(), "equinox")


__all__ = [
    "estimate_max_major_steps",
    "simulate",
    "Simulator",
]


def _raise_end_time_not_reached(tf, ctx_time, reason):
    if ((tf - ctx_time) / tf > 1e-3) and (
        reason != StepEndReason.TerminalEventTriggered
    ):
        raise RuntimeError(
            f"Simulator failed to reach specified end time. End time={tf}. "
            f"Reached time={ctx_time}. Try increasing Maximum Major Steps."
        )


@jax.jit
def error_end_time_not_reached(tf, ctx_time, reason):
    jax.debug.callback(_raise_end_time_not_reached, tf, ctx_time, reason)


def _raise_end_time_not_representable(tf, max_tf):
    if tf > max_tf:
        required_scale = tf / max_tf
        current_scale = IntegerTime.time_scale
        raise RuntimeError(
            " "
            f"Requested end time {tf} is greater than max representable time {max_tf}. "
            "Increase the time scale by setting `int_time_scale` in `SimulatorOptions`."
            f"Current time scale is {current_scale}, but this end time requires at least "
            f"time_scale={current_scale * required_scale}. The default value of 1e-12 "
            "(picosecond precision) is only capable of representing times up to ~0.3 "
            "years."
        )


@jax.jit
def error_end_time_not_representable(tf, max_tf):
    jax.debug.callback(_raise_end_time_not_representable, tf, max_tf)


def estimate_max_major_steps(
    system: SystemBase,
    tspan: tuple[float, float],
    max_major_step_length: float = None,
    safety_factor: int = 2,
) -> int:
    """Heuristic for estimating the required number of major steps.

    This is used to bound the number of iterations in the while loop in the
    `simulate` function when automatic differentiation is enabled.  The number
    of major steps is determined by the smallest discrete period in the system
    and the length of the simulation interval.  The number of major steps is
    bounded by the length of the simulation interval divided by the smallest
    discrete period, with a safety factor applied.  The safety factor accounts
    for unscheduled major steps that may be triggered by zero-crossing events.

    This function assumes static time variables, so cannot be called from within
    traced (JAX-transformed) functions.  This is typically the case when the
    beginning or end time of the simulation is a variable that will be
    differentiated.  In this case `estimate_max_major_steps` should be called
    statically ahead of time to determine a reasonable bound for `max_major_steps`.

    Args:
        system (SystemBase): The system to simulate.
        tspan (tuple[float, float]): The time interval to simulate over.
        max_major_step_length (float, optional): The maximum length of a major
            step. If provided, this will be used to bound the number of major
            steps. Otherwise it will be ignored.
        safety_factor (int, optional): The safety factor to apply to the number of
            major steps.  Defaults to 2.
    """
    # For autodiff of collimator.simulate, this path is not possible, JAX
    # throws an error. To work around this, create:
    #   options = SimulatorOptions(max_major_steps=<my value>)
    # outside collimator.simulate, and pass in like this:
    #   collimator.simulate(my_model, options=options)

    # Find the smallest period amongst the periodic events of the system
    if system.periodic_events.has_events or max_major_step_length is not None:
        # Initialize to infinity - will be overwritten by at least one conditional
        min_discrete_step = np.inf

        # Bound the number of major steps based on the smallest discrete period in
        # the system.
        if system.periodic_events.has_events:
            event_periods = jax.tree_util.tree_map(
                lambda event_data: event_data.period,
                system.periodic_events,
                is_leaf=is_event_data,
            )
            min_discrete_step = jax.tree_util.tree_reduce(min, event_periods)

        # Also bound the number of major steps based on the max major step length
        # in case that is shorter than any of the update periods.
        if max_major_step_length is not None:
            min_discrete_step = min(min_discrete_step, max_major_step_length)

        # in this case, we assume that, on average, major steps triggered by
        # zero crossing event, will be as frequent or less frequent than major steps
        # triggered by the smallest discrete period.
        # anything less than 100 is considered inadequate. user can override if they want this.
        max_major_steps = max(100, safety_factor * int(tspan[1] // min_discrete_step))
        logger.info(
            "max_major_steps=%s based on smallest discrete period=%s",
            max_major_steps,
            min_discrete_step,
        )
    else:
        # in this case we really have no valuable information on which to make an
        # educated guess. who knows how many events might occurr!!!
        # users will have to iterate.
        max_major_steps = 200
        logger.info(
            "max_major_steps=%s by default since no discrete period in system",
            max_major_steps,
        )
    return max_major_steps


def _check_options(
    system: SystemBase,
    options: SimulatorOptions,
    tspan: tuple[float, float],
    recorded_signals: dict[str, OutputPort],
) -> SimulatorOptions:
    """Check consistency of options and adjust settings where necessary."""

    if options is None:
        options = SimulatorOptions()

    # Check based on the options and the system whether JAX tracing is possible.
    math_backend, enable_tracing = _check_backend(options)

    # If we specified JAX but tracing is not enabled, we have fall back to numpy
    # TODO: Reconsider this logic - is there ever a time when untraced JAX is
    # useful, e.g. for debugging?
    if (math_backend == "jax") and not enable_tracing:
        logger.warning(
            "JAX backend is requested but JAX tracing is disabled. Falling back to "
            "numpy backend."
        )
        enable_tracing = False
        math_backend = "numpy"

    # Set the global numerical backend as determined by the options and above logic.
    cnp.set_backend(math_backend)

    if recorded_signals is None:
        recorded_signals = options.recorded_signals
    save_time_series = recorded_signals is not None

    # The while loop must be bounded in order for reverse-mode autodiff to work.
    # Also need this to set buffer sizes for signal recording in compiled JAX.
    # For the NumPy backend, this will be ignored, since neither bounded while
    # loops nor buffered recording is necessary.
    max_major_steps = options.max_major_steps
    if max_major_steps is None or max_major_steps <= 0:
        # logger.warning(
        #     "JAX backend requires a bounded number of major steps. This has not "
        #     "been specified in SimulatorOptions. Using a heuristic to estimate "
        #     "the maximum number of steps. If this fails, it may be because the "
        #     "final time is a traced variable.  If it is necessary to "
        #     "differentiate with respect to the end time of the simulation, then "
        #     "max_major_steps must be set manually. A reasonable value can be "
        #     "estimated using estimate_max_major_steps."
        # )
        max_major_steps = estimate_max_major_steps(
            system, tspan, options.max_major_step_length
        )

    buffer_length = options.buffer_length
    if buffer_length is None:
        buffer_length = max_major_steps

    # Check that the options are configured correctly for autodiff.
    if options.enable_autodiff:
        # JAX tracing is required for automatic differentiation
        if not enable_tracing:
            raise ValueError(
                "Autodiff is only supported with `options.enable_tracing=True`."
            )

        # Cannot record time series during autodiff - only final results can
        # be differentiated
        if save_time_series:
            raise ValueError(
                "Recording output time series is not supported with autodiff."
            )

    # Can optionally rescale integer time to allow for longer simulations or higher
    # precision.  The default integer time step corresponds to 1 picosecond, so
    # the default limit is around 0.3 years.  If the end time is greater than that but
    # less than ~300 years, we can use nanosecond integer time (set scale=1e-9), etc.
    if options.int_time_scale is not None:
        IntegerTime.set_scale(options.int_time_scale)
    error_end_time_not_representable(tspan[1], IntegerTime.max_float_time)

    return dataclasses.replace(
        options,
        recorded_signals=recorded_signals,
        save_time_series=save_time_series,
        max_major_steps=max_major_steps,
        math_backend=math_backend,
        enable_tracing=enable_tracing,
        buffer_length=buffer_length,
    )


def simulate(
    system: SystemBase,
    context: ContextBase,
    tspan: tuple[float, float],
    options: SimulatorOptions = None,
    results_options: ResultsOptions = None,
    recorded_signals: dict[str, OutputPort] = None,
    postprocess: bool = True,
) -> SimulationResults:
    """Simulate the hybrid dynamical system defined by `system`.

    The parameters and initial state are defined by `context`.  The simulation time
    runs from `tspan[0]` to `tspan[1]`.

    The simulation is "hybrid" in the sense that it handles dynamical systems with both
    discrete and continuous components.  The continuous components are integrated using
    an ODE solver, while discrete components are updated periodically as specified by
    the individual system components. The continuous and discrete states can also be
    modified by "zero-crossing" events, which trigger when scalar-valued guard
    functions cross zero in a specified direction.

    The simulation is thus broken into "major" steps, which consist of the following,
    in order:

    (1) Perform any periodic updates to the discrete state.
    (2) Check if the discrete update triggered any zero-crossing events and handle
        associated reset maps if necessary.
    (3) Advance the continuous state using an ODE solver until the next discrete
        update or zero-crossing, localizing the zero-crossing with a bisection search.
    (4) Store the results data.
    (5) If the ODE solver terminated due to a zero-crossing, handle the reset map.

    The steps taken by the ODE solver are "minor" steps in this simulation.  The
    behavior of the ODE solver and the hybrid simulation in general can be controlled
    by configuring `SimulatorOptions`.  Available settings are as follows:

    SimulatorOptions:
        enable_tracing (bool): Allow JAX tracing for JIT compilation
        max_major_step_length (float): Maximum length of a major step
        max_major_steps (int):
            The maximum number of major steps to take in the simulation. This is
            necessary for automatic differentiation - otherwise the "while" loop
            is non-differentiable.  With the default value of None, a heuristic
            is used to determine the maximum number of steps based on the periodic
            update events and time interval.
        rtol (float): Relative tolerance for the ODE solver. Default is 1e-6.
        atol (float): Absolute tolerance for the ODE solver. Default is 1e-8.
        min_minor_step_size (float): Minimum step size for the ODE solver.
        max_minor_step_size (float): Maximum step size for the ODE solver.
        ode_solver_method (str): The DE solver to use.  Default is "auto", which
            will use the Dopri5/Jax if JAX tracing is enabled, otherwise the
            SciPy Dopri5 solver.
        save_time_series (bool):
            This option determines whether the simulator saves any data.  If the
            simulation is initiated from `simulate` this will be set automatically
            depending on whether `recorded_signals` is provided.  Hence, this
            should not need to be manually configured.
        recorded_signals (dict[str, OutputPort]):
            Dictionary of ports or other cache sources for which the time series should
            be recorded. Note that if the simulation is initiated from `simulate` and
            `recorded_signals` is provided as a kwarg to `simulate`, anything set here
            will be overridden.  Hence, this should not need to be manually configured.
        return_context (bool):
            If the context is not needed for anything, opting to not return it can
            speed up compilation times.  For instance, typical simulation calls from
            the UI don't use the context for anything, so model_interface.py will
            set `return_context=False` for performance.
        postprocess (bool):
            If using buffered results recording (i.e. with JAX numerical backend), this
            determines whether to automatically trim the buffer after the simulation is
            complete. This is the default behavior, which will serve unless the full
            call to `simulate` needs to be traced (e.g. with `grad` or `vmap`).

    The return value is a `SimulationResults` object, which is a named tuple containing
    all recorded signals as well as the final context (if `options.return_context` is
    `True`). Signals can be recorded by providing a dict of (name, signal_source) pairs
    Typically the signal sources will be output ports, but they can actually be any
    `SystemCallback` object in the system.

    Args:
        system (SystemBase): The hybrid dynamical system to simulate.
        context (ContextBase): The initial state and parameters of the system.
        tspan (tuple[float, float]): The start and end times of the simulation.
        options (SimulatorOptions): Options for the simulation process and ODE solver.
        results_options (ResultsOptions): Options related to how the outputs are
            stored, interpolated, and returned.
        recorded_signals (dict[str, OutputPort]):
            Dictionary of ports for which the time series should be recorded.

    Returns:
        SimulationResults: A named tuple containing the recorded signals and the final
            context (if `options.return_context` is `True`).

    Notes:
        results_options is currently unused, pending:
            https://collimator.atlassian.net/browse/DASH-1350

        If `recorded_signals` is provided as a kwarg, it will override any entry in
        `options.recorded_signals`. This will be deprecated in the future in favor of
        only passing via `options`.
    """

    options = _check_options(system, options, tspan, recorded_signals)

    if results_options is None:
        results_options = ResultsOptions()

    if results_options.mode != ResultsMode.auto:
        raise NotImplementedError(
            f"Simulation output mode {results_options.mode.name} is not supported. "
            "Only 'auto' is presently supported."
        )

    # HACK: Wildcat presently does not use interpolant to produce
    # results sample between minor_step end times, so we clamp
    # the minor step size to the max_results_interval instead.
    if (
        results_options.max_results_interval is not None
        and results_options.max_results_interval > 0
        and results_options.max_results_interval < options.max_minor_step_size
    ):
        options = dataclasses.replace(
            options,
            max_minor_step_size=results_options.max_results_interval,
        )
        logger.info(
            "max_minor_step_size reduced to %s to match max_results_interval",
            options.max_minor_step_size,
        )

    ode_solver = ODESolver(system, options=options.ode_options)

    sim = Simulator(system, ode_solver=ode_solver, options=options)
    logger.info("Simulator ready to start: %s, %s", options, ode_solver)

    # Define a function to be traced by JAX, if allowed, closing over the
    # arguments to `_simulate`.
    def _wrapped_simulate() -> tuple[ContextBase, ResultsData]:
        t0, tf = tspan
        initial_context = context.with_time(t0)
        sim_state = sim.advance_to(tf, initial_context)
        error_end_time_not_reached(
            tf, sim_state.context.time, sim_state.step_end_reason
        )
        final_context = sim_state.context if options.return_context else None
        return final_context, sim_state.results_data

    # JIT-compile the simulation, if allowed
    if options.enable_tracing:
        _wrapped_simulate = jax.jit(_wrapped_simulate)
        _wrapped_simulate = Profiler.jaxjit_profiledfunc(
            _wrapped_simulate, "_wrapped_simulate"
        )

    # Run the simulation
    try:
        final_context, results_data = _wrapped_simulate()

        if postprocess and results_data is not None:
            time, outputs = results_data.finalize()
        else:
            time, outputs = None, None

    finally:
        system.post_simulation_finalize()

    # Reset the integer time scale to the default value in case we decreased precision
    # to reach the end time of a long simulation.  Typically this won't do anything.
    if options.int_time_scale is not None:
        IntegerTime.set_default_scale()

    return SimulationResults(
        final_context,
        time=time,
        outputs=outputs,
    )


class Simulator:
    """Class for orchestrating simulations of hybrid dynamical systems.

    See the `simulate` function for more details.
    """

    def __init__(
        self,
        system: SystemBase,
        ode_solver: ODESolverBase = None,
        options: SimulatorOptions = None,
    ):
        """Initialize the simulator.

        Args:
            system (SystemBase): The hybrid dynamical system to simulate.
            ode_solver (ODESolverBase):
                The ODE solver to use for integrating the continuous-time component
                of the system.  If not provided, a default solver will be used.
            options (SimulatorOptions):
                Options for the simulation process.  See `simulate` for details.
        """
        self.system = system

        if options is None:
            options = SimulatorOptions()

        # Determine whether JAX tracing can be used (jit, grad, vmap, etc)
        math_backend, self.enable_tracing = _check_backend(options)

        # Set the math backend
        cnp.set_backend(math_backend)

        # Should the simulation be run with autodiff enabled?  This will override
        # the `advance_to` method with a custom autodiff rule.
        self.enable_autodiff = options.enable_autodiff

        if ode_solver is None:
            ode_solver = ODESolver(system, options=options.ode_options)

        # Store configuration options
        self.buffer_length = options.buffer_length
        self.max_major_steps = options.max_major_steps
        self.max_major_step_length = options.max_major_step_length
        self.save_time_series = options.save_time_series
        self.recorded_outputs = options.recorded_signals
        self.zc_bisection_loop_count = options.zc_bisection_loop_count
        self.major_step_callback = options.major_step_callback

        if self.max_major_step_length is None:
            self.max_major_step_length = np.inf

        logger.debug("Simulator created with enable_tracing=%s", self.enable_tracing)

        self.ode_solver = ode_solver

        # Modify the default autodiff rule slightly to correctly capture variations
        # in end time of the simulation interval.
        self.has_terminal_events = system.zero_crossing_events.has_terminal_events
        self.advance_to = self._override_advance_to_vjp()

        # Also override the guarded ODE integration with a custom autodiff rule
        # to capture variations due to zero-crossing time.
        self.guarded_integrate = self._override_guarded_integrate_vjp()

    def while_loop(self, cond_fun, body_fun, val):
        """Structured control flow primitive for a while loop.

        Dispatches to a bounded while loop as necessary for autodiff. Otherwise
        it will call the "backend" implementation, which will either be
        `lax.while_loop` when JAX is the backend, or a pure-Python implementation
        when using NumPy.
        """
        # If autodiff is enabled, we need to use a custom while loop with a maximum
        # number of steps so that the loop is reverse-mode differentiable.
        # Otherwise we can use a standard unbounded while loop with lax backend.
        if self.enable_autodiff:
            return _bounded_while_loop(cond_fun, body_fun, val, self.max_major_steps)
        else:
            return backend.while_loop(cond_fun, body_fun, val)

    def initialize(self, context: ContextBase) -> SimulatorState:
        """Perform initial setup for the simulation."""
        logger.debug("Initializing simulator")
        # context.state.pprint(logger.debug)

        # Initial simulation time as integer (picoseconds)
        initial_int_time = IntegerTime.from_decimal(context.time)

        # Ensure that _next_update_time() can return the current time by perturbing
        # current time as slightly toward negative infinity as possible
        time_of_next_timed_event, timed_events = _next_update_time(
            self.system.periodic_events, initial_int_time - 1
        )

        # timed_events is now marked with the active events at the next update time
        logger.debug("Time of next timed event (int): %s", time_of_next_timed_event)
        logger.debug(
            "Time of next event (sec): %s",
            IntegerTime.as_decimal(time_of_next_timed_event),
        )
        timed_events.pprint(logger.debug)

        end_reason = cnp.where(
            time_of_next_timed_event == initial_int_time,
            StepEndReason.TimeTriggered,
            StepEndReason.NothingTriggered,
        )

        # Initialize the results data that will hold recorded time series data.
        if self.save_time_series:
            results_data = ResultsData.initialize(
                context, self.recorded_outputs, self.buffer_length
            )
        else:
            results_data = None

        return SimulatorState(
            context=context,
            timed_events=timed_events,
            step_end_reason=end_reason,
            int_time=initial_int_time,
            results_data=results_data,
            ode_solver_state=self.ode_solver.initialize(context),
        )

    def save_results(
        self, results_data: ResultsData, context: ContextBase
    ) -> ResultsData:
        """Update the results data with the current state of the system."""
        if not self.save_time_series:
            return results_data
        return results_data.update(context)

    def _override_advance_to_vjp(self) -> Callable:
        """Construct the `advance_to` method for the simulator.

        See the docstring for `Simulator._advance_to` for details.

        If JAX tracing is enabled for autodiff, wrap the advance function with a
        custom autodiff rule to correctly capture variation with respect to end
        time. If somehow autodiff works with tracing disabled, the derivatives will
        not account for possible variations in end time (for instance in finding
        limit cycles or when there are terminal conditions on the simulation).
        """
        if not self.enable_autodiff:
            return self._advance_to

        # This is the main function call whose autodiff rule will be overridden.
        def _wrapped_advance_to(
            sim: Simulator, boundary_time: float, context: ContextBase
        ) -> SimulatorState:
            return sim._advance_to(boundary_time, context)

        # The "forwards pass" to advance the simulation.  Also stores the nominal
        # VJP calculation and the continuous time derivative value, both of which
        # will be needed in the backwards pass.
        def _wrapped_advance_to_fwd(
            sim: Simulator, boundary_time: float, context: ContextBase
        ) -> tuple[SimulatorState, tuple]:
            primals, vjp_fun = jax.vjp(sim._advance_to, boundary_time, context)

            # Also need to keep the final continuous time derivative value for
            # computing the adjoint time variable
            xdot = sim.system.eval_time_derivatives(primals.context)

            # "Residual" information needed in the backwards pass
            res = (vjp_fun, xdot, primals.step_end_reason)

            return primals, res

        def _wrapped_advance_to_adj(
            _sim: Simulator, res: tuple, adjoints: SimulatorState
        ) -> tuple[float, ContextBase]:
            # Unpack the residuals from the forward pass
            vjp_fun, xdot, reason = res

            # Compute whatever the standard adjoint variables are using the
            # automatically derived VJP function.  The first return variable will
            # be the automatically computed tf_adj value, which we will ignore in
            # favor of the manually derived value computed next.
            _, context_adj = vjp_fun(adjoints)

            # The derivative with respect to end time is just the dot product of
            # the adjoint "seed" continuous state with the final time derivatives.
            # We can overwrite whatever the calculated adjoint time was with this.
            vc = adjoints.context.continuous_state
            vT_xdot = jax.tree_util.tree_map(
                lambda xdot, vc: jnp.dot(xdot, vc), xdot, vc
            )

            # On the other hand, if the simulation ends early due to a terminal
            # event, then the derivative with respect to end time is zero.
            tf_adj = jnp.where(
                reason == StepEndReason.TerminalEventTriggered,
                0.0,
                sum(jax.tree_util.tree_leaves(vT_xdot)),
            )

            # Return adjoints to match the inputs to _wrapped_advance_to, except for
            # the first argument (Simulator), which will be marked nondifferentiable.
            return (tf_adj, context_adj)

        advance_to = jax.custom_vjp(_wrapped_advance_to, nondiff_argnums=(0,))
        advance_to.defvjp(_wrapped_advance_to_fwd, _wrapped_advance_to_adj)

        # Copy the docstring and type hints to the overridden function
        advance_to.__doc__ = self._advance_to.__doc__
        advance_to.__annotations__ = self._advance_to.__annotations__

        return partial(advance_to, self)

    def _guarded_integrate(
        self,
        solver_state: ODESolverState,
        results_data: ResultsData,
        tf: float,
        context: ContextBase,
        zc_events: EventCollection,
    ) -> tuple[bool, ODESolverState, ContextBase, ResultsData, EventCollection]:
        """Guarded ODE integration.

        Advance continuous time using an ODE solver, localizing any zero-crossing events
        that occur during the requested interval.  If any zero-crossing events trigger,
        the dense interpolant is used to localize the events and the associated reset maps
        are handled.  The method then returns, guaranteeing that the major step terminates
        either at the end of the requested interval or at the time of a zero-crossing
        event.

        Args:
            solver_state (ODESolverState): The current state of the ODE solver.
            results_data (ResultsData): The results data that will hold recorded time
                series data.
            tf (float): The end time of the integration interval.
            context (ContextBase): The current state of the system.
            zc_events (EventCollection): The current zero-crossing events.

        Returns:
            tuple[bool, ODESolverState, ContextBase, ResultsData, EventCollection]:
                A tuple containing the following:
                - A boolean indicating whether the major step was terminated early due to
                  a zero-crossing event.
                - The updated state of the ODE solver.
                - The updated state of the system.
                - The updated results data.
                - The updated zero-crossing events.
        """
        solver = self.ode_solver
        func = solver.flat_ode_rhs  # Raveled ODE RHS function

        # Close over the additional arguments so that the RHS function has the
        # signature `func(y, t)`.
        def _func(y, t):
            return func(y, t, context)

        def _localize_zc_minor(
            solver_state, context_t0, context_tf, zc_events, results_data
        ):
            # Using the ODE solver interpolant, employ bisection to find a 'small' time
            # interval within which the earliest zero crossing occurrs. See
            # _bisection_step_fun for details about how bisection is employed for
            # localizing the zero crossing in time.
            int_t1 = IntegerTime.from_decimal(context_tf.time)
            int_t0 = IntegerTime.from_decimal(context_t0.time)
            _body_fun = partial(_bisection_step_fun, solver_state)
            carry = GuardIsolationData(int_t0, int_t1, zc_events, context_tf)
            search_data = backend.fori_loop(
                0, self.zc_bisection_loop_count, _body_fun, carry
            )
            context_tf = search_data.context
            zc_events = search_data.guards

            # record results sample for the ZC having 'occurred'
            minor_step_end_time = IntegerTime.as_decimal(int_t1)
            minor_step_start_time = IntegerTime.as_decimal(int_t0)
            zc_occur_time = context_tf.time - (
                minor_step_end_time - minor_step_start_time
            ) / (2 ** (self.zc_bisection_loop_count + 1))
            context_zc_time = context_tf.with_time(zc_occur_time)
            results_data = self.save_results(results_data, context_zc_time)

            # Handle any triggered zero-crossing events
            context_tf = self.system.handle_zero_crossings(zc_events, context_tf)

            # Re-initialize the solver, since the state may have been reset
            # Keep the last step size, since there's no reason to assume that the
            # dynamics have changed significantly as a result of the event.  If that is
            # the case, then we're relying on the adaptive stepping to re-calibrate.
            #
            # NOTE: this previously only updated the state and time of
            # the solver state, but with multistep solvers (e.g. BDF), the
            # solver needs to be fully reinitialized because the history
            # of differences needs to be cleared and rebuilt over the next few steps.
            solver_state = solver.initialize(context_tf)
            return solver_state, context_tf, zc_events, results_data

        def _no_events_fun(
            solver_state, context_t0, context_tf, zc_events, results_data
        ):
            return solver_state, context_tf, zc_events, results_data

        def _ode_step(carry):
            _, solver_state, context_t0, results_data, zc_events = carry

            # Save results at the top of the loop. This will save data at t=t0,
            # but not at t=tf.  This is okay, since we will save the results at
            # the top of the next major step, as well as at the end of the main
            # simulation loop.
            results_data = self.save_results(results_data, context_t0)

            zc_events = guard_interval_start(zc_events, context_t0)

            # Advance ODE solver
            solver_state = solver.step(_func, tf, solver_state)
            xc = solver_state.unraveled_state
            context = context_t0.with_time(solver_state.t).with_continuous_state(xc)

            # Check for zero-crossing events
            zc_events = determine_triggered_guards(zc_events, context)

            triggered = zc_events.has_triggered

            args = (solver_state, context_t0, context, zc_events, results_data)
            solver_state, context, zc_events, results_data = backend.cond(
                triggered, _localize_zc_minor, _no_events_fun, *args
            )

            return (triggered, solver_state, context, results_data, zc_events)

        def _cond_fun(carry):
            triggered, solver_state, _, _, _ = carry
            return (solver_state.t < tf) & (~triggered)

        carry = (False, solver_state, context, results_data, zc_events)
        triggered, solver_state, context, results_data, zc_events = backend.while_loop(
            _cond_fun,
            _ode_step,
            carry,
        )

        return triggered, solver_state, context, results_data, zc_events

    def _override_guarded_integrate_vjp(self):
        if not self.enable_autodiff:
            return self._guarded_integrate

        def _wrapped_solve(
            self: Simulator, solver_state, results_data, tf, context, zc_events
        ):
            return self._guarded_integrate(
                solver_state, results_data, tf, context, zc_events
            )

        def _wrapped_solve_fwd(
            self: Simulator, solver_state, _results_data, tf, context, zc_events
        ):
            # Run the forward major step as usual (primal calculation). Do not save
            # any results here

            # The return from the forward call has the state post-reset, including the
            # solver state, the context, and the zero-crossing events.
            t0 = solver_state.t

            (
                triggered,
                solver_state_out,
                context_out,
                _,
                zc_events_out,
            ) = self._guarded_integrate(solver_state, None, tf, context, zc_events)
            tf = solver_state_out.t

            # Define a differentiable function for the forward pass, knowing where the
            # zero-crossing occurs. Note that `tf` here should be the _actual_ interval
            # end time, not the requested end time.
            solver = self.ode_solver
            func = solver.flat_ode_rhs  # Raveled ODE RHS function

            def _forward(solver_state, tf, context, zc_events_out):
                solver_state_out = _odeint(solver, func, solver_state, tf, context)
                context = context.with_time(solver_state_out.t)
                context = context.with_continuous_state(
                    solver_state_out.unraveled_state
                )
                context = self.system.handle_zero_crossings(zc_events_out, context)
                return context

            # Get the VJP for the event handling
            _primals, vjp_fun = jax.vjp(
                _forward, solver_state, tf, context, zc_events_out
            )

            primals = (triggered, solver_state_out, context_out, None, zc_events_out)
            residuals = (triggered, solver_state_out, t0, tf, context, vjp_fun)
            return primals, residuals

        def _wrapped_solve_adj(self: Simulator, residuals, adjoints):
            triggered, primal_solver_state, t0, tf, context, vjp_fun = residuals
            (
                _triggered_adj,
                solver_state_adj,
                context_adj,
                _results_data_adj,
                _zc_events_adj,
            ) = adjoints

            context_adj = context_adj.with_time(solver_state_adj.t)
            context_adj = context_adj.with_continuous_state(
                solver_state_adj.unraveled_state
            )

            # The `_forward` function corresponding to `vjp_fun` has the signature
            # `context_out = _forward(solver_state, tf, context, zc_events_out)`.
            # For the adjoint, we have to call with `context_adj` as the input:
            solver_state_adj, tf_adj, context_adj, zc_events_adj = vjp_fun(context_adj)

            # The Jacobian with respect to the final time is just the time derivative of
            # the state at the final time.
            yf = primal_solver_state.y
            yf_bar = solver_state_adj.y
            func = self.ode_solver.flat_ode_rhs  # Raveled ODE RHS function
            tf_adj = jnp.dot(func(yf, tf, context), yf_bar)

            tf_adj = jnp.where(
                triggered,
                tf_adj,
                jnp.zeros_like(tf_adj),
            )

            return (solver_state_adj, None, tf_adj, context_adj, zc_events_adj)

        guarded_integrate = jax.custom_vjp(_wrapped_solve, nondiff_argnums=(0,))
        guarded_integrate.defvjp(_wrapped_solve_fwd, _wrapped_solve_adj)

        # Copy the docstring and type hints to the overridden function
        guarded_integrate.__doc__ = self._guarded_integrate.__doc__
        guarded_integrate.__annotations__ = self._guarded_integrate.__annotations__

        return partial(guarded_integrate, self)

    def _advance_continuous_time(
        self,
        cdata: ContinuousIntervalData,
    ) -> ContinuousIntervalData:
        """Advance the simulation to the next discrete update or zero-crossing event.

        This stores the values of all active guard functions and advances the
        continuous-time component of the system to the next discrete update or
        zero-crossing event, whichever comes first.  Zero-crossing events are
        localized using a bisection search defined by `_trigger_search`, which will
        also record the final guard function values at the end of the search interval
        and determine which (if any) zero-crossing events were triggered.
        """

        # Unpack inputs
        int_tf = cdata.tf
        context = cdata.context
        results_data = cdata.results_data

        zc_events = self.system.determine_active_guards(context)

        if self.system.has_continuous_state:
            solver_state = cdata.ode_solver_state
            tf = IntegerTime.as_decimal(int_tf)

            (
                triggered,
                solver_state,
                context,
                results_data,
                zc_events,
            ) = self.guarded_integrate(
                solver_state,
                results_data,
                tf,
                context,
                zc_events,
            )

            context = context.with_time(solver_state.t)
            context = context.with_continuous_state(solver_state.unraveled_state)

            # Converting from decimal -> integer time incurs a loss of precision.  This is
            # okay for unscheduled zero-crossing events, but problematic for timed events.
            # So only do this conversion if a zero-crossing was triggered.  Otherwise we
            # know we have reached the end of the interval and can keep the requested end
            # time.
            int_tf = cnp.where(
                triggered,
                IntegerTime.from_decimal(context.time),
                int_tf,
            )

        else:
            # Skip the ODE solver for systems without continuous state.  We still
            # have to check for triggered events here in case there are any
            # transitions triggered by time that need to be handled before the
            # periodic discrete update at the top of the next major step
            triggered = False
            solver_state = cdata.ode_solver_state

            zc_events = guard_interval_start(zc_events, context)
            results_data = self.save_results(results_data, context)

            # Advance time to the end of the interval
            context = context.with_time(IntegerTime.as_decimal(int_tf))

            # Record guard values after the discrete update and check if anything
            # triggered as a result of advancing time
            zc_events = guard_interval_end(zc_events, context)
            zc_events = determine_triggered_guards(zc_events, context)

            # Handle any triggered zero-crossing events
            context = self.system.handle_zero_crossings(zc_events, context)

        # Even though the zero-crossing events have already been "handled", the
        # information about whether a terminal event has been triggered is still in
        # the events collection (since "triggered" has not been cleared by a call
        # to determine_triggered_guards).
        terminate_early = zc_events.has_active_terminal

        return cdata._replace(
            triggered=triggered,
            terminate_early=terminate_early,
            context=context,
            tf=int_tf,
            results_data=results_data,
            ode_solver_state=solver_state,
        )

    def _handle_discrete_update(
        self, context: ContextBase, timed_events: EventCollection
    ) -> tuple[ContextBase, bool]:
        """Handle discrete updates triggered by time.

        This method is called at the beginning of each major step to handle any
        discrete updates that are triggered by time.  This includes both discrete
        updates that are triggered by time and any zero-crossing events that are
        triggered by the discrete update.

        This will also work when there are no zero-crossing events: the zero-crossing
        collection will be empty and only the periodic discrete update will happen.

        Args:
            context (ContextBase): The current state of the system.
            timed_events (EventCollection):
                The collection of timed events, with the active events marked.

        Returns:
            ContextBase: The updated state of the system.
            bool: Whether the simulation should terminate early as a result of a
                triggered terminal condition.
        """
        system = self.system

        # Get the collection of zero-crossing events that _might_ be activated
        # given the current state of the system.  For example, some events may
        # be de-activated as a result of the current state of a state machine.
        zc_events = system.determine_active_guards(context)

        # Record guard values at the start of the interval
        zc_events = guard_interval_start(zc_events, context)

        # Handle any active periodic discrete updates
        context = system.handle_discrete_update(timed_events, context)

        # Record guard values after the discrete update
        zc_events = guard_interval_end(zc_events, context)

        # Check if guards have triggered as a result of these updates
        zc_events = determine_triggered_guards(zc_events, context)
        terminate_early = zc_events.has_active_terminal

        # Handle any zero-crossing events that were triggered
        context = system.handle_zero_crossings(zc_events, context)

        return context, terminate_early

    # This method is marked private because it will be wrapped with a custom autodiff
    # rule to get the correct derivatives with respect to the end time of the
    # simulation interval using `_override_advance_to_vjp`.  This also copies the
    # docstring to the overridden function. Normally the wrapped attribute `advance_to`
    # is what should be called by users.
    def _advance_to(self, boundary_time: float, context: ContextBase) -> SimulatorState:
        """Core control flow logic for running a simulation.

        This is the main loop for advancing the simulation.  It is called by `simulate`
        or can be called directly if more fine-grained control is needed. This method
        essentially loops over "major steps" until the boundary time is reached. See
        the documentation for `simulate` for details on the order of operations in a
        major step.

        Args:
            boundary_time (float): The time to advance to.
            context (ContextBase): The current state of the system.

        Returns:
            SimulatorState:
                A named tuple containing the final state of the simulation, including
                the final context, a collection of pending timed events, and a flag
                indicating the reason that the most recent major step ended.

        Notes:
            API will change slightly as a result of WC-87, which will break out the
            initialization from the main loop so that `advance_to` can be called
            repeatedly.  See:
            https://collimator.atlassian.net/browse/WC-87
        """

        system = self.system
        sim_state = self.initialize(context)
        end_reason = sim_state.step_end_reason
        context = sim_state.context
        timed_events = sim_state.timed_events
        int_boundary_time = IntegerTime.from_decimal(boundary_time)

        # We will be limiting each step by the max_major_step_length.  However, if this
        # is infinite we should just use the end time of the simulation to avoid
        # integer overflow.  This could be problematic if the end time of the
        # simulation is close to the maximum representable integer time, but we can come
        # back to that if it's an issue.
        int_max_step_length = IntegerTime.from_decimal(
            cnp.minimum(self.max_major_step_length, boundary_time)
        )

        # Only activate timed events if the major step ended on a time trigger
        timed_events = activate_timed_events(timed_events, end_reason)

        # Called on the "True" branch of the conditional
        def _major_step(sim_state: SimulatorState) -> SimulatorState:
            end_reason = sim_state.step_end_reason
            context = sim_state.context
            timed_events = sim_state.timed_events
            int_time = sim_state.int_time

            if not self.enable_tracing:
                logger.debug("Starting a simulation step at t=%s", context.time)
                logger.debug("   merged_events: %s", timed_events)

            # Handle any discrete updates that are triggered by time along with
            # any zero-crossing events that are triggered by the discrete update.
            context, terminate_early = self._handle_discrete_update(
                context, timed_events
            )
            logger.debug("Terminate early after discrete update: %s", terminate_early)

            # How far can we go before we have to handle timed events?
            # The time returned here will be the integer time representation.
            time_of_next_timed_event, timed_events = _next_update_time(
                system.periodic_events, int_time
            )
            if not self.enable_tracing:
                logger.debug(
                    "Next timed event at t=%s",
                    IntegerTime.as_decimal(time_of_next_timed_event),
                )
                timed_events.pprint(logger.debug)

            # Determine whether the events include a timed update
            update_time = IntegerTime.max_int_time

            if timed_events.num_events > 0:
                update_time = time_of_next_timed_event

            # Limit the major step end time to the simulation end time, major step limit,
            # or next periodic update time.
            # This is the mechanism used to advance time for systems that have
            # no states and no periodic events.
            # Discrete systems] when there are discrete periodic events, we use those
            # to determine each major step end time.
            # Feedthrough system] when there are just feedthrough blocks (no states or
            # events), use max_major_step_length to determine each major step end time.
            int_tf_limit = int_time + int_max_step_length
            int_tf = cnp.min(
                cnp.array(
                    [
                        int_boundary_time,
                        int_tf_limit,
                        update_time,
                    ]
                )
            )
            if not self.enable_tracing:
                logger.debug(
                    "Expecting to integrate to t=%s",
                    IntegerTime.as_decimal(int_tf),
                )

            # Normally we will advance continuous time to the end of the major step
            # here. However, if a terminal event was triggered as part of the discrete
            # update, we should respect that and skip the continuous update.
            #
            # Construct the container used to hold various data related to advancing
            # continuous time.  This is passed to ODE solvers, zero-crossing
            # localization, and related functions.
            cdata = ContinuousIntervalData(
                context=context,
                terminate_early=terminate_early,
                triggered=False,
                t0=int_time,
                tf=int_tf,
                results_data=sim_state.results_data,
                ode_solver_state=sim_state.ode_solver_state,
            )
            cdata = backend.cond(
                (self.has_terminal_events & cdata.terminate_early),
                lambda cdata: cdata,  # Terminal event triggered - return immediately
                self._advance_continuous_time,  # Advance continuous time normally
                cdata,
            )

            # Unpack the results of the continuous time advance
            context = cdata.context
            terminate_early = cdata.terminate_early
            triggered = cdata.triggered
            int_tf = cdata.tf
            results_data = cdata.results_data
            ode_solver_state = cdata.ode_solver_state

            # Determine the reason why the major step ended.  Did a zero-crossing
            # trigger, did a timed event trigger, neither, or both?
            # terminate_early = terminate_early | zc_events.has_active_terminal
            logger.debug("Terminate early after major step: %s", terminate_early)
            end_reason = _determine_step_end_reason(
                triggered, terminate_early, int_tf, update_time
            )
            logger.debug("Major step end reason: %s", end_reason)

            # Conditionally activate timed events depending on whether the major step
            # ended as a result of a time trigger or zero-crossing event.
            timed_events = activate_timed_events(timed_events, end_reason)

            if self.major_step_callback:
                io_callback(self.major_step_callback, (), context.time)

            return SimulatorState(
                step_end_reason=end_reason,
                context=context,
                timed_events=timed_events,
                int_time=int_tf,
                results_data=results_data,
                ode_solver_state=ode_solver_state,
            )

        def _cond_fun(sim_state: SimulatorState):
            return (sim_state.int_time < int_boundary_time) & (
                sim_state.step_end_reason != StepEndReason.TerminalEventTriggered
            )

        # Initialize the "carry" values for the main loop.
        sim_state = SimulatorState(
            context=context,
            timed_events=timed_events,
            step_end_reason=end_reason,
            int_time=sim_state.int_time,
            results_data=sim_state.results_data,
            ode_solver_state=sim_state.ode_solver_state,
        )

        logger.debug(
            "Running simulation from t=%s to t=%s", context.time, boundary_time
        )

        try:
            # Main loop call
            sim_state = self.while_loop(_cond_fun, _major_step, sim_state)
            logger.debug("Simulation complete at t=%s", sim_state.context.time)
        except KeyboardInterrupt:
            # TODO: flag simulation as interrupted somewhere in sim_state
            logger.info("Simulation interrupted at t=%s", sim_state.context.time)

        # At the end of the simulation we need to handle any pending discrete updates
        # and store the solution one last time.
        # FIXME (WC-87): The returned simulator state can't be used with advance_to again,
        # since the discrete updates have already been performed. Should be broken out
        # into a `finalize` method as part of WC-87.

        # update discrete state to x+ at the simulation end_time
        if self.save_time_series:
            logger.debug("Finalizing solution...")
            # 1] do discrete update (will skip if the simulation was terminated early)
            context, _terminate_early = self._handle_discrete_update(
                sim_state.context, sim_state.timed_events
            )
            # 2] do update solution
            results_data = self.save_results(sim_state.results_data, context)
            sim_state = sim_state._replace(
                context=context,
                results_data=results_data,
            )
            logger.debug("Done finalizing solution")

        return sim_state


def _bounded_while_loop(
    cond_fun: Callable,
    body_fun: Callable,
    val: Any,
    max_steps: int,
) -> Any:
    """Run a while loop with a bounded number of steps.

    This is a workaround for the fact that JAX's `lax.while_loop` does not support
    reverse-mode autodiff.  The `max_steps` bound can usually be determined
    automatically during calls to `simulate` - see notes on `max_major_steps` in
    `SimulatorOptions` and `estimate_max_major_steps`.
    """

    def _loop_fun(_i, val):
        return backend.cond(
            cond_fun(val),
            body_fun,
            lambda val: val,
            val,
        )

    return backend.fori_loop(0, max_steps, _loop_fun, val)


def _check_backend(options: SimulatorOptions) -> tuple[str, bool]:
    """Check if JAX tracing can be used to simulate this system."""

    math_backend = options.math_backend or "auto"
    if math_backend == "auto":
        math_backend = backend.active_backend

    if math_backend != "jax":
        enable_tracing = False

    else:
        # Otherwise return whatever `options` requested
        enable_tracing = options.enable_tracing

    return math_backend, enable_tracing


def _determine_step_end_reason(
    guard_triggered: bool,
    terminate_early: bool,
    tf: int,
    update_time: int,
) -> StepEndReason:
    """Determine the reason why the major step ended."""
    logger.debug("[_determine_step_end_reason]: tf=%s, update_time=%s", tf, update_time)
    logger.debug("[_determine_step_end_reason]: guard_triggered=%s", guard_triggered)

    # If the integration terminated due to a triggered event, determine whether
    # there are any other events that should be triggered at the same time.
    guard_reason = cnp.where(
        tf == update_time,
        StepEndReason.BothTriggered,
        StepEndReason.GuardTriggered,
    )

    # No guard triggered; handle integration as usual.
    no_guard_reason = cnp.where(
        tf == update_time,
        StepEndReason.TimeTriggered,
        StepEndReason.NothingTriggered,
    )

    reason = cnp.where(guard_triggered, guard_reason, no_guard_reason)

    # No matter why the integration terminated, if a "terminal" event is also
    # active, that will be the overriding reason for the termination.
    return cnp.where(terminate_early, StepEndReason.TerminalEventTriggered, reason)


def _next_sample_time(current_time: int, event_data: PeriodicEventData) -> int:
    """Determine when the specified periodic event happens next.

    This is a helper function for `_next_update_time` for a specific event.
    """

    period, offset = event_data.period_int, event_data.offset_int

    # If we shift the current time by the offset, what would the index of the
    # next periodic sample time be?  This tells us how many samples from the
    # offset we are in either direction.  For example, if offset=dt and t=0,
    # the next "k" value will be -1.
    next_k = (current_time - offset) // period

    # What would the next periodic sample time be?  If the period is infinite,
    # the next sample time is also infinite.  This value is shifted back to the
    # original time frame by adding the offset.  If the sample is more than one
    # period away from the offset, this will be negative.
    next_t = cnp.where(
        cnp.isfinite(event_data.period),
        offset + next_k * period,
        period,
    )

    # If we are in between samples, next_t should be strictly greater than
    # the current time and that should be used as the target major step end time.
    # However, if we are at t = offset + k * period for some k, then the
    # calculation above will give us next_k = k and therefore next_t = t.
    # In this case we should bump to the next time in the series.
    next_sequence_time = cnp.where(
        next_t > current_time,
        next_t,
        offset + (next_k + 1) * period,
    )

    return cnp.where(
        current_time < offset,
        offset,
        next_sequence_time,
    )


def _next_update_time(periodic_events: EventCollection, current_time: int) -> int:
    """Compute next update time over all events in the periodic_events collection.

    This returns a tuple of the minimum next sample time along with a pytree with
    the same structure as `periodic_events` indicating which events are active at
    the next sample time.
    """
    periodic_events = periodic_events.mark_all_inactive()

    # 0. If no events, return an infinite time and empty event collection
    if not periodic_events.has_events:
        return IntegerTime.max_int_time, periodic_events

    # 1. Compute the next sample time for each event
    def _replace_sample_time(event_data):
        return dataclasses.replace(
            event_data,
            next_sample_time=_next_sample_time(current_time, event_data),
        )

    timed_events = jax.tree_util.tree_map(
        _replace_sample_time,
        periodic_events,
        is_leaf=is_event_data,
    )

    def _get_next_sample_time(event_data: PeriodicEventData) -> int:
        return event_data.next_sample_time

    # 2. Find the minimum next sample time across all events
    min_time = jax.tree_util.tree_reduce(
        cnp.minimum,
        jax.tree_util.tree_map(
            _get_next_sample_time,
            timed_events,
            is_leaf=is_event_data,
        ),
    )

    # 3. Find the events corresponding to the minimum time by updating the event data `active` field
    def _replace_active(event_data: PeriodicEventData):
        return dataclasses.replace(
            event_data,
            active=(event_data.next_sample_time == min_time),
        )

    active_events = jax.tree_util.tree_map(
        _replace_active,
        timed_events,
        is_leaf=is_event_data,
    )
    return min_time, active_events


def activate_timed_events(
    timed_events: EventCollection, end_reason: StepEndReason
) -> EventCollection:
    """Conditionally activate timed events.

    Only activate timed events if the major step ended on a time trigger and
    the event was already marked active (by the timing calculation). This will
    deactivate timed events if they were pre-empted by a zero-crossing.
    """

    deactivate = (end_reason != StepEndReason.TimeTriggered) & (
        end_reason != StepEndReason.BothTriggered
    )

    def activation_fn(event_data: PeriodicEventData):
        return event_data.active & ~deactivate

    return timed_events.activate(activation_fn)


def _is_zc_event(x):
    return isinstance(x, ZeroCrossingEvent)


def _record_guard_values(
    events: EventCollection, context: ContextBase, key: str
) -> EventCollection:
    """Store the current values of guard functions in the event data.

    The "key" can either be `"w0"` or `"w1"` to indicate whether the recorded
    values correspond to the start or end of the interval.
    """

    # Set the `w0`/`w1` field of event_data by evaluating the guard functions
    def _update(event: ZeroCrossingEvent):
        return dataclasses.replace(
            event,
            event_data=dataclasses.replace(
                event.event_data,
                **{key: event.guard(context)},
            ),
        )

    return jax.tree_util.tree_map(_update, events, is_leaf=_is_zc_event)


# Convenient partial functions for the two valid values of `key`.
guard_interval_start = partial(_record_guard_values, key="w0")
guard_interval_end = partial(_record_guard_values, key="w1")


def determine_triggered_guards(
    events: EventCollection, context: ContextBase
) -> EventCollection:
    """Determine which zero-crossing events are triggered.

    This is done by evaluating the guard functions at the end of the interval
    and comparing the sign of the values to the sign of the values at the
    beginning of the interval, using the "direction" rule for each individual
    event.

    The returned collection has the same pytree structure as the input, but
    the `active` field of the event data will be set to `True` for any events
    that triggered.
    """
    events = guard_interval_end(events, context)

    def _update(event: ZeroCrossingEvent):
        return dataclasses.replace(
            event,
            event_data=dataclasses.replace(
                event.event_data,
                triggered=event.should_trigger(),
            ),
        )

    return jax.tree_util.tree_map(_update, events, is_leaf=_is_zc_event)


def _bisection_step_fun(step_sol, i, carry: GuardIsolationData):
    """Perform one step of bisection.

    Appropriately return a new zero crossing localization time interval.

    Bisection is employed as follows:
        1] split t0->tf into t0->t_mid0->tf
        2] set guard interval end at t_mid
        3] check if any guards triggered
        4] if so, set t0=t0, and tf=t_mid0,
            which, when we return to step 1, we'll split t0->t_mid to t0->t_mid1->t_mid0,
            and check for triggers in t0->t_mid1
        5] else, set t0=t_mid0 and tf=tf,
            which, when we return to step 1, we'll split t_mid0->tf to t_mid0->t_mid1->tf,
            and check for triggers in t_mid0->t_mid1
    The process repeats until we reach a termination condition. Presently, the termination
    condition is just a fixed number of iterations.

    Note that `carry.context` should be the context at the _end_ of the search
    interval.  This allows the return value of `carry.context` to guarantee
    that it will contain a zero-crossing event.
    """

    # bisection algo part 1: find mid point (integer time stamp)
    int_time_mid = (
        carry.zc_before_time + (carry.zc_after_time - carry.zc_before_time) // 2
    )
    time_mid = IntegerTime.as_decimal(int_time_mid)

    # check for triggers in interval
    context_mid = carry.context.with_time(time_mid)
    states_mid = step_sol.eval_interpolant(time_mid)
    context_mid = context_mid.with_continuous_state(states_mid)
    guards_mid = determine_triggered_guards(carry.guards, context_mid)

    # bisection algo part 2: decide whether next step will search
    # the first half, or the second half of the current interval.
    # Replacing the carry data with "mid" ensures that the event collection
    # and context always refer to a state in which an event triggered
    carry_first_half = GuardIsolationData(
        carry.zc_before_time,
        int_time_mid,
        guards_mid,
        context_mid,
    )

    carry_second_half = GuardIsolationData(
        int_time_mid,
        carry.zc_after_time,
        carry.guards,
        carry.context,
    )

    return cond(
        guards_mid.has_triggered,
        lambda: carry_first_half,
        lambda: carry_second_half,
    )


#
# Custom VJP for advancing continuous time with an ODE solver
#
#
@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _odeint(solver: ODESolverBase, ode_rhs, solver_state, tf, context):
    """Unguarded ODE integration.

    This function unconditionally advances time to the end of an interval.
    It does not check for any zero-crossing events.  As such, it is not actually
    used directly in the simulation loop.  However, the adjoint (autodiff rule) for
    the unguarded ODE solve is more straightforward than the guarded version, so
    this is wrapped and called by the autodiff rule for the guarded ODE solve.

    Since it is _only_ called in the forward pass of the custom autodiff rule for
    the guarded ODE solve, it does not need to be conditionally wrapped by the simulator
    as the guarded solve does. Hence we can define it as a standalone function.
    """

    # Close over the additional arguments so that the RHS function has the
    # signature `func(y, t)`.  We can't do this anywhere upstream because
    # the data in the context has to be differentiable.
    def _func(y, t):
        return ode_rhs(y, t, context)

    def _ode_step(solver_state):
        # Advance the ODE solver one step
        return solver.step(_func, tf, solver_state)

    def cond_fun(solver_state):
        t, dt = solver_state.t, solver_state.dt
        return (t < tf) & (dt > 0)

    return backend.while_loop(cond_fun, _ode_step, solver_state)


# The "forward pass" through the ODE solve, but don't save any time-series data
def _odeint_fwd(solver, ode_rhs, solver_state, tf, context):
    primals = _odeint(solver, ode_rhs, solver_state, tf, context)
    residuals = (primals, solver_state.t, tf, context)
    return primals, residuals


# The "reverse pass" through the ODE solve, using an augmented dynamical
# system with the adjoint variables.
def _odeint_adj(solver, ode_rhs, residuals, adjoints):
    primals, t0, tf, context = residuals

    # The args may contain bools, ints, or otherwise non-differentiable data.
    # Here we can split the args into dynamic and static components, and only
    # pass the dynamic args through the adjoint system.
    dynamic_args, static_args = eqx.partition(context, eqx.is_inexact_array_like)

    yf = primals.y
    yf_bar = adjoints.y

    # Initial conditions for adjoint DAE system are given in 3.3 of
    # "Adjoint sensitivity analysis for DAEs" by Cao et al. (2003).
    # We only have to do this for nontrivial mass matrices.  The paper
    # includes some more general cases, so we could probably expand on
    # this in the future, but this will work for the kinds of semi-explicit
    # systems generated by our acausal modeling framework.
    if solver.supports_mass_matrix and solver.mass is not None:
        M = solver.mass

        # Check that the mass matrix corresponds to a semi-explicit index-1
        # system.  If not, we haven't implemented the adjoint initialization yet
        n_alg = sum(np.all(M == 0, axis=1))
        n_ode = M.shape[0] - n_alg
        # Check that the mass matrix is in block-identity form
        if (
            not np.all(M[:n_ode, :n_ode] == np.eye(n_ode))
            or not np.all(M[n_ode:, :n_ode] == 0)
            or not np.all(M[:n_ode, n_ode:] == 0)
            or not np.all(M[n_ode:, n_ode:] == 0)
        ):
            raise NotImplementedError(
                "Adjoint initialization only implemented for semi-explicit index-1 systems."
            )

        J = jax.jacfwd(ode_rhs)(yf, tf, context)
        dg_ode = J[
            n_ode:, :n_ode
        ]  # Jacobian of constraints with respect to differentials
        dg_alg = J[
            n_ode:, n_ode:
        ]  # Jacobian of constraints with respect to algebraic states

        yf_bar_ode, yf_bar_alg = yf_bar[:n_ode], yf_bar[n_ode:]
        yf_bar_ode = yf_bar_ode - dg_ode.T @ jnp.linalg.solve(dg_alg.T, yf_bar_alg)
        yf_bar = jnp.concatenate([yf_bar_ode, yf_bar_alg])

    init_adj_state = (
        yf,
        yf_bar,
        0.0,
        jax.tree_util.tree_map(jnp.zeros_like, dynamic_args),
    )
    solver_state, adj_dynamics = solver.initialize_adjoint(
        ode_rhs, init_adj_state, tf, context
    )

    solver_state = _odeint(solver, adj_dynamics, solver_state, -t0, context)
    _, y0_bar, t0_bar, ctx_bar = solver_state.unravel(solver_state.y)

    # The Jacobian with respect to the final time is just the time derivative of
    # the state at the final time.
    tf_bar = jnp.dot(ode_rhs(yf, tf, context), yf_bar)

    # Recombine the dynamic and static args
    ctx_bar = eqx.combine(ctx_bar, static_args)
    solver_state_bar = adjoints.with_state_and_time(y0_bar, t0_bar)

    return (solver_state_bar, tf_bar, ctx_bar)


_odeint.defvjp(_odeint_fwd, _odeint_adj)
