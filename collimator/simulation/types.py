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

import dataclasses
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

from dataclasses_json import dataclass_json

from ..backend import ODESolverOptions, ODESolverState, numpy_api

if TYPE_CHECKING:
    from ..framework import ContextBase, EventCollection, SystemCallback
    from ..backend.typing import Array, Scalar
    from ..backend.results_data import AbstractResultsData


__all__ = [
    "StepEndReason",
    "GuardIsolationData",
    "ContinuousIntervalData",
    "SimulatorOptions",
    "SimulatorState",
    "SimulationResults",
    "ResultsOptions",
    "ResultsMode",
]


# Internal data structure to determine why a major step ended.
class StepEndReason(IntEnum):
    NothingTriggered = 0
    TimeTriggered = 1
    GuardTriggered = 2
    BothTriggered = 3
    TerminalEventTriggered = 4


# Internal data structure for the bisection search for zero-crossing events.
class GuardIsolationData(NamedTuple):
    zc_before_time: int
    zc_after_time: int
    guards: EventCollection
    context: ContextBase


# Internal data structure for the results of advancing continuous time
class ContinuousIntervalData(NamedTuple):
    context: ContextBase
    triggered: bool  # Any zero-crossing events trigger?
    terminate_early: bool  # Terminal event triggered?
    t0: int  # Beginning of the interval (integer time stamp)
    tf: int  # End of the interval (integer time stamp)
    results_data: AbstractResultsData

    # The current state of the ODE solver
    ode_solver_state: ODESolverState = None


# Internal data structure to carry through the main simulation loop
class SimulatorState(NamedTuple):
    context: ContextBase
    timed_events: EventCollection
    step_end_reason: StepEndReason

    # Integer representation of simulation time - used for synchronizing
    # events without floating point drift.
    int_time: int

    results_data: AbstractResultsData

    # The current state of the ODE solver
    ode_solver_state: ODESolverState = None


# Container for options related to the Simulator class.
@dataclasses.dataclass
class SimulatorOptions:
    """Options for the hybrid simulator.

    See documentation for `simulate` for details on these options.
    This also contains all configuration for the ODE solver as a subset of options
    so that multiple options classes don't need to be created separately.
    """

    math_backend: str = dataclasses.field(
        default_factory=lambda: numpy_api.active_backend
    )
    enable_tracing: bool = True
    enable_autodiff: bool = False

    # If autodiff is enabled, max_major_steps must be set in order to bound the number
    # of iterations in the while loop.  When running a simulation using the `simulate`
    # function, this can typically be determined automatically based on the number of
    # periodic events in the system.  However, it should be specified manually in the
    # following cases:
    #   - When running a simulation by creating a `Simulator` object and calling the
    #     `advance_to` method directly. In this case the `Simulator` object does not
    #     attempt to automatically determine a bound on the number of major steps.
    #   - When autodiff is used to compute the sensitivity with respect to simulation
    #     end time, for example when computing periodic limit cycles. In this case the
    #     time variables passed to `estimate_max_major_steps` are JAX tracers and cannot
    #     be used to determine a fixed (static) bound on the number of major steps.
    #   - When the system has frequent zero-crossing events.  In this case the "safety
    #     factor" in the heuristic for estimating the number of major steps may be too
    #     small, underestimating the bound on the number of major steps.
    # In any case, `estimate_max_major_steps` can still be called statically ahead
    # of time to determine a reasonable value for `max_major_steps`, using for instance
    # a conservative bound on end time and safety factor.
    max_major_steps: int = None
    max_major_step_length: float = None

    # Length of the buffer for storing time series data.  When the buffer is full the
    # data will be dumped to NumPy arrays.  For best performance, set to a value that
    # can hold the entire simulation time series.  However, in most cases this should
    # not need to be modified by the user.
    buffer_length: int = 1000

    # ODE solver options
    ode_solver_method: str = "auto"  # Dopri5 (jax/scipy) or BDF (jax)
    rtol: float = 1e-6  # Relative tolerance for adaptive solvers
    atol: float = 1e-8  # Absolute tolerance for adaptive solvers
    min_minor_step_size: float = None
    max_minor_step_size: float = None

    # This option determines whether the simulator saves any data.  If the
    # simulation is initiated from `simulate` this will be set automatically
    # depending on whether `recorded_signals` is provided.  Hence, this
    # should not need to be manually configured.
    # FIXME: remove this and use `recorded_signals` instead. There are usecases
    # where simulate() is not used and we use the Simulator's advance_to function
    # directly. In those cases, recorded_signals can be set while save_time_series
    # is False which is confusing.
    save_time_series: bool = False

    # Dictionary of ports (or other cache sources) for which the time series should
    # be recorded. Note that if the simulation is initiated from `simulate` and
    # `recorded_signals` is provided as a kwarg to `simulate`, anything set here
    # will be overridden.  Hence, this should not need to be manually configured.
    recorded_signals: dict[str, SystemCallback] = None

    # If the context is not needed for anything, opting to not return it can
    # speed up compilation times.  For instance, typical simulation calls from
    # the UI don't use the context for anything, so model_interface.py will
    # set `return_context=False` for performance.
    return_context: bool = True

    # Zero crossings are localized in time using the ODE solver interpolant,
    # which provides state values for any time value in the previous integration
    # time interval.
    # Bisection is used to search the time interval. Rather than run bisection
    # in a while loop until the time interval is _small_, bisection is run for
    # fixed number of iterations, as this results in localizing zero crossings in
    # time within some small fraction of the integrated time interval.
    # e.g. if the major step length is 1.0 second, and bisection is run for 40
    # loops, the zero crossing time tolerance is approx. 1e-12, a.k.a. picosecond.
    zc_bisection_loop_count: int = 40

    # Scale of integer time used for event synchronization.  The default value is
    # 1e-12, corresponding to picosecond resolution.  The maximum representable
    # time in this case is around 0.3 years.  If longer simulations are needed,
    # the time scale can be increased to 1e-9, 1e-6, etc.
    int_time_scale: float = None

    # Called at the end of each major step with the current time as an argument.
    major_step_callback: Callable[[Scalar]] = None

    @property
    def ode_options(self) -> ODESolverOptions:
        return ODESolverOptions(
            rtol=self.rtol,
            atol=self.atol,
            min_step_size=self.min_minor_step_size,
            max_step_size=self.max_minor_step_size,
            method=self.ode_solver_method,
            enable_autodiff=self.enable_autodiff,
        )

    def __repr__(self) -> str:
        return (
            f"SimulatorOptions("
            f"math_backend={self.math_backend}, "
            f"enable_tracing={self.enable_tracing}, "
            f"max_major_step_length={self.max_major_step_length}, "
            f"max_major_steps={self.max_major_steps}, "
            f"ode_solver_method={self.ode_solver_method}, "
            f"rtol={self.rtol}, "
            f"atol={self.atol}, "
            f"min_minor_step_size={self.min_minor_step_size}, "
            f"max_minor_step_size={self.max_minor_step_size}, "
            f"zc_bisection_loop_count={self.zc_bisection_loop_count}, "
            f"save_time_series={self.save_time_series}, "
            f"recorded_signals={len(self.recorded_signals or [])}, "  # changed
            f"return_context={self.return_context}"
            f")"
        )


class ResultsMode(Enum):
    auto = 0
    discrete_steps_only = 1
    fixed_interval = 2


@dataclass_json
@dataclasses.dataclass
class ResultsOptions:
    mode: Optional[ResultsMode] = ResultsMode.auto
    max_results_interval: Optional[float] = None
    fixed_results_interval: Optional[float] = None
    # TODO: maybe include recorded_signals here?


class SimulationResults(NamedTuple):
    """Data structure for the results of a simulation.

    Attributes:
        context (ContextBase):
            The output context of the simulation, containing final states, times, etc.
            May be None if `return_context=False` was passed to `simulate`.
        outputs (dict[str, Array]):
            A dictionary of the outputs of the simulation, keyed by the name provided
            to `recorded_signals` in `simulate`.  May be None if `recorded_signals` is
            not provided to `simulate`.
        time (Array):
            The time vector of the simulation.
        parameters (dict[str, Any]):
            The parameters used in the simulation, used in ensemble simulations
            to identify different runs.
    """

    context: ContextBase
    time: Array = None
    outputs: dict[str, Array] = None
    parameters: dict[str, Any] = None
