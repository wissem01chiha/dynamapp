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
import abc
import dataclasses
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .typing import Array
    from ..framework import SystemBase, ContextBase
    from ..framework.state import StateComponent


__all__ = [
    "ODESolverOptions",
    "ODESolverBase",
]


@dataclasses.dataclass
class ODESolverOptions:
    """Options for the ODE solver.

    See documentation for `simulate` for details on these options.
    """

    rtol: float = 1e-3
    atol: float = 1e-6
    min_step_size: float = None
    max_step_size: float = None
    method: str = "auto"  # Dopri5 (jax/scipy) or BDF (jax)
    enable_autodiff: bool = False


@dataclasses.dataclass
class ODESolverState(metaclass=abc.ABCMeta):
    y: Array  # The current state of the system.
    t: float
    # `f` is the time derivative of `y` at `(y,, t)`.
    f: Array  # FIXME: Not needed for BDF... so this isn't a general interface.
    # The current step size.  The next step will `dt`, clipped to the interval
    # [0, tf - t].
    dt: float
    # The previous step time.  The interpolant is valid between (t_prev, t).
    t_prev: float = None

    @abc.abstractmethod
    def eval_interpolant(self, time: float) -> Array:
        """Interpolate the state at a given time between (t_prev, t)."""
        pass

    @property
    @abc.abstractmethod
    def unraveled_state(self) -> StateComponent:
        """Unravel the continuous state from the solver state."""
        pass

    @abc.abstractmethod
    def ravel(self, x: StateComponent) -> Array:
        """Ravel the continuous state to the solver state."""
        pass

    def with_state_and_time(self, y, t) -> ODESolverState:
        return dataclasses.replace(self, y=y, t=t, t_prev=t)

    def with_context(self, context: ContextBase) -> ODESolverState:
        y = self.ravel(context.continuous_state)
        return self.with_state_and_time(y, context.time)


@dataclasses.dataclass
class ODESolverBase(metaclass=abc.ABCMeta):
    """Common interface for defining ODE solvers.

    This should typically not be used directly by users.  Instead, use the `simulate`
    interface or its lower-level `Simulator` class for finer control.  For purely
    continuous systems, this will effectively do a standard ODE solve.
    """

    system: SystemBase
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step_size: float = None
    min_step_size: float = None
    method: str = "auto"
    enable_autodiff: bool = False

    supports_mass_matrix: bool = False

    def _finalize(self):
        """Hook for any class-specific finalization after __post_init__."""
        pass

    def __post_init__(self):
        self._finalize()

        # Check that if the system contains a non-trivial mass matrix (including a
        # semi-explicit index-1 DAE specified with singular mass matrix), then the
        # solver is capable of handling it.
        if (not self.supports_mass_matrix) and self.system.has_mass_matrix:
            raise ValueError(
                f"The solver {self.__class__.__name__} does not support systems with "
                "non-trivial mass matrices.  Use a different solver (currently only "
                "the JAX-backend BDF solver is compatible with mass-matrix ODEs)."
            )

    @abc.abstractmethod
    def initialize(self, context: ContextBase, dt: float = None) -> ODESolverState:
        """Set up the solver and return the initial state.

        Args:
            context: The simulation context.
            dt: The initial step size. If not provided, it will be estimated.
        """
        pass

    @abc.abstractmethod
    def step(
        self,
        func: Callable,
        boundary_time: float,
        solver_state: ODESolverState,
    ) -> ODESolverState:
        """Advance the solver forward one step.

        This will repeat the adaptive step attempt until a step is accepted,
        returning the result.  It accepts the ODE RHS function as an explicit
        argument because (a) the function can change between steps (as when
        the discrete state changes) and (b) the backwards adjoint solve can then
        reuse this method for the custom VJP.

        Args:
            func: The function to evaluate the time derivatives.
            boundary_time: The time at which to stop the step.
            solver_state: The current state of the solver.

        Returns:
            ODESolverState: The updated state of the solver.
        """
        pass

    def ode_rhs(
        self,
        y: StateComponent,
        t: float,
        context: ContextBase,
    ) -> StateComponent:
        """Evaluate the time derivatives of the system at a given time and state."""
        context = context.with_time(t)
        # Update the continuous state, holding discrete state fixed.
        context = context.with_continuous_state(y)
        xcdot: StateComponent = self.system.eval_time_derivatives(context)
        return xcdot

    def flat_ode_rhs(
        self,
        xc: Array,
        t: float,
        context: ContextBase,
    ) -> Array:
        """Evaluate the time derivatives of the system at a given time and state.

        This is a flattened version of the `ode_rhs` method, i.e. it should take
        and return a flat vector for the state and its derivative.

        Args:
            xc (Array): The state of the system.
            t (float): The time at which to evaluate the derivatives.
            context (ContextBase): The context to use for evaluation.

        Returns:
            Array: The time derivatives of the system at the given time and state.
        """
        raise NotImplementedError(
            "Classes must implement the `flat_ode_rhs` method or assign it during "
            "initialization"
        )
