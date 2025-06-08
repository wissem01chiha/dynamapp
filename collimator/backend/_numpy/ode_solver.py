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
from typing import TYPE_CHECKING, Callable

import jax
import numpy as np
from jax import tree_util
from jax.flatten_util import ravel_pytree

from ..ode_solver import ODESolverBase, ODESolverOptions, ODESolverState
from ...lazy_loader import LazyLoader

if TYPE_CHECKING:
    import scipy
    from scipy.integrate import DenseOutput
    from ..typing import Array
    from ...framework import ContextBase, SystemBase
    from ...framework.state import StateComponent
else:
    scipy = LazyLoader("scipy", globals(), "scipy")

__all__ = ["ODESolver"]


@dataclasses.dataclass
class ScipySolverState(ODESolverState):
    interp_coeff: Array = None  # Current dense interpolation coefficients
    unravel: Callable = None  # Unravel the flattened vector to the original pytree
    interpolant: DenseOutput = None

    def __post_init__(self):
        if self.t_prev is None:
            self.t_prev = self.t

    # Inherits docstring from `ODESolverState`
    def eval_interpolant(self, t_eval: float) -> Array:
        return self.unravel(self.interpolant(t_eval))

    # Inherits docstring from `ODESolverState`
    @property
    def unraveled_state(self) -> StateComponent:
        return self.unravel(self.y)

    def ravel(self, x: StateComponent) -> Array:
        return np.concatenate([np.ravel(e) for e in jax.tree_leaves(x)])


@dataclasses.dataclass
class ScipySolver(ODESolverBase):
    # FIXME (WC-290): Test that solvers besides RK45 work
    # At minimum check BDF and DOP853
    supported_methods = {
        "auto": "RK45",
        "default": "RK45",  # legacy support
        "non-stiff": "RK45",
        "stiff": "BDF",
        "RK45": "RK45",
        "RK23": "RK23",
        "DOP853": "DOP853",
        "Radau": "Radau",
        "BDF": "BDF",
        "LSODA": "LSODA",
    }

    @staticmethod
    def make_ravel(pytree):
        x, unravel = ravel_pytree(pytree)

        def ravel(x):
            return np.hstack(tree_util.tree_leaves(x)).reshape(-1)

        return x, ravel, unravel

    def _finalize(self):
        """Create a wrapper for scipy.integrate.solve_ivp for ODE solving

        This can be used in cases where diffrax cannot - specifically when the system
        time derivatives are not traceable by JAX. However, it is expected to typically
        be less efficient than the JIT-compiled versions from JAX or diffrax.
        """
        try:
            method = self.supported_methods[self.method]
        except KeyError:
            raise ValueError(
                f"Invalid method '{self.method}' for SciPy ODE solver. Must be one of "
                f"{list(self.supported_methods.keys())}"
            )

        self.options = {
            "method": method,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step": self.max_step_size or np.inf,
            "dense_output": True,
        }

        if self.method == "LSODA":
            self.options["min_step"] = self.min_step_size or 0.0

    # Inherits docstring from ODESolverBase
    def initialize(self, context: ContextBase, dt: float = None) -> ScipySolverState:
        xc0, ravel, unravel = self.make_ravel(context.continuous_state)
        t0 = context.time

        self._unravel = unravel
        self._ravel = ravel

        self._solver = scipy.integrate.RK45(
            lambda t, y: self.flat_ode_rhs(y, t, context),
            t0,
            xc0,
            t_bound=np.inf,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step_size or np.inf,
            first_step=dt,
        )

        return ScipySolverState(
            y=xc0,
            t=t0,
            f=self._solver.f,
            dt=self._solver.h_abs,
            unravel=unravel,
        )

    # Inherits docstring from ODESolverBase
    def flat_ode_rhs(self, y, t, context):
        if y is None or len(y) == 0:
            return None
        xc = self._unravel(y)
        xcdot = self.ode_rhs(xc, t, context)
        return self._ravel(xcdot)

    # Inherits docstring from ODESolverBase
    def step(
        self,
        func: Callable,
        boundary_time: float,
        solver_state: ScipySolverState,
    ) -> ScipySolverState:
        # This method assumes that the "source of truth" is `solver_state`
        # and not the solver object itself. So first we have to override
        # the solver attributes with the values from the state object. Then
        # after the step we store the results back in the state. This allows
        # the main loop to modify the solver state and treat this call as
        # "functionally pure", even though it does modify the solver.
        self._solver.fun = lambda t, y: func(y, t)
        self._solver.t_bound = boundary_time
        self._solver.t = solver_state.t
        self._solver.t_old = solver_state.t_prev
        self._solver.y = solver_state.y
        self._solver.f = solver_state.f
        self._solver.h_abs = solver_state.dt

        # The solver updates `y`, `t`, `t_old` etc. in place
        self._solver.step()

        # The solver will automatically set its status to "finished"
        # when the boundary time is reached.  However, we're controlling
        # time ourselves from the main loop, so we need to force the
        # status to "running" so that it doesn't throw an error.
        self._solver.status = "running"

        return ScipySolverState(
            y=self._solver.y,
            t=self._solver.t,
            t_prev=self._solver.t_old,
            f=self._solver.f,
            dt=self._solver.h_abs,
            unravel=solver_state.unravel,
            interpolant=self._solver.dense_output(),
        )


def ODESolver(
    system: SystemBase,
    options: ODESolverOptions = None,
) -> ODESolverBase:
    """Create an ODE solver used to advance continuous time in hybrid simulation.

    Args:
        system (SystemBase):
            The system to be simulated.
        options (ODESolverOptions, optional):
            Options for the ODE solver.  Defaults to None.
        enable_tracing (bool, optional):
            Whether to enable tracing of time derivatives for JAX solvers.  Defaults
            to True.

    Returns:
        ODESolverBase:
            An instance of a class implementing the `ODESolverBase` interface.
            The specific class will depend on the system and options.
    """

    if options is None:
        options = ODESolverOptions()
    options = dataclasses.asdict(options)

    return ScipySolver(system, **options)
