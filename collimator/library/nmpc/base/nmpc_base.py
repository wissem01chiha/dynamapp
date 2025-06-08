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

"""Base class for nonlinear MPCs"""

from abc import ABC, abstractmethod

import jax.numpy as jnp

from ....framework import LeafSystem

from ....backend import cond
from ....backend.typing import Array


class NonlinearMPCBase(LeafSystem, ABC):
    """
    Base class for nonlinear MPCs. It specifies the structure of a nonlinear MPC block
    with:

    Input ports:
        (0) x_0 : current state vector.
        (1) x_ref : reference state trajectory for the nonlinear MPC.
        (2) u_ref : reference input trajectory for the nonlinear MPC.

    Output ports:
        (1) u_opt : the optimal control input to be applied at the current time step
                    as determined by the nonlinear MPC.

    Parameters:
        dt: float
            Time step within the prediction and control horizons.
        nu: int
            Size of the control input vector.
        nopt: int
            Number of optimization variables in the nonlinear programming problem.

    Notes:
        The concrete implementation of the `_solve` method provided by derived classes
        should return an array whose first `nu` elements are `u_opt`.
    """

    def __init__(
        self,
        dt,
        nu,
        nopt,
        name=None,
    ):
        super().__init__(name=name)

        self.dt = dt
        self.nu = nu
        self.nopt = nopt

        # Input: current state (x_0)
        self.declare_input_port()

        # Input: x_ref (for Nx steps, x_0, x_1, ... x_Nx)
        self.declare_input_port()

        # Input: u_ref  (for Nu steps, u_0, u_1, ... u_Nu)
        self.declare_input_port()

        self._result_template = jnp.zeros(nopt)

        self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
            default_value=jnp.zeros(self.nu),
            requires_inputs=True,
        )

    def _output(self, time, state, *inputs):
        """Compute the optimal control trajectory from the current starting point."""
        args = (time, state, *inputs)
        optvars_sol = cond(jnp.isinf(time), self._dummy_solve, self._solve, *args)
        return optvars_sol[: self.nu]

    def _dummy_solve(self, _time, _state, *_inputs, **_params):
        """Safeguard for reconstructing the results during ODE solver minor steps.

        This can result in `inf` values passed to the ODE solver, which will raise
        errors in IPOPT.  Instead, we can just return another `inf` value of the
        right shape here.
        """
        return jnp.full(self._result_template.shape, jnp.inf)

    @abstractmethod
    def _solve(self, time, state, *inputs) -> Array:
        """
        Solve the NLP problem at the current time step.

        Parameters:
            time: float
                Current time.

            inputs: Tuple[Array, Array, Array]
                Tuple of current state, reference state, and reference input.
                    input[0] : Array
                        Current state vector `x_0`.
                    input[1] : Array
                        Reference state trajectory `x_ref`.
                    input[2] : Array
                        Reference control input trajectory `u_ref`.

        Returns: Array
            Solution of the nonlinear MPC problem, whose first `nu` elements represent
            the optimal control input `u_opt` to be applied at the current time step.
        """
        pass
