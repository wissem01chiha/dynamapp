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

"""
Implementation of nonlinear MPC with a direct shooting transcription and IPOPT as the
NLP solver.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .base import NonlinearMPCIpopt, NMPCProblemStructure

from ..utils import make_ode_rhs
from ..utils import rk4_major_step_constant_u


class DirectShootingNMPC(NonlinearMPCIpopt):
    """
    Implementation of nonlinear MPC with a direct shooting transcription and IPOPT as
    the NLP solver.

    Input ports:
        (0) x_0 : current state vector.
        (1) x_ref : reference state trajectory for the nonlinear MPC.
        (2) u_ref : reference input trajectory for the nonlinear MPC.

    Output ports:
        (1) u_opt : the optimal control input to be applied at the current time step
                    as determined by the nonlinear MPC.

    Parameters:
        plant: LeafSystem or Diagram
            The plant to be controlled.

        Q: Array
            State weighting matrix in the cost function.

        QN: Array
            Terminal state weighting matrix in the cost function.

        R: Array
            Control input weighting matrix in the cost function.

        N: int
            The prediction horizon, an integer specifying the number of steps to
            predict. Note: prediction and control horizons are identical for now.

        nh: int
            Number of minor steps to take within an RK4 major step.

        dt: float:
            Major time step, a scalar indicating the increment in time for each step in
            the prediction and control horizons.

        lb_u: Array
            Lower bound on the control input vector.

        ub_u: Array
            Upper bound on the control input vector.

        u_optvars_0: Array
            Initial guess for the control vector optimization variables in the NLP.
    """

    def __init__(
        self,
        plant,
        Q,
        QN,
        R,
        N,
        nh,
        dt,
        lb_u=None,
        ub_u=None,
        u_optvars_0=None,
        name=None,
    ):
        self.plant = plant

        self.Q = Q
        self.QN = QN
        self.R = R

        self.N = N
        self.nh = nh
        self.dt = dt

        self.lb_u = lb_u
        self.ub_u = ub_u

        self.nx = Q.shape[0]
        self.nu = R.shape[0]

        if lb_u is None:
            self.lb_u = -1e20 * jnp.ones(self.nu)

        if ub_u is None:
            self.ub_u = 1e20 * jnp.ones(self.nu)

        # Currently guesses are not taken into account
        self.u_optvars_0 = u_optvars_0  # Currently does nothing
        if u_optvars_0 is None:
            u_optvars_0 = jnp.zeros((N, self.nu))

        self.ode_rhs = make_ode_rhs(plant, self.nu)

        nlp_structure_ipopt = NMPCProblemStructure(
            self.num_optvars,
            self._objective,
        )

        super().__init__(
            dt,
            self.nu,
            self.num_optvars,
            nlp_structure_ipopt,
            name=name,
        )

    @property
    def num_optvars(self):
        return self.N * self.nu

    @property
    def num_constraints(self):
        return 0

    @property
    def bounds_optvars(self):
        lb = jnp.tile(self.lb_u, self.N)
        ub = jnp.tile(self.ub_u, self.N)
        return (lb, ub)

    @property
    def bounds_constraints(self):
        c_lb = []
        c_ub = []
        return (c_lb, c_ub)

    @partial(jax.jit, static_argnames=("self",))
    def _objective(self, optvars, t0, x0, x_ref, u_ref):
        u_flat = optvars
        u = jnp.array(u_flat.reshape((self.N, self.nu)))

        x = jnp.zeros((self.N + 1, x0.size))
        x = x.at[0].set(x0)

        def _update_function(idx, x):
            t_major_start = t0 + self.dt * idx
            x_current = x[idx]
            u_current = u[idx]
            x_next = rk4_major_step_constant_u(
                t_major_start,
                x_current,
                u_current,
                self.dt,
                self.nh,
                self.ode_rhs,
            )
            return x.at[idx + 1].set(x_next)

        x = jax.lax.fori_loop(0, self.N, _update_function, x)

        xdiff = x - x_ref
        udiff = u - u_ref

        # compute sum of quadratic products for x_0 to x_{N-1}
        A = jnp.dot(xdiff[:-1], self.Q)
        qp_x_sum = jnp.sum(xdiff[:-1] * A, axis=None)

        # Compute quadratic product for the x_N
        xN = xdiff[-1]
        qp_x_N = jnp.dot(xN, jnp.dot(self.QN, xN))

        # compute sum of quadratic products for u_0 to u_{N-1}
        B = jnp.dot(udiff, self.R)
        qp_u_sum = jnp.sum(udiff * B, axis=None)

        # Sum the quadratic products
        total_sum = qp_x_sum + qp_x_N + qp_u_sum
        return total_sum
