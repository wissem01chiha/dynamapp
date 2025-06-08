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
Implementation of nonlinear MPC with Hermite-Simpson collocation and IPOPT as the NLP
solver.
"""

from functools import partial

import jax
import jax.numpy as jnp

from ...backend import cond

from .base import NonlinearMPCIpopt, NMPCProblemStructure

from ..utils import make_ode_rhs


class HermiteSimpsonNMPC(NonlinearMPCIpopt):
    """
    Implementation of nonlinear MPC with Hermite-Simpson collocation and IPOPT as the
    NLP solver.

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

        dt: float:
            Major time step, a scalar indicating the increment in time for each step in
            the prediction and control horizons.

        lb_x: Array
            Lower bound on the state vector.

        ub_x: Array
            Upper bound on the state vector.

        lb_u: Array
            Lower bound on the control input vector.

        ub_u: Array
            Upper bound on the control input vector.

        include_terminal_x_as_constraint: bool
            If True, the terminal state is included as a constraint in the NLP.

        include_terminal_u_as_constraint: bool
            If True, the terminal control input is included as a constraint in the NLP.

        x_optvars_0: Array
            Initial guess for the state vector optimization variables in the NLP.

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
        dt,
        lb_x=None,
        ub_x=None,
        lb_u=None,
        ub_u=None,
        include_terminal_x_as_constraint=False,
        include_terminal_u_as_constraint=False,
        x_optvars_0=None,
        u_optvars_0=None,
        name=None,
    ):
        self.Q = Q
        self.QN = QN
        self.R = R

        self.N = N
        self.dt = dt

        self.lb_x = lb_x
        self.ub_x = ub_x
        self.lb_u = lb_u
        self.ub_u = ub_u

        self.include_terminal_x_as_constraint = include_terminal_x_as_constraint
        self.include_terminal_u_as_constraint = include_terminal_u_as_constraint

        self.nx = Q.shape[0]
        self.nu = R.shape[0]

        if lb_x is None:
            self.lb_x = -1e20 * jnp.ones(self.nx)

        if ub_x is None:
            self.ub_x = 1e20 * jnp.ones(self.nx)

        if lb_u is None:
            self.lb_u = -1e20 * jnp.ones(self.nu)

        if ub_u is None:
            self.ub_u = 1e20 * jnp.ones(self.nu)

        # Currently guesses are not taken into account
        self.x_optvars_0 = x_optvars_0  # Currently does nothing
        self.u_optvars_0 = u_optvars_0  # Currently does nothing
        if x_optvars_0 is None:
            x_optvars_0 = jnp.zeros((N + 1, self.nx))
        if u_optvars_0 is None:
            u_optvars_0 = jnp.zeros((N + 1, self.nu))

        self.ode_rhs = make_ode_rhs(plant, self.nu)

        nlp_structure_ipopt = NMPCProblemStructure(
            self.num_optvars,
            self._objective,
            self._constraints,
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
        return (self.N + 1) * (self.nx + self.nu)

    @property
    def num_constraints(self):
        # max size regardless of terminal constraints (for jit compilation)
        num_contraints = (self.N + 2) * self.nx + self.nu
        return num_contraints

    @property
    def bounds_optvars(self):
        lb = jnp.hstack(
            [jnp.tile(self.lb_u, self.N + 1), jnp.tile(self.lb_x, self.N + 1)]
        )
        ub = jnp.hstack(
            [jnp.tile(self.ub_u, self.N + 1), jnp.tile(self.ub_x, self.N + 1)]
        )
        return (lb, ub)

    @property
    def bounds_constraints(self):
        c_lb = jnp.zeros(self.num_constraints)
        c_ub = jnp.zeros(self.num_constraints)
        return (c_lb, c_ub)

    @partial(jax.jit, static_argnames=("self",))
    def _objective(self, optvars, t0, x0, x_ref, u_ref):
        u_and_x_flat = optvars

        u = u_and_x_flat[: self.nu * (self.N + 1)].reshape((self.N + 1, self.nu))
        x = u_and_x_flat[self.nu * (self.N + 1) :].reshape((self.N + 1, self.nx))

        xdiff = x - x_ref
        udiff = u - u_ref

        # compute sum of quadratic products for x_0 to x_{n-1}
        A = jnp.dot(xdiff[:-1], self.Q)
        qp_x_sum = jnp.sum(xdiff[:-1] * A, axis=None)

        # Compute quadratic product for the x_N
        xN = xdiff[-1]
        qp_x_N = jnp.dot(xN, jnp.dot(self.QN, xN))

        # compute sum of quadratic products for u_0 to u_{n-1}
        B = jnp.dot(udiff, self.R)
        qp_u_sum = jnp.sum(udiff * B, axis=None)

        # Sum the quadratic products
        total_sum = qp_x_sum + qp_x_N + qp_u_sum
        return total_sum

    @partial(jax.jit, static_argnames=("self",))
    def _constraints(self, optvars, t0, x0, x_ref, u_ref):
        u_and_x_flat = optvars

        u = u_and_x_flat[: self.nu * (self.N + 1)].reshape((self.N + 1, self.nu))
        x = u_and_x_flat[self.nu * (self.N + 1) :].reshape((self.N + 1, self.nx))

        h = self.dt
        t = t0 + h * jnp.arange(self.N + 1)

        dot_x = jnp.zeros((self.N + 1, self.nx))

        def loop_body_break(idx, dot_x):
            rhs = self.ode_rhs(x[idx], u[idx], t[idx])
            dot_x = dot_x.at[idx].set(rhs)
            return dot_x

        dot_x = jax.lax.fori_loop(0, self.N + 1, loop_body_break, dot_x)

        t = t0 + self.dt * jnp.arange(self.N + 1)
        t_c = 0.5 * (t[:-1] + t[1:])
        u_c = 0.5 * (u[:-1] + u[1:])
        x_c = 0.5 * (x[:-1] + x[1:]) + (h / 8.0) * (dot_x[:-1] - dot_x[1:])

        dot_x_c = (-3.0 / 2.0 / h) * (x[:-1] - x[1:]) - (1.0 / 4.0) * (
            dot_x[:-1] + dot_x[1:]
        )

        c0 = x0 - x[0]

        c_others = jnp.zeros((self.N, self.nx))

        def loop_body_colloc(idx, c_others):
            c_colocation = self.ode_rhs(x_c[idx], u_c[idx], t_c[idx]) - dot_x_c[idx]
            c_others = c_others.at[idx].set(c_colocation)
            return c_others

        c_others = jax.lax.fori_loop(0, self.N, loop_body_colloc, c_others)
        c_all = jnp.hstack([c0.ravel(), c_others.ravel()])

        c_terminal_x = x_ref[self.N] - x[self.N]
        c_terminal_u = u_ref[self.N] - u[self.N]

        c_all = cond(
            self.include_terminal_x_as_constraint,
            lambda c_all, c_terminal_x: jnp.hstack([c_all, c_terminal_x.ravel()]),
            lambda c_all, c_terminal_x: jnp.hstack([c_all, jnp.zeros(self.nx)]),
            c_all,
            c_terminal_x,
        )

        c_all = cond(
            self.include_terminal_u_as_constraint,
            lambda c_all, c_terminal_u: jnp.hstack([c_all, c_terminal_u.ravel()]),
            lambda c_all, c_terminal_x: jnp.hstack([c_all, jnp.zeros(self.nu)]),
            c_all,
            c_terminal_u,
        )

        return c_all
