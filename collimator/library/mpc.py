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

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from ..framework import LeafSystem
from ..backend import cond
from ..lazy_loader import LazyLoader

if TYPE_CHECKING:
    from jax.scipy import linalg
else:
    linalg = LazyLoader("linalg", globals(), "jax.scipy.linalg")


_load_error_msg = (
    "OSQP is not installed. You can install it with:\npip install cmake."
    "pip install pycollimator[nmpc]"
)

osqp = LazyLoader(
    "osqp",
    globals(),
    "osqp",
    error_message=_load_error_msg,
)

jaxopt = LazyLoader(
    "jaxopt",
    globals(),
    "jaxopt",
)

__all__ = [
    "LinearDiscreteTimeMPC",
    "LinearDiscreteTimeMPC_OSQP",
]


class LinearDiscreteTimeMPC(LeafSystem):
    """Model predictive control for a linear discrete-time system.

    Notes:
        This block is _feedthrough_, meaning that every time the output port is
        evaluated, the solver is run.  This is in order to avoid a "data flow" delay
        between the solver and the output port (see also the `PIDDiscrete` block).
        This means that either the input or output signal should be discrete-time in
        order for the block to work as intended.  Ideally, the output signal should
        be passed to a zero-order hold block so that the solver only needs to be run
        once per step.
    """

    def __init__(
        self,
        lin_sys,
        Q,
        R,
        N,
        dt,
        x_ref,
        lbu=-np.inf,
        ubu=np.inf,
        name=None,
        warm_start=False,
    ):
        super().__init__(name=name)
        self.n = lin_sys.A.shape[0]
        self.m = lin_sys.B.shape[1]
        self.N = N
        self.warm_start = warm_start

        # Convert to discrete time with Euler discretization
        A = jnp.eye(self.n) + dt * lin_sys.A
        B = dt * lin_sys.B

        # Input: current state (x0)
        self.declare_input_port()

        self.solve, init_params = self._make_solver(A, B, Q, R, lbu, ubu, N, x_ref)

        # Declare a feedthrough output port for the solver
        self.declare_output_port(
            self.solve,
            requires_inputs=True,
            period=dt,
            offset=0.0,
        )

    def _make_solver(self, A, B, Q, R, lbu, ubu, N, xf):
        from jax.experimental import sparse

        n = self.n
        m = self.m

        # Identity matrices of state and control dimension
        I_A = jnp.eye(n)
        I_B = jnp.eye(m)

        def e(k):
            """Unit vector in the kth direction"""
            return jnp.zeros(N).at[k].set(1.0)

        blocks = [Q, R] * N
        P = linalg.block_diag(*blocks)

        # The initial condition constraint is x[0] = x0
        L0 = jnp.eye(n, N * (n + m))

        # The defect constraint for step k is
        #    0 = (A * x[k] + B * u[k]) - x[k+1]
        L_defect = jnp.vstack(
            [
                jnp.kron(e(k), jnp.hstack([A, B]))
                + jnp.kron(e(k + 1), jnp.hstack([-I_A, 0 * B]))
                for k in range(N - 1)
            ]
        )

        # Constraint on terminal state
        Lf = jnp.kron(e(N - 1), jnp.hstack([I_A, 0 * B]))

        # Constraints on the control input
        L_input = jnp.vstack(
            [jnp.kron(e(k), jnp.hstack([0 * B.T, I_B])) for k in range(N)]
        )

        # Stack the constraint matrices and define bounds
        #  lb <= Lx <= ub
        L = jnp.vstack([L0, L_defect, Lf, L_input])

        def _get_bounds(x0):
            lb = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, lbu)]
            )
            ub = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, ubu)]
            )
            return lb, ub

        # self.qp = jaxopt.BoxOSQP(matvec_Q=_matvec_Q, matvec_A=_matvec_A)
        c = jnp.zeros(N * (n + m))

        # qp = jaxopt.BoxOSQP()

        P_sp = sparse.BCOO.fromdense(P)
        L_sp = sparse.BCOO.fromdense(L)

        # @sparse.sparsify
        @jax.jit
        def _matvec_Q(params_Q, x):
            """Matrix-vector product Q * x"""
            return P_sp @ x

        @jax.jit
        def _matvec_A(params_A, x):
            """Matrix-vector product A * x"""
            return L_sp @ x

        self.qp = jaxopt.BoxOSQP(matvec_Q=_matvec_Q, matvec_A=_matvec_A)

        lb, ub = _get_bounds(xf)
        z0 = jnp.zeros(N * (n + m))
        init_params = self.qp.init_params(
            z0, params_obj=(None, c), params_eq=None, params_ineq=(lb, ub)
        )

        def _solve(time, state, x0):
            lb, ub = _get_bounds(x0)

            if self.warm_start:
                raise NotImplementedError(
                    "Warm start not yet supported for JAX MPC block"
                )
            else:
                init_params = None
            # sol = qp.run(params_obj=(P, c), params_eq=L, params_ineq=(lb, ub)).params
            # sol = self.qp.run(params_obj=(None, c), params_ineq=(lb, ub)).params
            osqp_params = self.qp.run(
                init_params=init_params,
                params_obj=(None, c),
                params_ineq=(lb, ub),
            ).params

            xu_traj = osqp_params.primal[0].reshape(
                (self.n + self.m, self.N), order="F"
            )

            # Time series of control inputs
            u_opt = xu_traj[self.n :, :]

            # Return the first control value only
            return u_opt[:, 0]

        return jax.jit(_solve), init_params


class LinearDiscreteTimeMPC_OSQP(LeafSystem):
    """
    Same as above, but using OSQP.  This is an example of a case where a traced array gets passed
    to a function that doesn't know how to handle it.
    """

    def __init__(
        self,
        lin_sys,
        Q,
        R,
        N,
        dt,
        x_ref,
        lbu=-np.inf,
        ubu=np.inf,
        name=None,
    ):
        super().__init__(name=name)
        self.n = lin_sys.A.shape[0]
        self.m = lin_sys.B.shape[1]
        self.N = N

        # Convert to discrete time with Euler discretization
        A = jnp.eye(self.n) + dt * lin_sys.A
        B = dt * lin_sys.B

        self._make_solver(A, B, Q, R, lbu, ubu, N, x_ref)

        # Input: current state (x0)
        self.declare_input_port()

        self._result_template = jnp.zeros((self.n + self.m) * self.N)

        # Wrap the solve call as a JAX "pure callback" so that it can call
        # arbitrary non-JAX Python code (in this case IPOPT).
        self._solve = partial(jax.pure_callback, self.solve, self._result_template)

        self.declare_output_port(
            self._output,
            period=dt,
            offset=0.0,
            requires_inputs=True,
        )

    def _extract_u_opt(self, xu_flat_traj):
        xu_traj = np.reshape(xu_flat_traj, (self.n + self.m, self.N), order="F")

        # Split solution into states and controls
        u_opt = xu_traj[self.n :, :]

        # Return current best projected action
        return u_opt[:, 0]

    def _output(self, time, state, *inputs):
        """Output callback used when the block is in "feedthrough" mode."""
        args = (time, state, *inputs)
        u_flat_traj = cond(jnp.isinf(time), self._dummy_solve, self._solve, *args)
        return self._extract_u_opt(u_flat_traj)

    def _dummy_solve(self, _time, _state, *_inputs, **_params):
        """Safeguard for reconstructing the results during ODE solver minor steps.

        This can result in `inf` values passed to the ODE solver, which will raise
        errors in IPOPT.  Instead, we can just return another `inf` value of the
        right shape here.
        """
        return jnp.full(self._result_template.shape, jnp.inf)

    def solve(self, time, state, x0):
        # pylint: disable=not-callable
        lb, ub = self.get_bounds(x0)

        self.solver.update(l=np.array(lb), u=np.array(ub))

        # Solve problem
        sol = self.solver.solve()

        return sol.x

    def _make_solver(self, A, B, Q, R, lbu, ubu, N, xf):
        from scipy import sparse

        n = self.n
        m = self.m

        # Identity matrices of state and control dimension
        I_A = jnp.eye(n)
        I_B = jnp.eye(m)

        def e(k):
            """Unit vector in the kth direction"""
            return jnp.zeros(N).at[k].set(1.0)

        blocks = [Q, R] * N
        P = linalg.block_diag(*blocks)

        # The initial condition constraint is x[0] = x0
        L0 = jnp.eye(n, N * (n + m))

        # The defect constraint for step k is
        #    0 = (A * x[k] + B * u[k]) - x[k+1]
        L_defect = jnp.vstack(
            [
                jnp.kron(e(k), jnp.hstack([A, B]))
                + jnp.kron(e(k + 1), jnp.hstack([-I_A, 0 * B]))
                for k in range(N - 1)
            ]
        )

        # Constraint on terminal state
        Lf = jnp.kron(e(N - 1), jnp.hstack([I_A, 0 * B]))

        # Constraints on the control input
        L_input = jnp.vstack(
            [jnp.kron(e(k), jnp.hstack([0 * B.T, I_B])) for k in range(N)]
        )

        # Stack the constraint matrices and define bounds
        #  lb <= Lx <= ub
        L = jnp.vstack([L0, L_defect, Lf, L_input])

        def get_bounds(x0):
            lb = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, lbu)]
            )
            ub = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, ubu)]
            )
            return lb, ub

        self.get_bounds = jax.jit(get_bounds)
        self.solver = osqp.OSQP()

        lb, ub = get_bounds(jnp.zeros(n))  # Initialize solver with dummy variables
        self.solver.setup(
            P=sparse.csc_matrix(P),
            A=sparse.csc_matrix(L),
            l=np.array(lb),
            u=np.array(ub),
            verbose=False,
        )
