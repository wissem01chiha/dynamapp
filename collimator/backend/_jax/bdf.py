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

"""JAX-based Backwards Differentiation Formula ODE/DAE solver with adaptive stepsize.

References:
[1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical
    Solution of Ordinary Differential Equations", ACM Transactions on
    Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.
[2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
    COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
[3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I:
    Nonstiff Problems", Sec. III.2.
"""

from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, Callable
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .ode_solver_impl import ODESolverImpl, ODESolverState, norm
from ..typing import Array
from ...lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    from jax.scipy.linalg import lu_factor, lu_solve

    from ...framework.state import StateComponent
else:
    jax_scipy_linalg = LazyLoader("jax_scipy_linalg", globals(), "jax.scipy.linalg")
    lu_factor = LazyModuleAccessor(jax_scipy_linalg, "lu_factor")
    lu_solve = LazyModuleAccessor(jax_scipy_linalg, "lu_solve")

__all__ = [
    "BDFState",
    "BDFSolver",
]

EPS = np.finfo(float).eps

MAX_ORDER = 5
NEWTON_MAXITER = 4
ROOT_SOLVE_MAXITER = 15
MIN_FACTOR = 0.2
MAX_FACTOR = 10


# https://github.com/scipy/scipy/blob/v1.13.0/scipy/integrate/_ivp/bdf.py#L242
kappa = jnp.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])
GAMMA = jnp.hstack((0, jnp.cumsum(1 / jnp.arange(1, MAX_ORDER + 1))))
ALPHA = (1 - kappa) * GAMMA
ERROR_CONST = kappa * GAMMA + 1 / jnp.arange(1, MAX_ORDER + 2)


@dataclasses.dataclass
class BDFState(ODESolverState):
    # `t_return` is the time value to return in time series.  Will be inf when
    # the end time is reached.  Otherwise it should match `t`.
    t_return: float = None
    n_acc: int = 0  # Number of accepted steps
    n_rej: int = 0  # Number of rejected steps
    accepted: bool = False  # Whether the most recent attempted step was accepted

    # Unique to BDF:
    order: int = 1  # The current order of the BDF solver. Initialize to first-order.
    D: Array = None  # Table of backwards differences
    J: Array = None  # The Jacobian matrix
    M: Array = None  # The mass matrix
    LU: Array = None  # The LU factorization
    U: Array = None

    updated_jacobian: bool = False  # Whether the Jacobian has been updated
    n_equal_steps: int = 0  # Number of equal-length steps taken

    # Aux data (note this should all come at the end)
    unravel: Callable = None  # Unravel the flattened vector to the original pytree

    def __post_init__(self):
        if self.t_return is None:
            self.t_return = self.t
        if self.t_prev is None:
            self.t_prev = self.t

    # Inherits docstring from `ODESolverState`
    def eval_interpolant(self, t_eval: float) -> Array:
        if self.unravel is None:
            raise ValueError("Unravel function not set: cannot evaluate interpolant.")

        order = self.order
        t = self.t
        h = self.dt
        D = self.D

        def while_body(j, val):
            p, y = val
            p *= (t_eval - (t - h * j)) / (h * (1 + j))
            return (p, y + D[j + 1] * p)

        _, y = lax.fori_loop(0, order, while_body, (1.0, D[0]))
        return self.unravel(y)

    # Inherits docstring from `ODESolverState`
    @property
    def unraveled_state(self) -> StateComponent:
        return self.unravel(self.y)

    def ravel(self, x: StateComponent) -> Array:
        return jnp.concatenate([jnp.ravel(e) for e in jax.tree_util.tree_leaves(x)])

    @property
    def step_variables(self):
        return self.y, self.f, self.t, self.dt

    def tree_flatten(self):
        children = (
            self.y,
            self.t,
            self.f,
            self.dt,
            self.t_prev,
            self.t_return,
            self.n_acc,
            self.n_rej,
            self.accepted,
            self.order,
            self.D,
            self.J,
            self.M,
            self.LU,
            self.U,
            self.updated_jacobian,
            self.n_equal_steps,
        )
        aux_data = (self.unravel,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (unravel,) = aux_data
        return cls(*children, unravel)

    def with_state_and_time(self, y, t) -> BDFState:
        return dataclasses.replace(self, y=y, t=t, t_prev=t)


jax.tree_util.register_pytree_node(
    BDFState,
    lambda state: state.tree_flatten(),
    BDFState.tree_unflatten,
)


ROW_IDX = np.arange(MAX_ORDER + 3).reshape(-1, 1)
COL_IDX = np.arange(MAX_ORDER + 3)


def R_matrix(order, factor):
    # See Sec. 3.2 of Ref. [2] for the formula
    n = MAX_ORDER
    i = ROW_IDX[1 : n + 1]
    j = COL_IDX[1 : n + 1]
    M = jnp.zeros((n + 1, n + 1))
    M = M.at[1:, 1:].set((i - 1 - factor * j) / i)
    M = M.at[0, :].set(1)

    i = ROW_IDX[: n + 1]
    j = COL_IDX[: n + 1]
    mask = jnp.logical_and(i <= order, j <= order)
    M = jnp.where(mask, M, 1.0)
    R = jnp.where(mask, jnp.cumprod(M, axis=0), 0.0)
    return R


def _update_D(D, order, factor):
    # update D using equations in section 3.2 of Ref. [2]
    n = MAX_ORDER
    U = R_matrix(order, factor=1.0)
    R = R_matrix(order, factor)
    I_ = jnp.eye(n + 3)

    RU = jnp.where(
        (ROW_IDX <= order) & (COL_IDX <= order),
        I_.at[: n + 1, : n + 1].set(R.dot(U)),
        I_,
    )
    return jnp.dot(RU.T, D)


class BDFSolver(ODESolverImpl):
    """JAX-based Backwards Differentiation Formula ODE solver.

    The solver is variable-order and variable-stepsize, using a simplified
    Newton iteration to solve the nonlinear system of equations at each step.
    The order ranges from 1 to 5, and the step size is adjusted based on the
    error estimate from the Newton iteration.
    """

    def _finalize(self):
        self.supports_mass_matrix = True
        super()._finalize()
        self.newton_tol = max(10 * EPS / self.rtol, min(0.03, self.rtol**0.5))

    def _initialize_state(self, func, t0, xc0, mass, unravel, *args, dt=None):
        # Initialize the solver state using the first-order BDF method
        order = 1
        f0 = func(xc0, t0, *args)
        if dt is None:
            dt = self.initial_step_size(func, xc0, t0, order, f0, *args)

        # Initialize the BDF state matrix with the first-order (Euler) step
        D = jnp.zeros((MAX_ORDER + 3, len(xc0)), dtype=xc0.dtype)
        D = D.at[0].set(xc0)
        D = D.at[1].set(f0 * dt)
        c = dt / ALPHA[order]

        if mass is None:
            M = np.eye(len(xc0), dtype=xc0.dtype)
        else:
            M = mass

        jac = jax.jacfwd(func, argnums=0)
        J = jac(xc0, t0, *args)
        LU = self._lu_factor(M - c * J)

        state = BDFState(
            xc0,
            t0,
            f0,
            dt,
            unravel=unravel,
            order=order,
            D=D,
            J=J,
            M=M,
            LU=LU,
        )

        xc1 = self.predict(D, order=1)
        return dataclasses.replace(state, y=xc1)

    def predict(self, D, order) -> tuple[Array, Array]:
        # Predict new state value using the BDF formula
        k = jnp.repeat(ROW_IDX, D.shape[1], axis=1)
        return jnp.sum(jnp.where(k <= order, D, 0), axis=0)

    def solve_newton_system(self, func, t, y, c, psi, LU, M, scale):
        # Solve the BDF system of equations using a simplified Newton iteration.
        tol = self.newton_tol

        def _cond_fun(carry):
            exit_flag = carry[0]
            return ~exit_flag

        def _body_fun(carry):
            exit_flag, converged, y, d, dy_norm_old, k = carry

            f = func(y, t)

            dy = lu_solve(LU, c * f - M @ (psi + d))
            dy_norm = norm(M @ dy / scale)

            # NOTE: SciPy has a check here to exit early if the iterations appear
            # to be diverging, which saves a few iterations in some cases.  However,
            # this test does not appear to be robust with a mass matrix, so that test
            # is not implemented here.
            rate = jnp.where(jnp.isfinite(dy_norm_old), dy_norm / dy_norm_old, jnp.inf)

            y = y + dy
            d = d + dy

            converged = jnp.all(abs(f) <= EPS) | (
                (dy_norm == 0.0)
                | (jnp.isfinite(rate) & (rate / (1 - rate) * dy_norm < tol))
            )

            dy_norm_old = dy_norm
            k += 1
            exit_flag = (k >= NEWTON_MAXITER) | converged
            return exit_flag, converged, y, d, dy_norm_old, k

        d = jnp.zeros_like(y)
        dy_norm_old = jnp.inf
        exit_flag = False
        converged = False
        k = 0
        exit_flag, converged, y, d, dy_norm_old, k = jax.lax.while_loop(
            _cond_fun,
            _body_fun,
            (exit_flag, converged, y, d, dy_norm_old, k),
        )

        return converged, k, y, d

    def newton_iteration(self, state, func, boundary_time):
        y0, f, t, h = state.step_variables
        n_equal_steps = state.n_equal_steps
        order = state.order
        M, D, J, LU = state.M, state.D, state.J, state.LU

        t_new = t + h
        factor = abs(boundary_time - t) / h
        (t_new, D, n_equal_steps, recalc_lu) = jax.tree.map(
            partial(jnp.where, t_new - boundary_time > 0),
            (boundary_time, _update_D(D, order, factor), 0, True),
            (t_new, D, n_equal_steps, False),
        )
        h = t_new - t

        # Update LU: `c` has changed (maybe)
        c = h / ALPHA[order]
        LU = self._lu_factor(M - c * J, recalc_lu, LU)

        y_predict = self.predict(D, order=order)

        # Update the vector used in simplified Newton iterations
        # Since all arrays must be statically sized, extend `GAMMA` with zeros
        # to match the size of `D` (MAX_ORDER + 3)
        k = COL_IDX
        gamma = jnp.concatenate((GAMMA, np.array([0.0, 0.0])))
        gamma = jnp.where(k > 0, jnp.where(k <= order, gamma, 0), 0)
        k = jnp.repeat(ROW_IDX, D.shape[1], axis=1)
        D_submat = jnp.where(k > 0, jnp.where(k <= order, D, 0), 0)
        psi = jnp.dot(D_submat.T, gamma) / ALPHA[order]

        scale = self.atol + self.rtol * jnp.abs(y_predict)

        def _cond_fun(carry):
            exit_flag = carry[0]
            return ~exit_flag

        def _body_fun(carry):
            _exit_flag, _n_iter, _converged, current_jac, J, LU, _y_new, _d = carry

            converged, n_iter, y_new, d = self.solve_newton_system(
                func, t_new, y_predict, c, psi, LU, M, scale
            )

            # Will exit if either converged or the Jacobian is already up to date
            # This will trigger decreasing the step size and trying again.
            exit_flag = converged | current_jac

            # https://github.com/scipy/scipy/blob/v1.13.0/scipy/integrate/_ivp/bdf.py#L370-L375

            # The `cond_fun` check will break out of the loop if not converged but the Jacobian
            # is up to date.  Here we will just update the Jacobian if not converged.
            jac = jax.jacfwd(func, argnums=0)
            current_jac, J, recalc_lu = jax.tree.map(
                partial(jnp.where, ~converged & current_jac),
                (current_jac, J, False),
                (True, jac(y_predict, t_new), True),
            )
            # Update LU: Jacobian has changed (maybe)
            LU = self._lu_factor(M - c * J, recalc_lu, LU)

            return exit_flag, n_iter, converged, current_jac, J, LU, y_new, d

        # Set up loop variables
        converged = False
        d = jnp.zeros_like(y0)
        y_new = jnp.array(y0, copy=True)
        k = 0
        current_jac = state.updated_jacobian
        J = state.J
        exit_flag = False
        carry = (exit_flag, k, converged, current_jac, J, LU, y_new, d)

        # NOTE: This will run a maximum of twice, once with the current (possibly
        # out-of-date) Jacobian, and once with an updated Jacobian.  The loop
        # construction is not necessary but helps minimize "call sites" to the
        # ODE RHS and linear solver, which can be expensive to compile.
        _exit_flag, k, converged, current_jac, J, LU, y_new, d = jax.lax.while_loop(
            _cond_fun, _body_fun, carry
        )

        # Note that `dt` only changes here if it was clipped by the boundary
        # time at the top.  The outer `attempt_bdf_step` function needs to handle
        # updating both `t` and `y` in the state, since it manages adaptive step
        # size apart from the boundary check.
        state = dataclasses.replace(
            state,
            D=D,
            J=J,
            updated_jacobian=current_jac,
            LU=LU,
            dt=h,
            n_equal_steps=n_equal_steps,
        )
        return converged, k, y_new, d, state

    def _lu_factor(self, A, pred=None, LU=None):
        if pred is None:
            return lu_factor(A)
        return jax.tree.map(partial(jnp.where, pred), lu_factor(A), LU)

    def _update_dt(self, state, factor):
        order = state.order
        h = state.dt * factor
        D = _update_D(state.D, order, factor)
        c = h / ALPHA[order]

        # Redo LU factorization (timestep changed)
        LU = self._lu_factor(state.M - c * state.J)
        return dataclasses.replace(state, n_equal_steps=0, dt=h, D=D, LU=LU)

    def attempt_bdf_step(self, func, boundary_time, carry):
        state = carry[0]

        converged, n_iter, y_new, d, state = self.newton_iteration(
            state, func, boundary_time
        )

        # Three cases for adaptive step sizing:
        # (1) The Newton iterations did not converge but the Jacobian has already been
        #   updated, reduce step size by 0.5 and try again.
        # (2) The Newton iterations converged but the error estimate is too high, update
        #   the step size with the optimal factor and try again.
        # (3) The Newton iterations converged and the error estimate is acceptable,
        #   accept the step and update the state for the next step.  The factor for this
        #   case is `-inf`, since the other two cases will have positive scale factors
        updated_jacobian = state.updated_jacobian
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
        scale = self.atol + self.rtol * jnp.abs(y_new)
        error = ERROR_CONST[state.order] * (state.M @ d)
        error_norm = norm(error / scale)
        opt_factor = jnp.maximum(
            MIN_FACTOR, safety * error_norm ** (-1 / (state.order + 1))
        )

        factor = jnp.where(
            ~converged & updated_jacobian,
            0.5,
            jnp.where(error_norm > 1, opt_factor, -jnp.inf),
        )

        # If the factor is negative, then the step is accepted.  Otherwise, we have to
        # update the step size and LU factorization for the next iteration.
        (state, accepted) = jax.tree.map(
            partial(jnp.where, factor > 0),
            (self._update_dt(state, factor), False),
            (state, True),
        )

        return state, accepted, y_new, d, n_iter

    def _update_difference_matrix(self, state, d):
        D, order = state.D, state.order
        D = D.at[order + 2].set(d - D[order + 1])
        D = D.at[order + 1].set(d)

        def body_fun(j, D):
            i = order - j
            return D.at[i].add(D[i + 1])

        D = lax.fori_loop(0, order + 1, body_fun, D)
        return dataclasses.replace(state, D=D)

    def _update_difference_matrix_order_change(self, state, d, y, n_iter):
        D, order = state.D, state.order
        state = self._update_difference_matrix(state, d)
        D = state.D

        scale = self.atol + self.rtol * jnp.abs(y)
        error = ERROR_CONST[order] * d
        error_norm = norm(error / scale)
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

        # Optimal step size factor for order k-1 and k+1
        error_m_norm = jnp.where(
            order > 1,
            norm(ERROR_CONST[order - 1] * D[order] / scale),
            jnp.inf,
        )
        error_p_norm = jnp.where(
            order < MAX_ORDER,
            norm(ERROR_CONST[order + 1] * D[order + 2] / scale),
            jnp.inf,
        )

        error_norms = jnp.array([error_m_norm, error_norm, error_p_norm])
        factors = error_norms ** (-1 / (jnp.arange(3) + order))

        # Select new order to maximize resulting step size, then scale
        # by the corresponding factor
        max_index = jnp.argmax(factors)
        order += max_index - 1

        opt_factor = jnp.minimum(MAX_FACTOR, safety * factors[max_index])
        state = dataclasses.replace(state, D=D, order=order)
        return self._update_dt(state, opt_factor)

    # Inherits docstring from `ODESolverBase`
    def step(self, func, boundary_time, solver_state):
        # https://github.com/scipy/scipy/blob/v1.13.0/scipy/integrate/_ivp/bdf.py#L310-L324
        h = solver_state.dt
        t = solver_state.t
        D = solver_state.D
        order = solver_state.order
        n_equal_steps = solver_state.n_equal_steps
        hmax = self.hmax
        hmin = jnp.maximum(self.hmin, 10 * (jnp.nextafter(t, jnp.inf) - t))
        (h, D, n_equal_steps) = jax.tree.map(
            partial(jnp.where, h > hmax),
            (hmax, _update_D(D, order, hmax / h), 0),
            (h, D, n_equal_steps),
        )
        (h, D, n_equal_steps) = jax.tree.map(
            partial(jnp.where, h < hmin),
            (hmin, _update_D(D, order, hmin / h), 0),
            (h, D, n_equal_steps),
        )
        solver_state = dataclasses.replace(
            solver_state,
            dt=h,
            D=D,
            n_equal_steps=n_equal_steps,
        )

        def cond_fun(carry):
            _, accepted, _, _, _ = carry
            return ~accepted

        y = jnp.zeros_like(solver_state.y)
        d = jnp.zeros_like(solver_state.y)
        (solver_state, _accepted, y_new, d, n_iter) = lax.while_loop(
            cond_fun,
            partial(self.attempt_bdf_step, func, boundary_time),
            (solver_state, False, y, d, -1),
        )

        solver_state = dataclasses.replace(
            solver_state,
            t=solver_state.t + solver_state.dt,
            y=y_new,
            n_equal_steps=solver_state.n_equal_steps + 1,
        )

        return jax.tree.map(
            partial(jnp.where, n_equal_steps < solver_state.order + 1),
            self._update_difference_matrix(solver_state, d),
            self._update_difference_matrix_order_change(solver_state, d, y_new, n_iter),
        )
