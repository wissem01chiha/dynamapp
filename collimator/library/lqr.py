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
Implementation of LQR and finite horizon LQR
"""

import jax
import jax.numpy as jnp

from .utils import make_ode_rhs
from ..framework import LeafSystem

from ..library import FeedthroughBlock

from ..lazy_loader import LazyLoader

control = LazyLoader(
    "control", globals(), "control"
)  # For formatting state-space systems

diffrax = LazyLoader("diffrax", globals(), "diffrax")


class LinearQuadraticRegulator(FeedthroughBlock):
    """
    Linear Quadratic Regulator (LQR) for a continuous-time system:
            dx/dt = A x + B u.
    Computes the optimal control input:
            u = -K x,
    where u minimises the cost function over [0, ∞)]:
            J = ∫(x.T Q x + u.T R u) dt.

    Input ports:
        (0) x: state vector of the system.

    Output ports:
        (0) u: optimal control vector.

    Parameters:
        A: Array
            State matrix of the system.
        B: Array
            Input matrix of the system.
        Q: Array
            State cost matrix.
        R: Array
            Input cost matrix.
    """

    def __init__(self, A, B, Q, R, *args, **kwargs):
        self.K, S, E = control.lqr(A, B, Q, R)
        super().__init__(lambda x: jnp.matmul(-self.K, x), *args, **kwargs)


class DiscreteTimeLinearQuadraticRegulator(LeafSystem):
    """
    Linear Quadratic Regulator (LQR) for a discrete-time system:
            x[k+1] = A x[k] + B u[k].
    Computes the optimal control input:
            u[k] = -K x[k],
    where u minimises the cost function over [0, ∞)]:
            J = ∑(x[k].T Q x[k] + u[k].T R u[k]).

    Input ports:
        (0) x[k]: state vector of the system.

    Output ports:
        (0) u[k]: optimal control vector.

    Parameters:
        A: Array
            State matrix of the system.
        B: Array
            Input matrix of the system.
        Q: Array
            State cost matrix.
        R: Array
            Input cost matrix.
        dt: float
            Sampling period of the system.
    """

    def __init__(self, A, B, Q, R, dt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K, S, E = control.dlqr(A, B, Q, R)

        self.declare_input_port()  # for state x

        self.declare_output_port(
            self._get_opt_u,
            requires_inputs=True,
            period=dt,
            offset=0.0,
            default_value=jnp.zeros(B.shape[1]),
        )

    def _get_opt_u(self, time, state, x, **params):
        return jnp.matmul(-self.K, x)


class FiniteHorizonLinearQuadraticRegulator(LeafSystem):
    """
    Finite Horizon Linear Quadratic Regulator (LQR) for a continuous-time system.
    Solves the Riccati Differential Equation (RDE) to compute the optimal control
    for the following finitie horizon cost function over [t0, tf]:

    Minimise cost J:

        J = [x(tf) - xd(tf)].T Qf [x(tf) - xd(tf)]
            + ∫[(x(t) - xd(t)].T Q [(x(t) - xd(t)] dt
            + ∫[(u(t) - ud(t)].T R [(u(t) - ud(t)] dt
            + 2 ∫[(x(t) - xd(t)].T N [(u(t) - ud(t)] dt

    subject to the constraints:

    dx(t)/dt - dx0(t)/dt = A [x(t)-x0(t)] + B [u(t)-u0(t)] - c(t),

    where,
        x(t) is the state vector,
        u(t) is the control vector,
        xd(t) is the desired state vector,
        ud(t) is the desired control vector,
        x0(t) is the nominal state vector,
        u0(t) is the nominal control vector,
        Q, R, and N are the state, input, and cross cost matrices,
        Qf is the final state cost matrix,

    and A, B, and c are computed from linearisation of the plant `df/dx = f(x, u)`
    around the nominal trajectory (x0(t), u0(t)).

        A = df/dx(x0(t), u0(t), t)
        B = df/du(x0(t), u0(t), t)
        c = f(x0(t), u0(t), t) - dx0(t)/dt

    The optimal control `u` obtained by the solution of the above problem is output.

    See Section 8.5.1 of https://underactuated.csail.mit.edu/lqr.html#finite_horizon

    Parameters:
        t0 : float
            Initial time of the finite horizon.
        tf : float
            Final time of the finite horizon.
        plant : a `Plant` object which can be a LeafSystem or a Diagram.
            The plant to be controlled. This represents `df/dx = f(x, u)`.
        Qf : Array
            Final state cost matrix.
        func_Q : Callable
            A function that returns the state cost matrix Q at time `t`: `func_Q(t)->Q`
        func_R : Callable
            A function that returns the input cost matrix R at time `t`: `func_R(t)->R`
        func_N : Callable
            A function that returns the cross cost matrix N at time `t`. `func_N(t)->N`
        func_x_0 : Callable
            A function that returns the nominal state vector `x0` at time `t`.
            func_x_0(t)->x0
        func_u_0 : Callable
            A function that returns the nominal control vector `u0` at time `t`.
            func_u_0(t)->u0
        func_x_d : Callable
            A function that returns the desired state vector `xd` at time `t`.
            func_x_d(t)->xd.  If None, assumed to be the same as the nominal trajectory.
        func_u_d : Callable
            A function that returns the desired control vector `ud` at time `t`.
            func_u_d(t)->ud.  If None, assumed to be the same as the nominal trajectory.
    """

    def __init__(
        self,
        t0,
        tf,
        plant,
        Qf,
        func_Q,
        func_R,
        func_N,
        func_x_0,
        func_u_0,
        func_x_d=None,
        func_u_d=None,
        name=None,
    ):
        super().__init__(name=name)

        self.t0 = t0
        self.tf = tf

        if func_x_d is None:
            func_x_d = func_x_0

        if func_u_d is None:
            func_u_d = func_u_0

        self.func_R = func_R
        self.func_N = func_N
        self.func_x_0 = func_x_0
        self.func_u_0 = func_u_0
        self.func_x_d = func_x_d
        self.func_u_d = func_u_d

        func_dot_x_0 = jax.jacfwd(func_x_0)
        nu = func_R(t0).shape[0]

        ode_rhs = make_ode_rhs(plant, nu)
        get_A = jax.jacfwd(ode_rhs, argnums=0)
        self.get_B = jax.jacfwd(ode_rhs, argnums=1)

        @jax.jit
        def rde(t, rde_state, args):
            t = -t
            Sxx, sx = rde_state

            Sxx = (Sxx + Sxx.T) / 2.0

            # Get nominal trajectories, desired trajectories, and cost matrices
            x_0 = func_x_0(t)
            u_0 = func_u_0(t)

            x_d = func_x_d(t)
            u_d = func_u_d(t)

            Q = func_Q(t)
            R = func_R(t)
            N = func_N(t)

            # Calculate dynamics mismatch due to nominal traj not satisfying dynamics
            dot_x_0 = func_dot_x_0(t)
            dot_x_0_eval = ode_rhs(x_0, u_0, t)
            c = dot_x_0_eval - dot_x_0

            #  Get linearisation around x_0, u_0
            A = get_A(x_0, u_0, t)
            B = self.get_B(x_0, u_0, t)

            #  Desired trajectories relative to nominal
            x_d_0 = x_d - x_0
            u_d_0 = u_d - u_0

            #  Compute RHS of RDE
            qx = -jnp.dot(Q, x_d_0) - jnp.dot(N, u_d_0)
            ru = -jnp.dot(R, u_d_0) - jnp.dot(N.T, x_d_0)

            N_plus_Sxx_B = N + jnp.matmul(Sxx, B)

            Rinv = jnp.linalg.inv(R)
            Sxx_A = jnp.matmul(Sxx, A)

            dot_Sxx = (
                Q
                - jnp.matmul(N_plus_Sxx_B, jnp.matmul(Rinv, N_plus_Sxx_B.T))
                + Sxx_A
                + Sxx_A.T
            )

            dot_sx = (
                qx
                - jnp.dot(N_plus_Sxx_B, jnp.dot(Rinv, ru + jnp.dot(B.T, sx)))
                + jnp.dot(A.T, sx)
                + jnp.dot(Sxx, c)
            )

            return (dot_Sxx, dot_sx)

        term = diffrax.ODETerm(rde)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5, dtmax=0.1)
        saveat = diffrax.SaveAt(dense=True)

        # TODO: Use utilities in ../simulation/ for reduced reliance on diffrax
        self.sol_rde = diffrax.diffeqsolve(
            term,
            solver,
            -tf,
            -t0,
            y0=(Qf, -jnp.dot(Qf, func_x_d(tf) - func_x_0(tf))),
            dt0=0.0001,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )

        # Input: current state (x)
        self.declare_input_port()

        # Output port: Optimal finite horizon LQR control
        self.declare_output_port(self._eval_output, default_value=jnp.zeros(nu))

    def _eval_output(self, time, state, x, **params):
        rde_time = jnp.clip(time, self.t0, self.tf)
        rde_time = -rde_time

        Sxx, sx = self.sol_rde.evaluate(rde_time)

        x_d = self.func_x_d(time)
        u_d = self.func_u_d(time)

        x_0 = self.func_x_0(time)
        u_0 = self.func_u_0(time)

        x_d_0 = x_d - x_0
        u_d_0 = u_d - u_0

        B = self.get_B(x_0, u_0, time)

        R = self.func_R(time)
        N = self.func_N(time)
        Rinv = jnp.linalg.inv(R)

        ru = -jnp.dot(R, u_d_0) - jnp.dot(N.T, x_d_0)

        Rinv = jnp.linalg.inv(R)
        N_plus_Sxx_B = N + jnp.matmul(Sxx, B)

        u = (
            u_0
            - jnp.dot(Rinv, jnp.dot(N_plus_Sxx_B.T, (x - x_0)))
            - jnp.dot(Rinv, ru + jnp.dot(B.T, sx))
        )

        return u
