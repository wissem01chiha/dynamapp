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

"""Implements Extended Kalman Filter LeafSystem"""

import jax
import jax.numpy as jnp

from ...framework import parameters
from .utils import prepare_continuous_plant_for_nonlinear_kalman_filter

from .kalman_filter_base import KalmanFilterBase


class ExtendedKalmanFilter(KalmanFilterBase):
    """
    Extended Kalman Filter (EKF) for the following system:

        ```
        x[n+1] = f(x[n], u[n]) + G(t[n]) w[n]
        y[n]   = g(x[n], u[n]) + v[n]

        E(w[n]) = E(v[n]) = 0
        E(w[n]w'[n]) = Q(t[n], x[n], u[n])
        E(v[n]v'[n] = R(t[n])
        E(w[n]v'[n] = N(t[n]) = 0
        ```

    `f` and `g` are discrete-time functions of state `x[n]` and control `u[n]`,
    while R` and `G` are discrete-time functions of time `t[n]`. `Q` is a discrete-time
    function of `t[n], x[n], u[n]`. This last aspect is included for zero-order-hold
    discretization of a continuous-time system

    Input ports:
        (0) u[n] : control vector at timestep n
        (1) y[n] : measurement vector at timestep n

    Output ports:
        (1) x_hat[n] : state vector estimate at timestep n

    Parameters:
        dt: float
            Time step of the discrete-time system
        forward: Callable
            A function with signature f(x[n], u[n]) -> x[n+1] that represents `f` in
            the above equations.
        observation: Callable
            A function with signature g(x[n], u[n]) -> y[n] that represents `g` in
            the above equations.
        G_func: Callable
            A function with signature G(t[n]) -> G[n] that represents `G` in
            the above equations.
        Q_func: Callable
            A function with signature Q(t[n], x[n], u[n]) -> Q[n] that represents `Q`
            in the above equations.
        R_func: Callable
            A function with signature R(t[n]) -> R[n] that represents `R` in
            the above equations.
        x_hat_0: ndarray
            Initial state estimate
        P_hat_0: ndarray
            Initial state covariance matrix estimate
    """

    @parameters(
        static=[
            "dt",
            "forward",
            "observation",
            "G_func",
            "Q_func",
            "R_func",
            "x_hat_0",
            "P_hat_0",
        ],
    )
    def __init__(
        self,
        dt,
        forward,
        observation,
        G_func,
        Q_func,
        R_func,
        x_hat_0,
        P_hat_0,
        name=None,
        **kwargs,
    ):
        super().__init__(dt, x_hat_0, P_hat_0, name, **kwargs)

    def initialize(
        self,
        dt,
        forward,
        observation,
        G_func,
        Q_func,
        R_func,
        x_hat_0,
        P_hat_0,
    ):
        self.G_func = G_func
        self.Q_func = Q_func
        self.R_func = R_func

        self.nx = x_hat_0.size
        self.ny = self.R_func(0.0).shape[0]

        self.forward = forward
        self.observation = observation

        self.jac_forward = jax.jacfwd(forward)
        self.jac_observation = jax.jacfwd(observation)

        self.eye_x = jnp.eye(self.nx)

    def _correct(self, time, x_hat_minus, P_hat_minus, *inputs):
        u, y = inputs
        u = jnp.atleast_1d(u)
        y = jnp.atleast_1d(y)

        C = self.jac_observation(x_hat_minus, u).reshape((self.ny, self.nx))

        R = self.R_func(time)

        # TODO: improved numerics to avoud computing explicit inverse
        K = P_hat_minus @ C.T @ jnp.linalg.inv(C @ P_hat_minus @ C.T + R)

        x_hat_plus = x_hat_minus + jnp.dot(
            K, y - self.observation(x_hat_minus, u)
        )  # n|n

        P_hat_plus = jnp.matmul(self.eye_x - jnp.matmul(K, C), P_hat_minus)  # n|n

        return x_hat_plus, P_hat_plus

    def _propagate(self, time, x_hat_plus, P_hat_plus, *inputs):
        # Predict -- x_hat_plus of current step is propagated to be the
        # x_hat_minus of the next step
        # k+1|k in current step is n|n-1 for next step

        u, y = inputs
        u = jnp.atleast_1d(u)

        A = self.jac_forward(x_hat_plus, u).reshape((self.nx, self.nx))

        G = self.G_func(time)
        Q = self.Q_func(time, x_hat_plus, u)
        GQGT = G @ Q @ G.T

        x_hat_minus = self.forward(x_hat_plus, u)  # n+1|n
        P_hat_minus = A @ P_hat_plus @ A.T + GQGT  # n+1|n

        return x_hat_minus, P_hat_minus

    #######################################
    # Make filter for a continuous plant  #
    #######################################

    @staticmethod
    def for_continuous_plant(
        plant,
        dt,
        G_func,
        Q_func,
        R_func,
        x_hat_0,
        P_hat_0,
        discretization_method="euler",
        discretized_noise=False,
        name=None,
        ui_id=None,
    ):
        """
        Extended Kalman Filter system for a continuous-time plant.

        The input plant contains the deterministic forms of the forward and observation
        operators:

        ```
            dx/dt = f(x,u)
            y = g(x,u)
        ```

        Note: (i) Only plants with one vector-valued input and one vector-valued output
        are currently supported. Furthermore, the plant LeafSystem/Diagram should have
        only one vector-valued integrator; (ii) the user may pass a plant with
        disturbances (not recommended) as the input plant. In this case, the forward
        and observation evaluations will be corrupted by noise.

        A plant with disturbances of the following form is then considered:

        ```
            dx/dt = f(x,u) + G(t) w         -- (C1)
            y = g(x,u) +  v                 -- (C2)
        ```

        where:

            `w` represents the process noise,
            `v` represents the measurement noise,

        and

        ```
            E(w) = E(v) = 0
            E(ww') = Q(t)
            E(vv') = R(t)
            E(wv') = N(t) = 0
        ```

        This plant is discretized to obtain the following form:

        ```
            x[n+1] = fd(x[n], u[n]) + Gd w[n]  -- (D1)
            y[n]   = gd(x[n], u[n]) + v[n]     -- (D2)

            E(w[n]) = E(v[n]) = 0
            E(w[n]w'[n]) = Qd
            E(v[n]v'[n] = Rd
            E(w[n]v'[n] = Nd = 0
        ```

        The above discretization is performed either via the `euler` or the `zoh`
        method, and an Extended Kalman Filter estimator for the system of equations
        (D1) and (D2) is returned.

        Note: If `discretized_noise` is True, then it is assumed that the user is
        directly providing Gd, Qd and Rd. If False, then Qd and Rd are computed from
        continuous-time Q, R, and G, and Gd is set to an Identity matrix.

        The returned system will have:

        Input ports:
            (0) u[n] : control vector at timestep n
            (1) y[n] : measurement vector at timestep n

        Output ports:
            (1) x_hat[n] : state vector estimate at timestep n

        Parameters:
            plant : a `Plant` object which can be a LeafSystem or a Diagram.
            dt: float
                Time step for the discretization.
            G_func: Callable
                A function with signature G(t) -> G that represents `G` in
                the continuous-time equations (C1) and (C2).
            Q_func: Callable
                A function with signature Q(t) -> Q that represents `Q` in
                the continuous-time equations (C1) and (C2).
            R_func: Callable
                A function with signature R(t) -> R that represents `R` in
                the continuous-time equations (C1) and (C2).
            x_hat_0: ndarray
                Initial state estimate
            P_hat_0: ndarray
                Initial state covariance matrix estimate. If `None`, an Identity
                matrix is assumed.
            discretization_method: str ("euler" or "zoh")
                Method to discretize the continuous-time plant. Default is "euler".
            discretized_noise: bool
                Whether the user is directly providing Gd, Qd and Rd. Default is False.
                If True, `G_func`, `Q_func`, and `R_func` provide Gd(t), Qd(t), and
                Rd(t), respectively.
        """

        (
            forward,
            observation,
            Gd_func,
            Qd_func,
            Rd_func,
        ) = prepare_continuous_plant_for_nonlinear_kalman_filter(
            plant,
            dt,
            G_func,
            Q_func,
            R_func,
            x_hat_0,
            discretization_method,
            discretized_noise,
        )

        nx = x_hat_0.size
        if P_hat_0 is None:
            P_hat_0 = jnp.eye(nx)

        # TODO: If Gd_func is None, compute Gd automatically with u = u + w

        ekf = ExtendedKalmanFilter(
            dt,
            forward,
            observation,
            Gd_func,
            Qd_func,
            Rd_func,
            x_hat_0,
            P_hat_0,
            name=name,
            ui_id=ui_id,
        )

        return ekf

    ###################################################################################
    # Make filter from direct specification of forward/observaton operators and noise #
    ###################################################################################

    @staticmethod
    def from_operators(
        dt,
        forward,
        observation,
        G_func,
        Q_func,
        R_func,
        x_hat_0,
        P_hat_0,
        name=None,
        ui_id=None,
    ):
        """
        Extended Kalman Filter (UKF) for the following system:

        ```
            x[n+1] = f(x[n], u[n]) + G(t[n]) w[n]
            y[n]   = g(x[n], u[n]) + v[n]

            E(w[n]) = E(v[n]) = 0
            E(w[n]w'[n]) = Q(t[n], x[n], u[n])
            E(v[n]v'[n] = R(t[n])
            E(w[n]v'[n] = N(t[n]) = 0
        ```

        `f` and `g` are discrete-time functions of state `x[n]` and control `u[n]`,
        while `Q` and `R` and `G` are discrete-time functions of time `t[n]`.

        Input ports:
            (0) u[n] : control vector at timestep n
            (1) y[n] : measurement vector at timestep n

        Output ports:
            (1) x_hat[n] : state vector estimate at timestep n

        Parameters:
            dt: float
                Time step of the discrete-time system
            forward: Callable
                A function with signature f(x[n], u[n]) -> x[n+1] that represents `f`
                in the above equations.
            observation: Callable
                A function with signature g(x[n], u[n]) -> y[n] that represents `g` in
                the above equations.
            G_func: Callable
                A function with signature G(t[n]) -> G[n] that represents `G` in
                the above equations.
            Q_func: Callable
                A function with signature Q(t[n]) -> Q[n] that represents
                `Q` in the above equations.
            R_func: Callable
                A function with signature R(t[n]) -> R[n] that represents `R` in
                the above equations.
            x_hat_0: ndarray
                Initial state estimate
            P_hat_0: ndarray
                Initial state covariance matrix estimate
        """

        def Q_func_aug(t, x_k, u_k):
            return Q_func(t)

        ekf = ExtendedKalmanFilter(
            dt,
            forward,
            observation,
            G_func,
            Q_func_aug,
            R_func,
            x_hat_0,
            P_hat_0,
            name=name,
            ui_id=ui_id,
        )

        return ekf
