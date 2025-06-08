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

"""Implements Unscented Kalman Filter LeafSystem"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ...framework import parameters
from .utils import prepare_continuous_plant_for_nonlinear_kalman_filter

from .kalman_filter_base import KalmanFilterBase


class UnscentedKalmanFilter(KalmanFilterBase):
    """
    Unscented Kalman Filter (UKF) for the following system:

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
        alpha: float
            Sigma point spread to control the amount of nonlinearities taken into
            account. Usually set to a value (1e-04<= alpha <= 1.0). Default is 1.0.
        beta: float
            Scaling constant to include prior information about the distribution of
            the state. Default is 0.0.
        kappa: float
            Relatively non-critical parameter to control the kurtosis of sigma point
            distribution. Default is 0.0.
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
            "alpha",
            "beta",
            "kappa",
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
        alpha=1.0,
        beta=0.0,
        kappa=0.0,
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
        alpha=1.0,
        beta=0.0,
        kappa=0.0,
    ):
        self.G_func = G_func
        self.Q_func = Q_func
        self.R_func = R_func

        self.nx = x_hat_0.size
        self.ny = self.R_func(0.0).shape[0]

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.forward = forward
        self.observation = observation

        self.forward_sigma_points = jax.vmap(forward, in_axes=(0, None))
        self.observation_sigma_points = jax.vmap(observation, in_axes=(0, None))

        self.num_sigma_points = 2 * self.nx + 1
        self.lamb = (self.alpha**2.0) * (self.nx + self.kappa) - self.nx
        self.lamb_plus_nx = self.lamb + self.nx

        self.weights_mean = jnp.full(2 * self.nx + 1, 0.5 / (self.lamb + self.nx))
        self.weights_mean = self.weights_mean.at[0].set(
            self.lamb / (self.lamb + self.nx)
        )

        self.weights_cov = jnp.full(2 * self.nx + 1, 0.5 / (self.lamb + self.nx))
        self.weights_cov = self.weights_cov.at[0].set(
            self.lamb / (self.lamb + self.nx) + (1.0 - alpha**2 + beta)
        )

    def _gen_sigma_points(self, mean, cov):
        chol_cov = jsp.linalg.cholesky(
            self.lamb_plus_nx * cov
        )  # upper triangular Cholesky fact.

        sigma_points_plus = mean + chol_cov
        sigma_points_minus = mean - chol_cov

        sigma_points = jnp.vstack([mean, sigma_points_plus, sigma_points_minus])

        return sigma_points

    def _get_weighted_mean_and_cov_from_sigma_points(self, sigma_points):
        mean = jnp.dot(self.weights_mean, sigma_points)
        delta_sigma_points = sigma_points - mean
        cov = delta_sigma_points.T @ jnp.diag(self.weights_cov) @ delta_sigma_points

        return mean, cov

    def _get_weighted_cross_covariance_from_sigma_points(
        self, sigma_points_x, sigma_points_y
    ):
        mean_x = jnp.dot(self.weights_mean, sigma_points_x)
        delta_sigma_points_x = sigma_points_x - mean_x

        mean_y = jnp.dot(self.weights_mean, sigma_points_y)
        delta_sigma_points_y = sigma_points_y - mean_y

        cov_xy = (
            delta_sigma_points_x.T @ jnp.diag(self.weights_cov) @ delta_sigma_points_y
        )

        return cov_xy

    def _correct(self, time, x_hat_minus, P_hat_minus, *inputs):
        u, y = inputs
        u = jnp.atleast_1d(u)
        y = jnp.atleast_1d(y)

        sigma_points_x_minus = self._gen_sigma_points(x_hat_minus, P_hat_minus).reshape(
            (self.num_sigma_points, self.nx)
        )

        sigma_points_y_minus = self.observation_sigma_points(
            sigma_points_x_minus, u
        ).reshape((self.num_sigma_points, self.ny))

        y_mean, Py = self._get_weighted_mean_and_cov_from_sigma_points(
            sigma_points_y_minus
        )

        Pxy = self._get_weighted_cross_covariance_from_sigma_points(
            sigma_points_x_minus,
            sigma_points_y_minus,
        )

        R = self.R_func(time)
        S = Py + R

        # TODO: improved numerics to avoud computing explicit inverse
        K = jnp.matmul(Pxy, jnp.linalg.inv(S))

        x_hat_plus = x_hat_minus + jnp.dot(K, y - y_mean)  # n|n
        P_hat_plus = P_hat_minus - K @ S @ K.T  # n|n

        return x_hat_plus, P_hat_plus

    def _propagate(self, time, x_hat_plus, P_hat_plus, *inputs):
        # Predict -- x_hat_plus of current step is propagated to be the
        # x_hat_minus of the next step
        # k+1|k in current step is k|k-1 for next step

        u, y = inputs
        u = jnp.atleast_1d(u)

        G = self.G_func(time)
        Q = self.Q_func(time, x_hat_plus, u)
        GQGT = G @ Q @ G.T

        sigma_points_x_plus = self._gen_sigma_points(x_hat_plus, P_hat_plus).reshape(
            (self.num_sigma_points, self.nx)
        )

        sigma_points_x_minus = self.forward_sigma_points(
            sigma_points_x_plus, u
        ).reshape((self.num_sigma_points, self.nx))

        x_hat_minus, Px = self._get_weighted_mean_and_cov_from_sigma_points(
            sigma_points_x_minus
        )  # n+1|n

        P_hat_minus = Px + GQGT  # n+1|n

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
        alpha=1.0,
        beta=0.0,
        kappa=0.0,
        name=None,
        ui_id=None,
    ):
        """
        Unscented Kalman Filter system for a continuous-time plant.

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
        method, and an Unscented Kalman Filter estimator for the system of equations
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
            alpha: float
                Sigma point spread to control the amount of nonlinearities taken into
                account. Usually set to a value (1e-04<= alpha <= 1.0). Default is 1.0.
            beta: float
                Scaling constant to include prior information about the distribution of
                the state. Default is 0.0.
            kappa: float
                Relatively non-critical parameter to control the kurtosis of sigma
                point distribution. Default is 0.0.
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

        ukf = UnscentedKalmanFilter(
            dt,
            forward,
            observation,
            Gd_func,
            Qd_func,
            Rd_func,
            x_hat_0,
            P_hat_0,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            name=name,
            ui_id=ui_id,
        )

        return ukf

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
        alpha=1.0,
        beta=0.0,
        kappa=0.0,
        name=None,
        ui_id=None,
    ):
        """
        Unscented Kalman Filter (UKF) for the following system:

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
            alpha: float
                Sigma point spread to control the amount of nonlinearities taken into
                account. Usually set to a value (1e-04<= alpha <= 1.0). Default is 1.0.
            beta: float
                Scaling constant to include prior information about the distribution of
                the state. Default is 0.0.
            kappa: float
                Relatively non-critical parameter to control the kurtosis of sigma
                point distribution. Default is 0.0.
        """

        def Q_func_aug(t, x_k, u_k):
            return Q_func(t)

        ukf = UnscentedKalmanFilter(
            dt,
            forward,
            observation,
            G_func,
            Q_func_aug,
            R_func,
            x_hat_0,
            P_hat_0,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            name=name,
            ui_id=ui_id,
        )

        return ukf
