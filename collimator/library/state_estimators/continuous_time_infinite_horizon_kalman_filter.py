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

"""Implements Continuous-time Infinite Horizon Kalman Filter LeafSystem"""

from ...backend import numpy_api as cnp

from .utils import linearize_plant

from ...framework import LeafSystem

from ...lazy_loader import LazyLoader

control = LazyLoader(
    "control", globals(), "control"
)  # For formatting state-space systems


class ContinuousTimeInfiniteHorizonKalmanFilter(LeafSystem):
    """
    Continuous-time Infinite Horizon Kalman Filter for the following system:

    ```
    dot_x =  A x + B u + G w
    y   = C x + D u + v

    E(w) = E(v) = 0
    E(ww') = Q
    E(vv') = R
    E(wv') = N = 0
    ```

    Input ports:
        (0) u : continuous-time control vector
        (1) y : continuous-time measurement vector

    Output ports:
        (1) x_hat : continuous-time state vector estimate

    Parameters:
        A: ndarray
            State transition matrix
        B: ndarray
            Input matrix
        C: ndarray
            Output matrix
        D: ndarray
            Feedthrough matrix
        G: ndarray
            Process noise matrix
        Q: ndarray
            Process noise covariance matrix
        R: ndarray
            Measurement noise covariance matrix
        x_hat_0: ndarray
            Initial state estimate
    """

    def __init__(self, A, B, C, D, G, Q, R, x_hat_0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.G = G
        self.Q = Q
        self.R = R

        self.nx, self.nu = B.shape
        self.ny = C.shape[0]

        L, P, E = control.lqe(A, G, C, Q, R)

        self.A_minus_LC = A - cnp.matmul(L, C)
        self.B_minus_LD = B - cnp.matmul(L, D)
        self.L = L

        self.declare_input_port()  # u
        self.declare_input_port()  # y

        self.declare_continuous_state(
            ode=self._ode, shape=x_hat_0.shape, default_value=x_hat_0, as_array=True
        )  # continuous state for x_hat

        self.declare_continuous_state_output()

    def _ode(self, time, state, *inputs, **params):
        x_hat = state.continuous_state

        u, y = inputs

        u = cnp.atleast_1d(u)
        y = cnp.atleast_1d(y)

        dot_x_hat = (
            cnp.dot(self.A_minus_LC, x_hat)
            + cnp.dot(self.B_minus_LD, u)
            + cnp.dot(self.L, y)
        )

        return dot_x_hat

    #######################################
    # Make filter for a continuous plant  #
    #######################################
    @staticmethod
    def for_continuous_plant(
        plant,
        x_eq,
        u_eq,
        Q,
        R,
        G=None,
        x_hat_bar_0=None,
        name=None,
    ):
        """
        Obtain a continuous-time Infinite Horizon Kalman Filter system for a
        continuous-time plant after linearization at equilibrium point (x_eq, u_eq)

        The input plant contains the deterministic forms of the forward and observation
        operators:

        ```
            dx/dt = f(x,u)
            y = g(x,u)
        ```

        Note: Only plants with one vector-valued input and one vector-valued output
        are currently supported. Furthermore, the plant LeafSystem/Diagram should have
        only one vector-valued integrator.

        A plant with disturbances of the following form is then considered
        following form:

        ```
            dx/dt = f(x,u) + G w
            y = g(x,u) +  v
        ```

        where:

            `w` represents the process noise,
            `v` represents the measurement noise,

        and

        ```
            E(w) = E(v) = 0
            E(ww') = Q
            E(vv') = R
            E(wv') = N = 0
        ```

        This plant with disturbances is linearized (only `f` and `q`) around the
        equilibrium point to obtain:

        ```
            d/dt (x_bar) = A x_bar + B u_bar + G w    --- (C1)
            y_bar = C x_bar + D u_bar + v             --- (C2)
        ```

        where,

        ```
            x_bar = x - x_eq
            u_bar = u - u_eq
            y_bar = y - y_bar
            y_eq = g(x_eq, u_eq)
        ```

        A continuous-time Kalman Filter estimator for the system of equations (C1) and
        (C2) is returned. This filter is in the `x_bar`, `u_bar`, and `y_bar`
        states.

        The returned system will have

        Input ports:
            (0) u_bar : continuous-time control vector relative to equilibrium point
            (1) y_bar : continuous-time measurement vector relative to equilibrium point

        Output ports:
            (1) x_hat_bar : continuous-time state vector estimate relative to
                            equilibrium point

        Parameters:
            plant : a `Plant` object which can be a LeafSystem or a Diagram.
            x_eq: ndarray
                Equilibrium state vector for discretization
            u_eq: ndarray
                Equilibrium control vector for discretization
            Q: ndarray
                Process noise covariance matrix.
            R: ndarray
                Measurement noise covariance matrix.
            G: ndarray
                Process noise matrix. If `None`, `G=B` is assumed making disrurbances
                additive to control vector `u`, i.e. `u_disturbed = u_orig + w`.
            x_hat_bar_0: ndarray
                Initial state estimate relative to equilibrium point.
                If None, an identity matrix is assumed.
        """

        y_eq, linear_plant = linearize_plant(plant, x_eq, u_eq)

        A, B, C, D = linear_plant.A, linear_plant.B, linear_plant.C, linear_plant.D

        nx, nu = B.shape
        ny, _ = D.shape

        if G is None:
            G = B

        if x_hat_bar_0 is None:
            x_hat_bar_0 = cnp.zeros(nx)

        # Instantiate a Kalman Filter instance for the linearized plant
        kf = ContinuousTimeInfiniteHorizonKalmanFilter(
            A,
            B,
            C,
            D,
            G,
            Q,
            R,
            x_hat_bar_0,
            name=name,
        )

        return y_eq, kf
