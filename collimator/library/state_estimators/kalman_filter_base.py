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

"""Base class for discrete-time Kalman Filters and its variants"""

from abc import ABC, abstractmethod
from typing import NamedTuple

from collimator.backend import numpy_api as cnp

from ...framework import LeafSystem


class KalmanFilterBase(LeafSystem, ABC):
    """
    Base class for discrete-time Kalman Filters and its variants.

    ```
                        +----------------------------+
    --- u[n] --->-------|                            |
                        |  Kalman Filter or variant  |----->--- x_hat[n] ---
    --- y[n] --->------+|                            |
                        +----------------------------+
    ```

    The filter starts with a guess `x_hat_0` of the state vector `x` and a guess
    `P_hat_0` of the state covariance matrix `P`.

    At the beginning of each timestep [n], the filter corrects its state and covariance
    estimates based on the control input `u[n]` and the measurement `y[n]`. This is the
    `correct` step.

    Correct step:
        `x_hat[n|n], P_hat[n|n] = correct(x_hat[n|n-1], P_hat[n|n-1], u[n], y[n])`

        Here, x_hat[n|n-1] and P_hat[n|n-1] represent the state estimate given all the
        measurements up tp t[n-1]

    This corrected state is the output of the filter between t[n] and t[n+1].

    Subsequently, after state correction, and also at the beginning of the time step
    t[n], the corrected state and covariance are propagated forwad to the next time
    step t[n+1] using the dynamics. This is the 'predict' or `propagate` step.

    Propagate step:
        `x_hat[n+1|n], P_hat[n+1|n] = propagate(x_hat[n|n], P_hat[n|n], u[n])`

        These internal predicted state and covariance (n+1|n) are the pre-corrected
        state and covariance for time step t[n+1].

    In the implementation below, the pre-corrected state and covariances are denoted by
    `x_hat_minus` and `P_hat_minus`, respectively, and the post-corrected state and
    covariances are denoted by `x_hat_plus` and `P_hat_plus`, respectively.

    The various filters that inherit from this class implement the `correct` and
    the `predict` steps. The `correct` step is implemented by the `_correct` method,
    and the `predict` step is implemented by the `_propagate` method.

    All filters created by this class will have the following ports:

    Input ports:
        (0) u[n] : control vector at timestep n
        (1) y[n] : measurement vector at timestep n

    Output ports:
        (1) x_hat[n] : state vector estimate at timestep n

    Parameters (apart from those defined by child classes)
        dt: float
            Time step of the discrete-time system
        x_hat_0: ndarray
            Initial state estimate
        P_hat_0: ndarray
            Initial state covariance matrix estimate
    """

    class DiscreteStateType(NamedTuple):
        """
        The state maintained by the Filter. It contains both `_minus` and `plus` states
        because, the output of the filter at t[n] is the `_plus` state, while the
        propagated state (the `minus` state of the next time step) needs to be stored
        for the next time step.
        """

        x_hat_minus: cnp.ndarray
        P_hat_minus: cnp.ndarray
        x_hat_plus: cnp.ndarray
        P_hat_plus: cnp.ndarray

    def __init__(
        self,
        dt,
        x_hat_0,
        P_hat_0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dt = dt

        self.declare_input_port()  # u
        self.declare_input_port()  # y

        self.declare_discrete_state(
            default_value=self.DiscreteStateType(
                x_hat_minus=x_hat_0,
                P_hat_minus=P_hat_0,
                x_hat_plus=x_hat_0,
                P_hat_plus=P_hat_0,
            ),
            as_array=False,
        )

        self.declare_periodic_update(
            self._update,
            period=dt,
            offset=0.0,
        )

        self.declare_output_port(
            self._feedthrough_output_x_hat,
            period=dt,
            offset=0.0,
            default_value=x_hat_0,
            requires_inputs=True,
        )

    @abstractmethod
    def _correct(self, time, x_hat_minus, P_hat_minus, *inputs):
        """
        Implements the correct step:

            `x_hat_plus, P_hat_plus = correct(x_hat_minus, P_hat_minus, u[n], y[n])`

        where:
            x_hat_minus = x_hat[n|n-1]
            P_hat_minus = P_hat[n|n-1]
            u[n], y[n] = inputs

            x_hat_plus = x_hat[n|n]
            P_hat_plus = P_hat[n|n]

        and returns:
            x_hat_plus, P_hat_plus
        """
        pass

    @abstractmethod
    def _propagate(self, time, x_hat_plus, P_hat_plus, *inputs):
        """
        Implements the propagate step:

            `x_hat_minus, P_hat_minus = propagate(x_hat_plus, P_hat_plus, u[n], y[n])`

        where:
            x_hat_plus = x_hat[n|n]
            P_hat_plus = P_hat[n|n]
            u[n], y[n] = inputs

            x_hat_minus = x_hat[n+1|n] (or x_hat[n|n-1] for the next step)
            P_hat_minus = P_hat[n+1|n] (or P_hat[n|n-1] for the next step)

        and returns:
            x_hat_minus, P_hat_minus
        """
        pass

    def _update(self, time, state, *inputs, **params):
        x_hat_minus = state.discrete_state.x_hat_minus  # n|n-1
        P_hat_minus = state.discrete_state.P_hat_minus  # n|n-1

        # FIXME: the computation below has already been performed in the
        # `feedthrough_output_x_hat` method. We should be able to reuse that with
        # caching?
        x_hat_plus, P_hat_plus = self._correct(
            time, x_hat_minus, P_hat_minus, *inputs
        )  # n|n

        x_hat_minus, P_hat_minus = self._propagate(
            time, x_hat_plus, P_hat_plus, *inputs
        )  # n+1|n (or n|n-1 for next step)

        return self.DiscreteStateType(
            x_hat_minus=x_hat_minus,
            P_hat_minus=P_hat_minus,
            x_hat_plus=x_hat_plus,
            P_hat_plus=P_hat_plus,
        )

    def _feedthrough_output_x_hat(self, time, state, *inputs, **params):
        x_hat_minus = state.discrete_state.x_hat_minus  # n|n-1
        P_hat_minus = state.discrete_state.P_hat_minus  # n|n-1

        x_hat_plus, P_hat_plus = self._correct(
            time, x_hat_minus, P_hat_minus, *inputs
        )  # n|n

        return x_hat_plus
