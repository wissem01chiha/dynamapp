from typing import List, Optional
import pandas as pd
import jax.numpy as jnp

from .state_space import StateSpace
from .math_utils  import validate_matrix_shape

class Kalman:
    r"""
    Implementation [1] of a Kalman filter for a state-space model ``state_space``:

    .. math::
        \begin{cases}
            x_{k+1} &= A x_k + B u_k + w_k \\
            y_k &= C x_k + D u_k + v_k
        \end{cases}

    The matrices :math:`(A, B, C, D)` are taken from the state-space model ``state_space``.
    The measurement-noise :math:`v_k` and process-noise :math:`w_k` have a covariance matrix
    ``noise_covariance`` defined as

    .. math::
        \texttt{noise\_covariance} := \mathbb{E} \bigg (
        \begin{bmatrix}
            v \\ w
        \end{bmatrix}
        \begin{bmatrix}
            v \\ w
        \end{bmatrix}^\mathrm{T}
        \bigg )

    [1] Verhaegen, Michel, and Vincent Verdult. *Filtering and system identification: a least squares approach.*
    Cambridge university press, 2007.
    """
    output_label = 'output'
    """ Label given to an output column in ``self.to_dataframe``. """
    standard_deviation_label = 'standard deviation'
    """ Label given to a standard deviation column in ``self.to_dataframe``. """
    actual_label = 'actual'
    """ Label given to a column in ``self.to_dataframe``, indicating measured values. """
    filtered_label = 'filtered'
    """ Label given to a column in ``self.to_dataframe``, indicating the filtered state of the Kalman filter. """
    next_predicted_label = 'next predicted (no input)'
    """
    Label given to a column in ``self.to_dataframe``, indicating the predicted state of the Kalman filter under the
    absence of further inputs.
    """
    next_predicted_corrected_label = 'next predicted (input corrected)'
    """
    Label given to a column in ``self.to_dataframe``, indicating the predicted state of the Kalman filter corrected
    by previous inputs. The inputs to the state-space model are known, but not at the time that the prediction was
    made. In order to make a fair comparison for prediction performance, the direct effect of the input on the output
    by the matrix :math:`D` is removed in this column.
    
    The latest prediction will have ``np.nan`` in this column, since the input is not yet known.
    """

    def __init__(
            self,
            state_space: StateSpace,
            noise_covariance: jnp.ndarray
    ):
        self.state_space = state_space

        validate_matrix_shape(
            noise_covariance,
            (self.state_space.y_dim + self.state_space.x_dim,
             self.state_space.y_dim + self.state_space.x_dim),
            'noise_covariance')
        self.r = noise_covariance[:self.state_space.y_dim, :self.state_space.y_dim]
        self.s = noise_covariance[self.state_space.y_dim:, :self.state_space.y_dim]
        self.q = noise_covariance[self.state_space.y_dim:, self.state_space.y_dim:]

        self.x_filtereds = []
        self.x_predicteds = []
        self.p_filtereds = []
        self.p_predicteds = []
        self.us = []
        self.ys = []
        self.y_filtereds = []
        self.y_predicteds = []
        self.kalman_gains = []

    def step(
            self,
            y: Optional[jnp.ndarray],
            u: jnp.ndarray
    ):
        """
        Given an observed input ``u`` and output ``y``, update the filtered and predicted states of the Kalman filter.
        Follows the implementation of the conventional Kalman filter in [1] on page 140.

        The output ``y`` can be missing by setting ``y=None``.
        In that case, the Kalman filter will obtain the next internal state by stepping the state space model.

        [1] Verhaegen, Michel, and Vincent Verdult. *Filtering and system identification: a least squares approach.*
        Cambridge university press, 2007.
        """
        if y is not None:
            validate_matrix_shape(y, (self.state_space.y_dim, 1), 'y')
        validate_matrix_shape(u, (self.state_space.u_dim, 1), 'u')

        x_pred = self.x_predicteds[-1] if self.x_predicteds else jnp.zeros((self.state_space.x_dim, 1))
        p_pred = self.p_predicteds[-1] if self.p_predicteds else jnp.eye(self.state_space.x_dim)

        k_filtered = p_pred @ self.state_space.c.T @ jnp.linalg.pinv(
            self.r + self.state_space.c @ p_pred @ self.state_space.c.T
        )

        self.p_filtereds.append(
            p_pred - k_filtered @ self.state_space.c @ p_pred
        )

        self.x_filtereds.append(
            x_pred + k_filtered @ (y - self.state_space.d @ u - self.state_space.c @ x_pred)
            if y is not None else x_pred
        )

        k_pred = (self.s + self.state_space.a @ p_pred @ self.state_space.c.T) @ jnp.linalg.pinv(
            self.r + self.state_space.c @ p_pred @ self.state_space.c.T
        )

        self.p_predicteds.append(
            self.state_space.a @ p_pred @ self.state_space.a.T
            + self.q
            - k_pred @ (self.s + self.state_space.a @ p_pred @ self.state_space.c.T).T
        )

        x_predicted = self.state_space.a @ x_pred + self.state_space.b @ u
        if y is not None:
            x_predicted += k_pred @ (y - self.state_space.d @ u - self.state_space.c @ x_pred)
        self.x_predicteds.append(
            x_predicted
        )

        self.us.append(u)
        self.ys.append(y if y is not None else jnp.full((self.state_space.y_dim, 1), jnp.nan))
        self.y_filtereds.append(self.state_space.output(self.x_filtereds[-1], self.us[-1]))
        self.y_predicteds.append(self.state_space.output(self.x_predicteds[-1]))
        self.kalman_gains.append(k_pred)

        return self.y_filtereds[-1], self.y_predicteds[-1]

    def extrapolate(
            self,
            timesteps
    ) -> pd.DataFrame:
        """
        Make a ``timesteps`` number of steps ahead prediction about the output of the state-space model
        ``self.state_space`` given no further inputs.
        The result is a ``pd.DataFrame`` where the columns are ``self.state_space.y_column_names``:
        the output columns of the state-space model ``self.state_space``.
        """
        if not self.x_predicteds:
            raise Exception('Prediction is only possible once Kalman estimation has been performed.')

        state_space = StateSpace(
            self.state_space.a,
            self.state_space.b,
            self.state_space.c,
            self.state_space.d,
            x_init=self.x_predicteds[-1],
            y_column_names=self.state_space.y_column_names,
            u_column_names=self.state_space.u_column_names
        )

        for _ in range(timesteps):
            state_space.step()

        return state_space.to_dataframe()[state_space.y_column_names]

    def _measurement_and_state_standard_deviation(
            self,
            state_covariance_matrices: List[jnp.ndarray]
    ) -> List[jnp.ndarray]:
        """
        Calculates the expected standard deviations on the output, assuming independence (!) in the noise of the state
        estimate and the process noise.
        Returns a list of row-vectors containing the standard deviations for the outputs.
        """
        covars_process_y = [
            self.state_space.c @ p @ self.state_space.c.T for p in state_covariance_matrices
        ]

        var_process_ys = [
            jnp.maximum(
                jnp.diagonal(p), 0
            )
            for p in covars_process_y
        ]
        var_measurement_y = jnp.maximum(jnp.diagonal(self.r), 0)

        return [
            jnp.sqrt(
                var_process_y + var_measurement_y
            ).reshape(
                (self.state_space.y_dim, 1)
            )
            for var_process_y in var_process_ys
        ]

    @staticmethod
    def _list_of_states_to_array(
            list_of_states: List[jnp.ndarray]
    ) -> jnp.ndarray:
        return jnp.array(list_of_states).squeeze(axis=2)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns the output of the Kalman filter as a ``pd.DataFrame``. The returned value contains information about
        filtered and predicted states of the Kalman filter at different timesteps.
        The expected standard deviation of the output is given, assuming independence (!) of the state estimation error
        and measurement noise.

        The rows of the returned dataframe correspond to timesteps.
        The columns of the returned dataframe are a 3-dimensional multi-index with the following levels:

        1. The output name, in the list ``self.state_space.y_column_names``.
        2. An indication of whether the value is
            - a value that was actually measured, these values were given to `self.step` as the `y` parameter,
            - a filtered state,
            - a predicted state given no further input or
            - a predicted state where the effect of the next input has been corrected for.
              This column is useful for comparing prediction performance.
        3. Whether the column is a value or the corresponding expected standard deviation.
        """
        input_corrected_predictions = [
            output + self.state_space.d @ input_state
            for input_state, output
            in zip(self.us[1:], self.y_predicteds[:-1])
        ] + [jnp.empty((self.state_space.y_dim, 1)) * jnp.nan]

        output_frames = [
            pd.DataFrame({
                (self.actual_label, self.output_label): outputs,
                (self.filtered_label, self.output_label): filtereds,
                (self.filtered_label, self.standard_deviation_label): filtered_stds,
                (self.next_predicted_label, self.output_label): predicteds,
                (self.next_predicted_label, self.standard_deviation_label): predicted_stds,
                (self.next_predicted_corrected_label, self.output_label): input_corrected_prediction,
                (self.next_predicted_corrected_label, self.standard_deviation_label): predicted_stds
            }).applymap(lambda array: array[0])
            for (
                outputs,
                filtereds,
                predicteds,
                filtered_stds,
                predicted_stds,
                input_corrected_prediction
            ) in zip(
                zip(*self.ys),
                zip(*self.y_filtereds),
                zip(*self.y_predicteds),
                zip(*self._measurement_and_state_standard_deviation(self.p_filtereds)),
                zip(*self._measurement_and_state_standard_deviation(self.p_predicteds)),
                zip(*input_corrected_predictions)
            )
        ]
        return pd.concat(output_frames, axis=1, keys=self.state_space.y_column_names)