from typing import List
import pandas as pd
import jax.numpy as jnp

from .math_utils import validate_matrix_shape

class StateSpace:
    r"""
    A state-space model defined by the following equations:

    .. math::
        \begin{cases}
            x_{k+1} &= A x_k + B u_k + K e_k \\
            y_k &= C x_k + D u_k + e_k
        \end{cases}

    The shapes of the matrices are checked for consistency and will raise if inconsistent.
    If a matrix does not exist in the model representation, the corresponding ``jnp.ndarray`` should have dimension
    zero along that axis. See the example below.

    Example
    -------
    An autonomous state-space model has no matrices :math:`B` and :math:`D`.
    An autonomous model with a one-dimensional internal state and output, can be represented as follows:

    >>> model = StateSpace(
    >>>     jnp.ones((1, 1)),
    >>>     jnp.ones((1, 0)),
    >>>     jnp.ones((1, 1)),
    >>>     jnp.ones((1, 0))
    >>> )

    :param a: matrix :math:`A`
    :param b: matrix :math:`B`
    :param c: matrix :math:`C`
    :param d: matrix :math:`D`
    :param k: matrix :math:`K`, optional
    :param x_init: initial state :math:`x_0` of the model, optional
    :param y_column_names: list of output column names, optional
    :param u_column_names: list of input column names, optional
    """
    def __init__(
            self,
            a: jnp.ndarray,
            b: jnp.ndarray,
            c: jnp.ndarray,
            d: jnp.ndarray,
            k: jnp.ndarray = None,
            x_init: jnp.ndarray = None,
            y_column_names: List[str] = None,
            u_column_names: List[str] = None
    ):
        self._set_dimensions(a, b, c)
        self.set_x_init(x_init)
        self._set_column_names(u_column_names, y_column_names)
        self.set_matrices(a, b, c, d, k)

    def _set_dimensions(
            self,
            a: jnp.ndarray,
            b: jnp.ndarray,
            c: jnp.ndarray
    ):
        """ Determine the dimensions of the internal states, outputs and inputs, based on the matrix shapes. """
        self.x_dim = a.shape[0]
        self.y_dim = c.shape[0]
        self.u_dim = b.shape[1]

        if self.y_dim == 0:
            raise ValueError('The dimension of the output should be at least 1')

    def set_matrices(
            self,
            a: jnp.ndarray,
            b: jnp.ndarray,
            c: jnp.ndarray,
            d: jnp.ndarray,
            k: jnp.ndarray = None
    ):
        """ Validate if the shapes make sense and set the system matrices. """
        if k is None:
            k = jnp.zeros((self.x_dim, self.y_dim))
        validate_matrix_shape(a, (self.x_dim, self.x_dim), 'a')
        validate_matrix_shape(b, (self.x_dim, self.u_dim), 'b')
        validate_matrix_shape(c, (self.y_dim, self.x_dim), 'c')
        validate_matrix_shape(d, (self.y_dim, self.u_dim), 'd')
        validate_matrix_shape(k, (self.x_dim, self.y_dim), 'k')
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k
        self.xs = []
        self.ys = []
        self.us = []

    def _set_column_names(
            self,
            u_column_names: List[str],
            y_column_names: List[str]
    ):
        """ Set the column names of the input and output. """
        if y_column_names is None:
            y_column_names = [f'$y_{i}$' for i in range(self.y_dim)]
        if u_column_names is None:
            u_column_names = [f'$u_{i}$' for i in range(self.u_dim)]
        if len(y_column_names) != self.y_dim:
            raise ValueError(f'Length of `y_column_names` should be {self.y_dim}, not {len(y_column_names)}')
        if len(u_column_names) != self.u_dim:
            raise ValueError(f'Length of `u_column_names` should be {self.u_dim}, not {len(u_column_names)}')
        self.y_column_names = y_column_names
        self.u_column_names = u_column_names

    def set_x_init(self, x_init: jnp.ndarray):
        """ Set the initial state, if it is given. """
        if x_init is None:
            x_init = jnp.zeros((self.x_dim, 1))
        validate_matrix_shape(x_init, (self.x_dim, 1), 'x_dim')
        self._x_init = x_init

    def step(
            self,
            u: jnp.ndarray = None,
            e: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Calculates the output of the state-space model and returns it.
        Updates the internal state of the model as well.
        The input ``u`` is optional, as is the noise ``e``.
        """
        if u is None:
            u = jnp.zeros((self.u_dim, 1))
        if e is None:
            e = jnp.zeros((self.y_dim, 1))
            
        validate_matrix_shape(u, (self.u_dim, 1), 'u')
        validate_matrix_shape(e, (self.y_dim, 1), 'e')

        x = self.xs[-1] if self.xs else self._x_init
        x, y = (
            self.a @ x + self.b @ u + self.k @ e,
            self.output(x, u, e)
        )
        self.us.append(u)
        self.xs.append(x)
        self.ys.append(y)
        return y

    def output(
            self,
            x: jnp.ndarray,
            u: jnp.ndarray = None,
            e: jnp.ndarray = None):
        """
        Calculate the output of the state-space model.
        This function calculates the updated :math:`y_k` of the state-space model in the class description.
        The current state ``x`` is required.
        Providing an input ``u`` is optional.
        Providing a noise term ``e`` to be added is optional as well.
        """
        if u is None:
            u = jnp.zeros((self.u_dim, 1))
        if e is None:
            e = jnp.zeros((self.y_dim, 1))

        validate_matrix_shape(x, (self.x_dim, 1), 'x')
        validate_matrix_shape(u, (self.u_dim, 1), 'u')
        validate_matrix_shape(e, (self.y_dim, 1), 'e')

        return self.c @ x + self.d @ u + e

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the inputs and outputs of the state-space model as a dataframe, where the columns are the input-
        and output-columns.
        """
        inputs_df = pd.DataFrame(jnp.array(self.us).squeeze(axis=2), columns=self.u_column_names)
        outputs_df = pd.DataFrame(jnp.array(self.ys).squeeze(axis=2), columns=self.y_column_names)
        return pd.concat([inputs_df, outputs_df], axis=1)

