import logging 
import jax.numpy as jnp
from collections import namedtuple
from typing import Tuple

logger = logging.getLogger(__name__)

"""
Eigenvalue decomposition of a matrix ``matrix`` such that ``left_orthogonal @ eigenvalues @ right_orthogonal``
equals ``matrix``.
"""
Decomposition = namedtuple('Decomposition', ['left_orthogonal', 'eigenvalues', 'right_orthogonal'])

def condition_number(M, threshold=1e-5):
    """
    Computes the condition number of a matrix with a check for SVD convergence.
    The condition number of a matrix is given by the equation:
    
    .. math::
        \kappa = \sigma_{max} / \sigma_{min}
    
    - :math:`\sigma_{max}` is the maximum singular value.
    - :math:`\sigma_{min}` is the minimum singular value.
    
    Args:
        M (np.ndarray): Input matrix.
        threshold (float): Threshold for the condition number.
    """
    try:
        cond_number = jnp.linalg.cond(M)
        return cond_number < threshold
    except jnp.linalg.LinAlgError as e:
        logger.error(f"Linear Algebra error: {e}")
        return False

def validate_matrix_shape(
        matrix: jnp.ndarray,
        shape: Tuple[float, float],
        name: str
):
    """
    Raises if ``matrix`` does not have shape ``shape``. The error message will contain ``name``.
    """
    if matrix.shape != shape:
        raise ValueError(f'Dimensions of `{name}` {matrix.shape} are inconsistent. Expected {shape}.')

def eigenvalue_decomposition(
        matrix: jnp.ndarray
) -> Decomposition:
    """
    Calculate eigenvalue decomposition of ``matrix`` as a ``Decomposition``.
    """
    u, eigenvalues, vh = jnp.linalg.svd(matrix)
    eigenvalues_mat = jnp.zeros((u.shape[0], vh.shape[0]))
    eigenvalues_mat = eigenvalues_mat.at[jnp.diag_indices(u.shape[0])].set(eigenvalues)
    return Decomposition(u, eigenvalues_mat, vh)

def reduce_decomposition(
        decomposition: Decomposition,
        rank: int
) -> Decomposition:
    """
    Reduce an eigenvalue decomposition ``decomposition`` such that only ``rank`` number of biggest eigenvalues
    remain. Returns another ``Decomposition``.
    """
    u, s, vh = decomposition
    return Decomposition(
        u[:, :rank],
        s[:rank, :rank],
        vh[:rank, :]
    )

def block_hankel_matrix(
        matrix: jnp.ndarray,
        num_block_rows: int
) -> jnp.ndarray:
    """
    Calculate a block Hankel matrix based on input matrix ``matrix`` with ``num_block_rows`` block rows.
    The shape of ``matrix`` is interpreted in row-order, like the structure of a ``pd.DataFrame``:
    the rows are measurements and the columns are data sources.

    The returned block Hankel matrix has a columnar structure. Every column of the returned matrix consists
    of ``num_block_rows`` block rows (measurements). See the examples for details.

    Examples
    --------
    Suppose that the input matrix contains 4 measurements of 2-dimensional data:

    >>> matrix = np.array([
    >>>     [0, 1],
    >>>     [2, 3],
    >>>     [4, 5],
    >>>     [6, 7]
    >>> ])

    If the number of block rows is set to ``num_block_rows=2``, then the block Hankel matrix will be

    >>> np.array([
    >>>     [0, 2, 4],
    >>>     [1, 3, 5],
    >>>     [2, 4, 6],
    >>>     [3, 5, 7]
    >>> ])
    """
    hankel_rows_dim = num_block_rows * matrix.shape[1]
    hankel_cols_dim = matrix.shape[0] - num_block_rows + 1

    hankel = jnp.zeros((hankel_rows_dim, hankel_cols_dim))
    for block_row_index in range(hankel_cols_dim):
        flattened_block_rows = matrix[block_row_index:block_row_index+num_block_rows,
                                        :].flatten()
        hankel =  hankel.at[:, block_row_index].set(flattened_block_rows)
    return hankel

def vectorize(
        matrix: jnp.ndarray
) -> jnp.ndarray:
    """
    Given a matrix ``matrix`` of shape ``(a, b)``, return a vector of shape ``(a*b, 1)`` with all columns of
    ``matrix`` stacked on top of eachother.
    """
    return jnp.reshape(matrix.flatten(order='F'), (matrix.shape[0] * matrix.shape[1], 1))

def unvectorize(
        vector: jnp.ndarray,
        num_rows: int
) -> jnp.ndarray:
    """
    Given a vector ``vector`` of shape ``(num_rows*b, 1)``, return a matrix of shape ``(num_rows, b)`` such that
    the stacked columns of the returned matrix equal ``vector``.
    """
    if vector.shape[0] % num_rows != 0 or vector.shape[1] != 1:
        raise ValueError(f'Vector shape {vector.shape} and `num_rows`={num_rows} are incompatible')
    return vector.reshape((num_rows, vector.shape[0] // num_rows), order='F')

def is_skew_symmetric(matrix):
    """Check if the input matrix is skew-symmetric."""
    matrix_transpose = jnp.transpose(matrix)
    status = jnp.array_equal(matrix, -matrix_transpose)
    return status
