import numpy as np 
from jax import jit 
from scipy.linalg import solve_discrete_are
from scipy.signal import place_poles

def solve_riccati_equation(A, B, Q, R):
    """
    Solve the discrete time algebric riccati equation given by :
    ..::math:
    
    Args: 
        - A, B  : System Matrices 
    Returns:
        - P ARE solution. 
    """
    assert A.shape[0] == A.shape[1],"Matrix A must be square."
    assert B.shape[0] == A.shape[0],"Matrix B must have the same number of rows as A."
    assert Q.shape[0] == Q.shape[1] == A.shape[0],"Matrix Q must be square and match the dimensions of A."
    assert R.shape[0] == R.shape[1] == B.shape[1],"Matrix R must be square and match the number of columns of B."
    
    P  =  solve_discrete_are(A, B, R, Q)
    return P

def solve_discrete_state_depend_are(A, B, Q, R):
    """ 
    Solve the discrete state depend Riccati equation given by:
    .. math::
        R^{T}A = Q

    Args:
        - A, B, Q, R
    Ref:
    
    """
    P =1
    return P

def luenberger_observer(A, B, C, desired_poles):
    """
    Computes the Luenberger Observer gain matrix L.
    The gain matrix L is computed by placing the poles of the observer at the desired_poles.
    it should be noted that the observer is given by the following equation:
    .. math::
        \dot{\hat{x}} = A\hat{x} + Bu + L(y - C\hat{x})
    where:
    - A is the system matrix.
    - B is the input matrix.
    - C is the output matrix.
    - L is the observer gain matrix.
    Args::
        A (ndarray): System matrix.
        B (ndarray): Input matrix.
        C (ndarray): Output matrix.
        desired_poles (list): desired poles for the observer.

    Returns:
        L (ndarray): Observer gain matrix.
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    assert A.shape[0] == A.shape[1], "Matrix A must be square."
    assert B.shape[0] == A.shape[0], "The number of rows in B must match the number of rows in A."
    assert C.shape[1] == A.shape[0], "The number of columns in C must match the number of rows in A."
    At = A.T
    Ct = C.T
    placed_poles = place_poles(At, Ct, desired_poles)
    Lt = placed_poles.gain_matrix
    L = Lt.T
    
    return L

