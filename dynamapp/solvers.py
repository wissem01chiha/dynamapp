from jax import jit 
import jax.numpy as jnp

@jit
def solve_least_square(W: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """ 
    Solves the least squares problem: WX = Y for X.
    
    Args:
        - W: (m, n) matrix (design matrix).
        - Y: (m, k) matrix (target values).
    Returns:
        - X: (n, k) matrix (solution to WX = Y).
    """
    X = jnp.linalg.pinv(W) @ Y  
    return X

@jit
def solve_riccati_equation(A, B, Q, R):
    """
    Solve the discrete-time Algebraic Riccati Equation (ARE):
    
    .. math::
        P = A.T @ P @ A - (A.T @ P @ B) @ jnp.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
    
    Args:
        - A: State matrix (n, n)
        - B: Control matrix (n, m)
        - Q: State cost matrix (n, n)
        - R: Control cost matrix (m, m)

    Returns:
        - P: Solution to the Riccati equation (n, n)
    """
    assert A.shape[0] == A.shape[1], "Matrix A must be square."
    assert B.shape[0] == A.shape[0], "Matrix B must have the same number of rows as A."
    assert Q.shape[0] == Q.shape[1] == A.shape[0], "Matrix Q must be square and match the dimensions of A."
    assert R.shape[0] == R.shape[1] == B.shape[1], "Matrix R must be square and match the number of columns of B."
    
    n = A.shape[0]
    P = jnp.eye(n)  
    
    for _ in range(100): 
        P_new = A.T @ P @ A - (A.T @ P @ B) @ jnp.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
        if jnp.allclose(P_new, P, atol=1e-6):
            break
        P = P_new
    return P

@jit
def luenberger_observer(A, B, C, desired_poles):
    """
    Computes the Luenberger Observer gain matrix L by placing the poles of the observer at the desired_poles.
    
    Args:
        A (ndarray): System matrix.
        B (ndarray): Input matrix.
        C (ndarray): Output matrix.
        desired_poles (list): Desired poles for the observer.
        
    Returns:
        L (ndarray): Observer gain matrix.
    TODO :we need to solve the pole placement problem using a custom algorithm 
        the acctual version return a null gain matrix, 
    """
    A = jnp.array(A)
    B = jnp.array(B)
    C = jnp.array(C)
    
    assert A.shape[0] == A.shape[1], "Matrix A must be square."
    assert B.shape[0] == A.shape[0], "The number of rows in B must match the number of rows in A."
    assert C.shape[1] == A.shape[0], "The number of columns in C must match the number of rows in A."
    
    n = A.shape[0]
    Q = B
    for i in range(1, n):
        Q = jnp.hstack((Q, jnp.linalg.matrix_power(A, i) @ B))
    
    desired_poles_matrix = jnp.poly(desired_poles)  
    L = jnp.zeros((n, n)) 
    return L

