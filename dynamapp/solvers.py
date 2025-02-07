import jax
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

def solve_riccati_equation(A, B, Q, R):
    """
    Solve the discrete-time Algebraic Riccati Equation (ARE):
    
    .. math::
        P = A^T . P . A - (A^T . P . B) . (R + B.T .P . B)^{-1}.(B^T . P . A) + Q
    
    Args:
        - A: State matrix (n, n)
        - B: Control matrix (n, m)
        - Q: State cost matrix (n, n)
        - R: Control cost matrix (m, m)

    Returns:
        - P: Solution to the Riccati equation (n, n)
    
    .. note::
        This function do not work with @jit decorator 
    """
    n = A.shape[0]
    P = jnp.eye(n)  
    
    def step(P):
        P_new = A.T@P@A-(A.T@P@B)@ jnp.linalg.inv(R+B.T@P@B)@(B.T@P@A)+Q
        diff = jnp.abs(P_new - P)
        max_diff = jnp.max(diff)
        converged = max_diff < 1e-6
        P_next = jax.lax.cond(converged, lambda _: P_new, lambda _: P, None)
        return P_next, converged

    P_init, converged = step(P)
    while(converged):
        P, converged = step(P_init)
        P_init = P
    return P

@jit
def luenberger_observer(A, B, C, desired_poles):
    """
    Computes the Luenberger Observer gain matrix L by placing 
    the poles of the observer at the desired_poles.
    
    Args:
        - A (ndarray): System matrix.
        - B (ndarray): Input matrix.
        - C (ndarray): Output matrix.
        - desired_poles (list): Desired poles for the observer.
        
    Returns:
        L (ndarray): Observer gain matrix.
        
    .. todo::
        we need to solve the pole placement problem using a custom algorithm 
        the acctual version return a null gain matrix, 
    """
    A = jnp.array(A)
    B = jnp.array(B)
    C = jnp.array(C)
    n = A.shape[0]
    Q = B
    for i in range(1, n):
        Q = jnp.hstack((Q, jnp.linalg.matrix_power(A, i) @ B))
    desired_poles_matrix = jnp.poly(jnp.array(desired_poles))
    L = jnp.zeros((n, n)) 
    return L

