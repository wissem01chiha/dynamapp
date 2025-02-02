from jax import jit
import jax.numpy as jnp

@jit
def compute_coulomb_friction_force(v: jnp.ndarray, fc: jnp.ndarray, 
                        fs: jnp.ndarray)->jnp.ndarray:
    """
    Compute the Coulomb  friction model.

    Args:
        - v  (jnp.ndarray): Velocity array.
        - fc (jnp.ndarray): Coulomb friction coefficient.
        - fs (jnp.ndarray): Viscous friction coefficient.

    Returns:
       - jax - array: friction force array.
    """
    return fc * jnp.sign(v) + fs * v

@jit
def compute_friction_force(alpha:jnp.array, beta:jnp.array, gamma:jnp.array,
                        q, v, a):
    """
    Computes the frictional force for a joint using a simple 
    generalized friction model.
    
    Args:
        - alpha (jax-array): Coefficients for the position-related terms.
        - beta (jax-array): Coefficients for the velocity-related terms.
        - gamma (jax-array): Coefficients for the acceleration-related terms.
        - q (float): The position of the joint.
        - v (float): The velocity of the joint.
        - a (float): The acceleration of the joint.
    
    Returns:
        - float: The computed frictional force.
    """
    pos = jnp.polyval(alpha,q)  
    vel = jnp.polyval(beta,v) 
    acc = jnp.polyval(gamma,a)  
    
    return pos + vel + acc