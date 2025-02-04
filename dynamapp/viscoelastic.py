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
       - jax-array: friction force array.
    """
    return fc * jnp.sign(v) + fs * v

@jit
def compute_friction_force(alpha:jnp.array, beta:jnp.array, gamma:jnp.array,
                        q, v, a):
    """
    Computes the frictional force for a joint using a simple 
    generalized friction model.
    
    .. math::
        \begin{cases} 
            f_i = \alpha_{1i} q_i + \alpha_{2i} q_i^2 + \ldots + \alpha_{ni} q_i^n + 
            \beta_{1i} v_i + \beta_{2i} v_i^2 + \ldots + \beta_{ki} v_i^k + 
            \gamma_{1i} a_i + \gamma_{2i} a_i^2 + \ldots + \gamma_{pi} a_i^p
        \end{cases}

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