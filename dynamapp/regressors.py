import logging
import jax
from jax import jit
import jax.numpy as jnp

from .model import Model
from .model_state import ModelState
from .trajectory import *
from .viscoelastic import *

logger = logging.getLogger(__name__)

def generalized_torques_wrt_inertia (m:Model, q:jnp.ndarray,
                                v:jnp.ndarray,a:jnp.ndarray)->jnp.ndarray:
    """
    Computes the regressor tensor of the generalized torques with respect to
    the inertia links vector evaluated at (q, v, a)
    
    Args:
        - m: Model instance containing necessary methods and parameters
        - q: Position vector
        - v: Velocity vector
        - a: Acceleration vector
    
    Returns:
        - W: Regressor tensor (m.ndof, m.ndof, m.ndof, 6, 6)
    """   
    @jit
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]  
        return m.generalized_torques(q, v, a)
    
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(m.Imats, axis=-1))
    
    return regressor_jacobian

def inertia_tensor_wrt_inertia(m:Model, q:jnp.ndarray)->jnp.ndarray:
    """
    Compute the regressor of the inertia matrix with respect to inertia 
    tensor values, derive the mass matrix wrt links inertial compennats
    
    Returns:
        - W Jax tensor (m.ndof, m.ndof, 6, 6, m.ndof, 6)
    """
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])] 
        return m.inertia_tensor(q)
    
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(m.Imats, axis=-1))
    return regressor_jacobian

def generalized_torques_wrt_dhparams(m:Model,q:jnp.ndarray,
                                    v:jnp.ndarray,a:jnp.ndarray)->jnp.ndarray:
    """
    Computes the regressor tensor of the generlized torques with respect to the 
    dh-params 
    
    Returns:
        - W Jax tensor (m.ndof, )
    """
    @jit
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        m.dhparams = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return m.generalized_torques(q,v,a)
    
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(m.dhparams, axis=-1))
    return regressor_jacobian

def generalized_torques_wrt_damping(m:Model,q:jnp.ndarray,
                                    v:jnp.ndarray,a:jnp.ndarray):
    """
    Computes the regressor tensor of the genrlized torques with respect to the 
    joints damping vector 
    
    ..::math:
        \tau = W X_d
    """
    W = 1
    return W

def full_torques_wrt_inerial(m:Model):
    """ 
    
    """
    return 

def eigvals_wrt_inertia(ms:ModelState,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray):
    """
    """
    W = 1
   
    return W 

def eigvals_wrt_dhparams(ms:ModelState,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray):
    """
    """
    W = 1 
  
    return W 

def eigvals_wrt_damping(ms:ModelState,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray):
    """
    """
    W =1
 
    return W 






