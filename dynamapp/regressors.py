import logging
import jax
from jax import jit
import jax.numpy as jnp

from .model import Model
from .model_state import ModelState
from .trajectory import *

logger = logging.getLogger(__name__)

def generalized_torques_wrt_inertia(m:Model,q:jnp.ndarray,
                                v:jnp.ndarray,a:jnp.ndarray)->jnp.ndarray:
    """
    Computes the regressor tensor of the genrlized torques with respect to
    the inertia links vector evaluated at (q,v,a)
    
    Args:
        - q, v, a position , velcoty, accelation vectors 
    ..::math
        \tau = W(q,v,a) X
        where: X = [m, mz, my, mx, Izz, Iyz, Iyy, Ixz, Ixy, Ixx ]^{T} of each link (i)
    W(q,v,a) of size (ndof, 6*ndof)
    """
    W = jnp.zeros((m.ndof,6*m.ndof))
    inertia_vec = m.Imats
    
    return W

def generalized_torques_wrt_dhparams(m:Model,q:jnp.ndarray,
                                    v:jnp.ndarray,a:jnp.ndarray)->jnp.ndarray:
    """
    Computes the regressor tensor of the genrlized torques with respect to the 
    dh params 
    """
    W = 1
     
    return W 

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






