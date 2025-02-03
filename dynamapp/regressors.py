import jax
from jax import jit
import jax.numpy as jnp

from .model import Model
from .model_state import ModelState

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
        - W Jax tensor (m.ndof * m.ndof * 4 *  m.ndof)
    """
    @jit
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        m.dhparams = [tensor[:, i] for i in range(tensor.shape[1])]
        return m.generalized_torques(q,v,a)
    
    jm_dhparams_jax = [jnp.array(sublist) for sublist in m.dhparams]
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(jm_dhparams_jax,axis=-1))
    return regressor_jacobian

def generalized_torques_wrt_damping(m:Model,q:jnp.ndarray,
                                    v:jnp.ndarray,a:jnp.ndarray):
    """
    Computes the regressor tensor of the genrlized torques with respect to the 
    joints damping vector.
    NOTE: this function require that dampings are not None
    Returns:
        - jax tensor (m.ndof * m.ndof * m.nodf)
    """
    @jit
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        m.dampings = [tensor[i] for i in range(tensor.shape[0])]
        return m.generalized_torques(q,v,a)
    
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(m.dampings, axis=-1))
    return regressor_jacobian

def full_torques_wrt_inerial(m:Model,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray,
                            alpha:jnp.array, beta:jnp.array, gamma:jnp.array,
                            dampings:list):
    """ 
    Compute the regressor tensor of the full torques with respect to the
    links inertial paramters
    """
    @jit 
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return m.full_torques(alpha,beta,gamma,dampings,q,v,a)
    
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(m.Imats, axis=-1))
    return regressor_jacobian 

def full_torques_wrt_friction(m:Model,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray):
    """
    Compute the rgressor tensor of the full torques with respect to the links 
    friction coefficents
    Returns:
        - jax tensor (m.ndof * )
    """
    @jit
    def regressor(alpha, beta, gamma)-> jnp.ndarray:
        return m.full_torques(alpha,beta,gamma,m.dampings,q,v,a)
    
    regressor_jacobian  = jax.jacobian(regressor, (0, 1, 2))
    return regressor_jacobian 

def eigvals_wrt_inertia(ms:ModelState,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray):
    """
    Compute the eigvalues jacobian with respect to inertial paramters
    Returns:
        - jax tensor (m.ndof  * )
    """
    x = jnp.concatenate([q, v])
    @jit
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        ms.model.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return ms.compute_eigvals(x)
    
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(ms.model.Imats, axis=-1))
    return regressor_jacobian

def eigvals_wrt_dhparams(ms:ModelState,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray):
    """
    Compute the eigvalues jacobian wit respect to dhparams coefficents
    Returns:
        - jax tensor (m.ndof * )
    """
    x = jnp.concatenate([q, v])
    @jit
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        ms.model.dhparams = [tensor[:, i] for i in range(tensor.shape[1])]
        return ms.compute_eigvals(x)
    
    jm_dhparams_jax = [jnp.array(sublist) for sublist in ms.model.dhparams]
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(jm_dhparams_jax,axis=-1))
    return regressor_jacobian

def eigvals_wrt_damping(ms:ModelState,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray):
    """
    Compute the eigvalues jacobian with respect to joints damping coefficents
    Returns:
        - jax tensor ( m.ndof * )
    """
    x = jnp.concatenate([q, v])
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        ms.model.dampings = [tensor[:, i] for i in range(tensor.shape[1])]
        return ms.compute_eigvals(x)

    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(ms.model.dampings, axis=-1))
    return regressor_jacobian

def state_matrix_a_wrt_inertia(ms:ModelState,q:jnp.ndarray,v:jnp.ndarray,a:jnp.ndarray):
    """
    Compute the jacobian tensor of the state matrix A with respect to the links
    inertial coefficents 
    Returns:
        - jax tensor (m.ndof * )
    """
    x = jnp.concatenate([q, v])
    @jit
    def regressor(tensor:jnp.ndarray)-> jnp.ndarray:
        ms.model.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return ms.get_state_matrix_a(x)
         
    regressor_jacobian = jax.jacobian(regressor)(jnp.stack(ms.model.Imats, axis=-1))
    return regressor_jacobian
