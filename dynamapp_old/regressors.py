import jax
import jax.numpy as jnp

from .model import Model
from .model_state import ModelState

def generalized_torques_wrt_inertia(m: Model, q: jnp.ndarray, 
                                v: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Computes the regressor tensor of generalized torques with respect to inertia."""
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return m.generalized_torques(q, v, a)
    
    return jax.jacobian(regressor)(jnp.stack(m.Imats, axis=-1))

def inertia_tensor_wrt_inertia(m: Model, q: jnp.ndarray) -> jnp.ndarray:
    """Computes the regressor of the inertia matrix with respect to inertia tensor values."""
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return m.inertia_tensor(q)

    return jax.jacobian(regressor)(jnp.stack(m.Imats, axis=-1))

def generalized_torques_wrt_dhparams(m: Model, q: jnp.ndarray, 
                                    v: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Computes the regressor tensor of generalized torques with respect to DH parameters."""
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        m.dhparams = [tensor[:, i] for i in range(tensor.shape[1])]
        return m.generalized_torques(q, v, a)

    return jax.jacobian(regressor)(jnp.stack([jnp.array(sublist) for sublist in m.dhparams],
                                            axis=-1))

def generalized_torques_wrt_damping(m: Model, q: jnp.ndarray,
                                v: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Computes the regressor tensor of generalized torques with respect to joint damping."""
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        m.dampings = list(tensor)
        return m.generalized_torques(q, v, a)

    return jax.jacobian(regressor)(jnp.stack(m.dampings, axis=-1))

def full_torques_wrt_inertia(m: Model, q: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray,
                             alpha: jnp.ndarray, beta: jnp.ndarray, gamma: jnp.ndarray,
                             dampings: list) -> jnp.ndarray:
    """Computes the regressor tensor of full torques with respect to inertia parameters."""
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return m.full_torques(alpha, beta, gamma, dampings, q, v, a)

    return jax.jacobian(regressor)(jnp.stack(m.Imats, axis=-1))

def full_torques_wrt_friction(m: Model, q: jnp.ndarray, 
                            v: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Computes the regressor tensor of full torques with respect to friction coefficients."""
    def regressor(alpha, beta, gamma) -> jnp.ndarray:
        return m.full_torques(alpha, beta, gamma, m.dampings, q, v, a)

    return jax.jacobian(regressor, (0, 1, 2))

def eigvals_wrt_inertia(ms: ModelState, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Computes the Jacobian of eigenvalues with respect to inertia parameters."""
    x = jnp.concatenate([q, v])
    
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        ms.model.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return ms.compute_eigvals(x)

    return jax.jacobian(regressor)(jnp.stack(ms.model.Imats, axis=-1))

def eigvals_wrt_dhparams(ms: ModelState, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Computes the Jacobian of eigenvalues with respect to DH parameters."""
    x = jnp.concatenate([q, v])
    
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        ms.model.dhparams = [tensor[:, i] for i in range(tensor.shape[1])]
        return ms.compute_eigvals(x)

    return jax.jacobian(regressor)(jnp.stack([jnp.array(sublist) for sublist
                                            in ms.model.dhparams], axis=-1))

def eigvals_wrt_damping(ms: ModelState, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Computes the Jacobian of eigenvalues with respect to damping coefficients."""
    x = jnp.concatenate([q, v])
    
    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        ms.model.dampings = list(tensor)
        return ms.compute_eigvals(x)

    return jax.jacobian(regressor)(jnp.stack(ms.model.dampings, axis=-1))

def state_matrix_a_wrt_inertia(ms: ModelState, q: jnp.ndarray, v: jnp.ndarray)->jnp.ndarray:
    """Computes the Jacobian of state matrix A with respect to inertia coefficients."""
    x = jnp.concatenate([q, v])

    def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
        ms.model.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
        return ms.get_state_matrix_a(x)

    return jax.jacobian(regressor)(jnp.stack(ms.model.Imats, axis=-1))

def state_matrix_a_wrt_state(ms: ModelState, q: jnp.ndarray, v: jnp.ndarray)->jnp.ndarray:
    """Computes the Jacobian of state matrix A with respect to state variables (q, v)."""
    x = jnp.concatenate([q, v])

    def regressor(tensor):
        return ms.get_state_matrix_a(tensor)

    return jax.jacobian(regressor)(x)
