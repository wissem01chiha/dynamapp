import jax
import jax.numpy as jnp

from .model import Model
from .model_state import ModelState

class ModelJacobian():
    """
    A base class for computing the Jacobians of joint quantities 
    (torques, forces, etc.) with respect to specified model parameters.

    This class provides methods to compute the derivatives of generalized 
    torques and inertia tensors concerning inertia parameters, DH parameters, 
    and joint damping coefficients.

    Attributes:
        - m (Model): An instance of the Model class initialized with inertia 
                   matrices, DH parameters, gravity, and damping coefficients.
    """
    def __init__(self,Imats:list, 
                dhparams:list, 
                gravity = -9.81, 
                dampings:list = None):
        
        self.m = Model(Imats,dhparams,gravity,dampings)
        
    def generalized_torques_wrt_inertia(self, q: jnp.ndarray, 
                                    v: jnp.ndarray, a: jnp.ndarray)->jnp.ndarray:
        """
        Computes the regressor tensor of generalized torques with respect to inertia.
        """
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
            return self.m.generalized_torques(q, v, a)
        
        return jax.jacobian(regressor)(jnp.stack(self.m.Imats, axis=-1))
    
    def inertia_tensor_wrt_inertia(self, q: jnp.ndarray)->jnp.ndarray:
        """
        Computes the regressor of the inertia matrix with respect to inertia tensor values.
        """
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
            return self.m.inertia_tensor(q)
        
        return jax.jacobian(regressor)(jnp.stack(self.m.Imats, axis=-1))
    
    def generalized_torques_wrt_dhparams(self, q: jnp.ndarray, 
                                    v: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the regressor tensor of generalized torques with respect to DH-parameters.
        """
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.m.dhparams = [tensor[:, i] for i in range(tensor.shape[1])]
            return self.m.generalized_torques(q, v, a)

        return jax.jacobian(regressor)(jnp.stack([jnp.array(sublist) for sublist in self.m.dhparams],
                                            axis=-1))
    
    def generalized_torques_wrt_damping(self, q: jnp.ndarray,
                                v: jnp.ndarray, a: jnp.ndarray)->jnp.ndarray:
        """
        Computes the regressor tensor of generalized torques with respect to joint damping.
        """
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.m.dampings = list(tensor)
            return self.m.generalized_torques(q, v, a)

        return jax.jacobian(regressor)(jnp.stack(self.m.dampings, axis=-1))

    def full_torques_wrt_inertia(self, q: jnp.ndarray, v: jnp.ndarray, a: jnp.ndarray,
                             alpha: jnp.ndarray, beta: jnp.ndarray, gamma: jnp.ndarray,
                             dampings: list) -> jnp.ndarray:
        """
        Computes the regressor tensor of full torques with respect to inertia parameters.
        """
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.m.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
            return self.m.full_torques(alpha, beta, gamma, dampings, q, v, a)

        return jax.jacobian(regressor)(jnp.stack(self.m.Imats, axis=-1))

    def full_torques_wrt_friction(self, q: jnp.ndarray, 
                            v: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the regressor tensor of full torques with respect to friction coefficients.
        """
        def regressor(alpha, beta, gamma) -> jnp.ndarray:
            return self.m.full_torques(alpha, beta, gamma, self.m.dampings, q, v, a)

        return jax.jacobian(regressor, (0, 1, 2))

class ModelStateJacobian():
    """
    A base class for computing Jacobians of eigenvalues and state matrices 
    with respect to model parameters.

    This class provides methods to compute the derivatives of eigenvalues 
    and the state matrix A concerning inertia parameters, DH parameters, 
    damping coefficients, and state variables.

    Attributes:
        ms (ModelState): An instance of the ModelState class initialized 
                         with inertia matrices, DH parameters, gravity, 
                         and damping coefficients.
    """
    def __init__(self,Imats:list, 
                dhparams:list, 
                gravity = -9.81, 
                dampings:list = None, 
                x_init:jnp.ndarray = None ):
        
        self.ms = ModelState(Imats,dhparams,gravity,dampings)
    
    def eigvals_wrt_inertia(self,q: jnp.ndarray, v: jnp.ndarray)->jnp.ndarray:
        """
        Computes the Jacobian of eigenvalues with respect to inertia parameters.
        """
        x = jnp.concatenate([q, v])
    
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.ms.model.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
            return self.ms.compute_eigvals(x)

        return jax.jacobian(regressor)(jnp.stack(self.ms.ms.model.Imats, axis=-1))

    def eigvals_wrt_dhparams(self, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Jacobian of eigenvalues with respect to DH parameters.
        """
        x = jnp.concatenate([q, v])
        
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.ms.model.dhparams = [tensor[:, i] for i in range(tensor.shape[1])]
            return self.ms.compute_eigvals(x)

        return jax.jacobian(regressor)(jnp.stack([jnp.array(sublist) for sublist
                                                in self.ms.model.dhparams], axis=-1))

    def eigvals_wrt_damping(self, q: jnp.ndarray, v: jnp.ndarray)->jnp.ndarray:
        """
        Computes the Jacobian of eigenvalues with respect to damping coefficients.
        """
        x = jnp.concatenate([q, v])
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.ms.model.dampings = list(tensor)
            return self.ms.compute_eigvals(x)

        return jax.jacobian(regressor)(jnp.stack(self.ms.model.dampings, axis=-1))

    def state_matrix_a_wrt_inertia(self, q: jnp.ndarray, v: jnp.ndarray)->jnp.ndarray:
        """
        Computes the Jacobian of state matrix A with respect to inertia coefficients.
        """
        x = jnp.concatenate([q, v])
        def regressor(tensor: jnp.ndarray) -> jnp.ndarray:
            self.ms.model.Imats = [tensor[:, :, i] for i in range(tensor.shape[2])]
            return self.ms.get_state_matrix_a(x)

        return jax.jacobian(regressor)(jnp.stack(self.ms.model.Imats, axis=-1))

    def state_matrix_a_wrt_state(self, q: jnp.ndarray, v: jnp.ndarray)->jnp.ndarray:
        """
        Computes the Jacobian of state matrix A with respect to state variables (q, v).
        """
        x = jnp.concatenate([q, v])
        def regressor(tensor):
            return self.ms.get_state_matrix_a(tensor)

        return jax.jacobian(regressor)(x)
