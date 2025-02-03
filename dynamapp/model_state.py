import jax
import jax.numpy as jnp

from .math_utils import validate_matrix_shape
from .model import Model
from .state_space import StateSpace

class ModelState:
    r"""
    Base class for state space mutijoint models, these model are typically complex
    because, the base matrices A, B, C and D is a function of the state vector x.
    which is not the case for the classical state space model, these model are called
    state depend state space models, these property makes all derived equations (eg Riccati,
    observer, etc) to be state dependant as the following equation:
    
    .. math::
        \begin{cases}
            x_{k+1} &= A(x_k) x_k + B(x_k) u_k + K e_k \\
            y_k &= C(x_k) x_k + D(x_k) u_k + e_k
        \end{cases}
    
    Args:
        - model             - multijoint base model 
        - model_state_space - state space representation of the model
    Examples:
        >>> model = ModelState(file_path, x_init)
        >>> y = model.output(x, u)
        
    """
    def __init__(self, Imats:list, 
                dhparams:list, 
                gravity = -9.81, 
                dampings:list = None, 
                x_init:jnp.ndarray = None) -> None:
        
        self.model = Model(Imats,dhparams,gravity,dampings)
        self.model_state_space = StateSpace(jnp.zeros((2*self.model.ndof, 2*self.model.ndof)),
                                            jnp.zeros((2*self.model.ndof, self.model.ndof)),
                                            jnp.zeros((self.model.ndof, 2*self.model.ndof)),
                                            jnp.zeros((self.model.ndof, self.model.ndof)))
        self.x_dim = 2*self.model.ndof
        self.u_dim = self.model.ndof
        self.y_dim = self.model.ndof
        self.set_x_init(x_init)
        self._compute_matrices(self.x_init)
        self.xs = []
        self.ys = []
        self.us = []
            
    def _compute_matrices(self, x:jnp.ndarray)->None:
        """
        Compute the state space matrices of the state model and update the
        state space model.
        
        Args:
            - x: jax array : state vector  
        """
        q = x[0:self.u_dim].ravel()
        qp = x[self.u_dim:self.x_dim].ravel()
        
        A = jnp.zeros((2*self.u_dim, 2*self.u_dim))
        B = jnp.zeros((2*self.u_dim, self.u_dim))
        C = jnp.zeros((self.u_dim, 2*self.u_dim))
        D = jnp.zeros((self.u_dim, self.u_dim))

        c = self.model.coriolis_tensor(q, qp)
        M = self.model.inertia_tensor(q)
        K = self.model.damping_tensor()

        A_up = -jnp.dot(jnp.linalg.inv(M) , c)
        A = A.at[self.u_dim:2*self.u_dim, self.u_dim:2*self.u_dim].set(A_up)
        A = A.at[0:self.u_dim, self.u_dim:2*self.u_dim].set(jnp.eye(self.u_dim))
        
        A_up = -jnp.dot(jnp.linalg.inv(M) , K)
        A = A.at[self.u_dim:2*self.u_dim, 0:self.u_dim].set(A_up)
        A = A.at[0:self.u_dim, 0:self.u_dim].set(jnp.zeros((self.u_dim, self.u_dim)))
        
        B_up = jnp.linalg.inv(M)
        B= B.at[self.u_dim:2*self.u_dim,0:self.u_dim].set(B_up)
        C = C.at[0:self.u_dim, 0:self.u_dim].set(jnp.eye(self.u_dim))
        self.model_state_space.set_matrices(A, B, C, D)

    def output(self,
            x:jnp.ndarray,
            u:jnp.ndarray,
            e:jnp.ndarray)->jnp.ndarray:
        """
        Compute the system output vector given the state vector x and the input vector u
        with repect to the equation :
        
        .. math::
            y_k = C x_k + D u_k + e_k
        
        Returns:
            - y_k - np.ndarray (2.ndof * 1) the output vector of the system at time t
        """
        self._compute_matrices(x)
        return self.model_state_space.output(x,u)
    
    def step(self,
            u: jnp.ndarray = None,
            e: jnp.ndarray = None
        ) -> jnp.ndarray:
        """
        Compute the output of the state-space model and returns it at time t+1
        Updates the internal state of the model as well.
        
        Args:
            - u input vector 
            - e : noise vector
        """
        if u is None:
            u = jnp.zeros((self.u_dim, 1))
        if e is None:
            e = jnp.zeros((self.y_dim, 1))
        validate_matrix_shape(u, (self.u_dim, 1), 'u')
        validate_matrix_shape(e, (self.y_dim, 1), 'e')
        x = self.xs[-1] if self.xs else self.x_init
        self._compute_matrices(x)
        y = self.model_state_space.step(u,e)
        self.us.append(u)
        self.xs.append(x)
        self.ys.append(y)
        return y
        
    def set_x_init(self, x_init: jnp.ndarray):
        """ 
        Set the initial state, if it is given. 
        """
        if x_init is None:
            x_init = jnp.zeros((self.x_dim, 1))
        validate_matrix_shape(x_init, (self.x_dim, 1), 'x_dim')
        self.x_init = x_init
    
    def compute_eigvals(self, x:jnp.ndarray)->jnp.ndarray:
        """ Computes the system eigvalues at time t given the state vector x."""
        self._compute_matrices(x)       
        return  jnp.linalg.eigvals(self.model_state_space.a)
    
    def _is_stable(self,x:jnp.ndarray)->bool:
        """ 
        Check what ever the system is stable or not at time t and for a given state vector x
        NOTE: the system is stable if all the eigenvalues of the system matrix A are less than 1
        Returns:
            - nattive python bool
        """
        eigenvalues = jnp.linalg.eigvals(self.model_state_space.a)
        return bool(jnp.all(jnp.abs(eigenvalues) < 1).item())
    
    def lsim(self, u:jnp.ndarray, e:jnp.ndarray)->jnp.ndarray:
        """
        Simulate the system response with a given input u subject to the noise e
        Args:
            - u : input vector (ndof * NSamples)
            - e : noise vector (ndof * NSamples)
        Return:
            - xs : numpy-ndarry store iteration states vectors. ( 2.ndof * NSamples)
        """
        def wrap_step(carry, inputs):
            u_t, e_t = inputs 
            u_t = u_t[:,None]
            e_t = e_t[:,None]
            y = self.step(u_t, e_t)  
            return carry, y
        
        _, xs = jax.lax.scan(wrap_step, None, (u.T, e.T))
        return xs
    
    def compute_obs_matrix(self, x:jnp.ndarray)->jnp.ndarray:
        """
        Compute the observaliblite matrix of the system at time t given the state vector x
        The observability matrix is given by the following equation:
        
        .. math::
            O =[ C , CA,  CA^2, ..., CA^{n-1}]
        
        Args:
            - x : state vector (2.ndof * 1)
        Returns:
            - obs_matrix : numpy-ndarry (2.ndof * 2.ndof)
        """
        self._compute_matrices(x)
        n = self.model_state_space.a.shape[0]
        return jnp.vstack([self.model_state_space.c @ 
                           jnp.linalg.matrix_power(self.model_state_space.a, i) 
                           for i in range(n)])
    
    def compute_ctlb_matrix(self, x: jnp.ndarray):
        """
        Compute the controllability matrix of the system
        The controllability matrix is given by the following equation:
        
        .. math::
            C = [B , AB , A^2B , ... A^{n-1}B]
              
        Args:
            - x : state vector (2.ndof * 1)
        Returns:
            - ctlb_matrix : numpy-ndarry (2.ndof * 2.ndof)
        """
        self._compute_matrices(x)
        n = self.model_state_space.a.shape[0]
        return jnp.hstack([jnp.linalg.matrix_power(self.model_state_space.a, i) @ 
                           self.model_state_space.b for i in 
                           range(n)])
        
    def get_state_matrix_a(self, x:jnp.ndarray):
        """
        Compute the state matrix A given an input state vector x at time date t
        """
        self._compute_matrices(x)
        return self.model_state_space.a
            

