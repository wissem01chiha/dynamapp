from typing import Tuple
import jax
import jax.numpy as jnp

from .model import Model 
from .model_state import ModelState
from .trajectory import *

class ModelDataGenerator:
    """
    This class generates data based on a multibody system for a given trajectory.
    It computes the joint positions, velocities, accelerations, and torques based on
    the defined trajectory and model dynamics.

    Args:
        - model (Model): The multibody model to be used.
        - trajectory (Trajectory): The trajectory to generate data for.
    """
    
    def __init__(self, model: Model, trajectory: Trajectory):
        self.model = model
        self.trajectory = trajectory
        
    def generate_trajectory_data(self):
        """
        Generate full data for the trajectory with respect to the model's kinematics and dynamics.
        
        Returns:
            - dict: A dictionary containing joint positions, velocities, accelerations, and torques
                over the trajectory time.
        """
        q_data = self.trajectory.compute_full_trajectory()
        q_dot_data = self.compute_velocities(q_data)
        q_ddot_data = self.compute_accelerations(q_data, q_dot_data)

        tau_data = self.compute_torques(q_data, q_dot_data, q_ddot_data)

        return {
            'q': q_data, 
            'q_dot': q_dot_data,  
            'q_ddot': q_ddot_data,  
            'tau': tau_data   
        }

    def compute_velocities(self, q_data):
        """
        Compute the joint velocities (q_dot) using the trajectory data.
        
        Args:
            - q_data (jnp.ndarray): Joint positions over the trajectory time.
        
        Returns:
            - jnp.ndarray: Joint velocities (q_dot).
        """
        q_dot_data = jnp.gradient(q_data, axis=0) / self.trajectory.sampling
        return q_dot_data
    
    def compute_accelerations(self, q_data, q_dot_data):
        """
        Compute the joint accelerations (q_ddot) using the joint position and velocity data.
        
        Args:
            - q_data (jnp.ndarray): Joint positions over the trajectory time.
            - q_dot_data (jnp.ndarray): Joint velocities over the trajectory time.
        
        Returns:
            - jnp.ndarray: Joint accelerations (npoints * ndof).
        """
        q_ddot_data = jnp.gradient(q_dot_data, axis=0) / self.trajectory.sampling
        return q_ddot_data
    
    def compute_torques(self, q_data, q_dot_data, q_ddot_data):
        """
        Compute the joint torques using the Recursive Newton-Euler Algorithm (RNEA).
        
        Args:
            - q_data (jnp.ndarray): Joint positions over the trajectory time.
            - q_dot_data (jnp.ndarray): Joint velocities over the trajectory time.
            - q_ddot_data (jnp.ndarray): Joint accelerations over the trajectory time.
        
        Returns:
            - jnp.ndarray: Joint torques (tau).
        """
        assert q_data.shape == q_dot_data.shape == q_ddot_data.shape, \
        " data arrays do not have the same shape"
        assert q_data.ndim == 2, "Input arrays should be 2D JAX arrays"
        tau_data = jnp.zeros(q_data.shape)
        for i in range(self.model.ndof):
            tau_up = self.model.generalized_torque(i,q_data[:,i],q_dot_data[:,i],q_ddot_data[:,i])
            tau_data = tau_data.at[0,i].set(tau_up[0])
            tau_data = tau_data.at[1,i].set(tau_up[1])
            tau_data = tau_data.at[2,i].set(tau_up[2])
            
        return tau_data

class ModelStateDataGenerator:
    r"""
    Class to generate data for state-dependent multijoint system models.
    
    This class simulates the `ModelState` class over time and generates a 
    sequence of state, input, and output data.
    
    Args:
        - model_state     - instance of the `ModelState` class
        - num_samples     - number of samples to generate
        - time_steps      - number of time steps per sample
        - noise_magnitude - magnitude of noise to add to the output
        - u_init          - initial input vector
        - x_init          - initial state vector
    
    Examples:
        >>> model_state = ModelState(Imats, dhparams)
        >>> data_generator = ModelStateDataGenerator(model_state, num_samples=100, time_steps=200)
        >>> x_data, u_data, y_data = data_generator.generate_data()
    
    Attributes:
        - model_state     - instance of the `ModelState` class
        - num_samples     - number of samples to generate
        - time_steps      - number of time steps per sample
        - noise_magnitude - magnitude of noise to add to the output
        - u_init          - initial input vector
        - x_init          - initial state vector
    """
    def __init__(self, 
                 model_state: 'ModelState', 
                 num_samples: int, 
                 time_steps: int, 
                 noise_magnitude: float = 0.1,
                 u_init: jnp.ndarray = None, 
                 x_init: jnp.ndarray = None) -> None:
        self.model_state = model_state
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.noise_magnitude = noise_magnitude
        self.u_init = u_init if u_init is not None else jnp.zeros((self.model_state.u_dim, 1))
        self.x_init = x_init if x_init is not None else jnp.zeros((self.model_state.x_dim, 1))
    
    def generate_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate data for a multijoint system, consisting of input (u), state (x), and output (y).
        
        Args:
            None
            
        Returns:
            - x_data : jnp.ndarray (num_samples, time_steps, 2*ndof) state data
            - u_data : jnp.ndarray (num_samples, time_steps, ndof) input data
            - y_data : jnp.ndarray (num_samples, time_steps, ndof) output data
        """
        x_data = []
        u_data = []
        y_data = []
        
        for _ in range(self.num_samples):
            u_seq = []
            x_seq = [self.x_init]
            y_seq = []
            
            for t in range(self.time_steps):
                u_t = self._get_input_signal(t)
                e_t = self._get_noise_signal(t)
                y_t = self.model_state.step(u=u_t, e=e_t)
                
                u_seq.append(u_t)
                x_seq.append(self.model_state.xs[-1])
                y_seq.append(y_t)
            
            u_data.append(jnp.array(u_seq))
            x_data.append(jnp.array(x_seq[1:])) 
            y_data.append(jnp.array(y_seq))
        
        return jnp.array(x_data), jnp.array(u_data), jnp.array(y_data)
    
    def _get_input_signal(self, t: int) -> jnp.ndarray:
        """
        Generate an input signal for the system at a given time step `t`.
        
        Args:
            t : int - time step index
        
        Returns:
            - u_t : jnp.ndarray - input signal at time step `t`
        """
        u_t = self.u_init * jnp.sin(2 * jnp.pi * t / self.time_steps)
        return u_t
    
    def _get_noise_signal(self, t: int) -> jnp.ndarray:
        """
        Generate a noise signal for the system at a given time step `t`.
        
        Args:
            t : int - time step index
        
        Returns:
            - e_t : jnp.ndarray - noise signal at time step `t`
        """
        noise = jax.random.normal(jax.random.PRNGKey(t), shape=(self.model_state.y_dim, 1))
        return self.noise_magnitude * noise

    def compute_data_statistics(self, x_data: jnp.ndarray, u_data: jnp.ndarray, y_data: jnp.ndarray) -> dict:
        """
        Compute some statistics (mean, std) over the generated data.
        
        Args:
            x_data : jnp.ndarray (num_samples, time_steps, 2*ndof) state data
            u_data : jnp.ndarray (num_samples, time_steps, ndof) input data
            y_data : jnp.ndarray (num_samples, time_steps, ndof) output data
        
        Returns:
            - stats : dict - statistics (mean, std) for state, input, and output data
        """
        stats = {
            "x_mean": jnp.mean(x_data, axis=(0, 1)),
            "x_std": jnp.std(x_data, axis=(0, 1)),
            "u_mean": jnp.mean(u_data, axis=(0, 1)),
            "u_std": jnp.std(u_data, axis=(0, 1)),
            "y_mean": jnp.mean(y_data, axis=(0, 1)),
            "y_std": jnp.std(y_data, axis=(0, 1)),
        }
        return stats



