import jax.numpy as jnp
import abc

class Trajectory(abc.ABC):
    """
    Base class for trajectory motion generation.
    """
    def __init__(self, n: int, sampling: int, ti: float, tf: float):
        self.n = n
        self.sampling = sampling
        self.ti = ti
        self.tf = tf
        self.time = jnp.linspace(ti, tf, sampling)
        
    @abc.abstractmethod
    def get_value(self, t: float):
        pass
    
    @abc.abstractmethod
    def compute_with_constraints(self, qmin, qmax, qpmin, qpmax, qppmin, qppmax):
        pass
    
    @abc.abstractmethod
    def compute_full_trajectory(self):
        pass

class SplineTrajectory(Trajectory):
    """
    Spline-based trajectory.
    """
    def __init__(self, ndof, sampling, ti, tf, control_points):
        super().__init__(ndof, sampling, ti, tf)
        self.control_points = jnp.array(control_points)

    def get_value(self, t: float):
        return jnp.interp(t, self.time, self.control_points)

    def compute_full_trajectory(self):
        return jnp.interp(self.time, self.time, self.control_points)

class TrapezoidalTrajectory(Trajectory):
    """
    Trapezoidal velocity profile trajectory.
    """
    def __init__(self, n, sampling, ti, tf, q0, qf, acc, vel):
        super().__init__(n, sampling, ti, tf)
        self.q0 = jnp.array(q0)
        self.qf = jnp.array(qf)
        self.acc = jnp.array(acc)
        self.vel = jnp.array(vel)

    def get_value(self, t: float):
        return self.q0 + (self.qf - self.q0) * (t - self.ti) / (self.tf - self.ti)

    def compute_full_trajectory(self):
        return self.get_value(self.time)

class PeriodicTrajectory(Trajectory):
    """
    Periodic trajectory based on Fourier series.[1]
    
    Ref:
        [1] Fourier-based optimal excitation trajectories for the dynamic 
        identification of robots, Kyung.Jo Park - Robotica - 2006. 
    """
    def __init__(self, n, sampling, ti, tf, frequency, Aij, Bij, nb_terms):
        super().__init__(n, sampling, ti, tf)
        self.frequency = frequency
        self.Aij = jnp.array(Aij)
        self.Bij = jnp.array(Bij)
        self.nb_terms = nb_terms

    def get_value(self, t: float):
        omega = 2 * jnp.pi * self.frequency
        q = jnp.zeros(self.ndof)
        for i in range(self.ndof):
            for j in range(1, self.nb_terms + 1):
                q = q + (self.Aij[i, j-1] * jnp.cos(omega * j * t) +
                         self.Bij[i, j-1] * jnp.sin(omega * j * t))
        return q
    
    def compute_full_trajectory(self):
        return jnp.array([self.get_value(t) for t in self.time])

class StepTrajectory(Trajectory):
    """
    Step trajectory with fixed small duration epsilon and given amplitude.
    """
    def __init__(self, ndof, sampling, ti, tf, epsilon, amplitude):
        super().__init__(ndof, sampling, ti, tf)
        self.epsilon = epsilon
        self.amplitude = amplitude

    def get_value(self, t: float):
        return jnp.where(t % (2 * self.epsilon) < self.epsilon, self.amplitude, 0)

    def compute_full_trajectory(self)->jnp.array:
        return jnp.array([self.get_value(t) for t in self.time])