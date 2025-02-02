"""
Viscoelastic Module
====================
"""
from jax import jit
import jax.numpy as jnp

@jit
def coulomb_friction_force(v: jnp.ndarray, fc: jnp.ndarray, 
                        fs: jnp.ndarray)->jnp.ndarray:
    """
    Compute the Coulomb and viscous friction model.

    Args:
        v  (jnp.ndarray): Velocity array.
        fc (jnp.ndarray): Coulomb friction coefficient.
        fs (jnp.ndarray): Viscous friction coefficient.

    Returns:
        jnp.ndarray: Friction force array.
    """
    return fc * jnp.sign(v) + fs * v


# class LuGre:
#     """
#     Class to compute LuGre Friction Model 
    
#     Params:
#         - Fc (float): Coulomb friction coefficient.
#         - Fs (float): Stribeck friction coefficient.
#         - v (float): Joint velocity.
#         - vs (float): Kinetic velocity transition.
#         - sigma0 (float): Model parameter sigma0.
#         - sigma1 (float): Model parameter sigma1.
#         - sigma2 (float): Model parameter sigma2.
#         - tinit (float): Initial simulation time.
#         - ts (float): Step time simulation.
#         - tspan (float): Final simulation time.
#         - z0 (float): Initial value of internal state z.
#     """

#     def __init__(self, Fc, Fs, v, sigma0, sigma1, sigma2, tspan, ts=0.001, tinit=0, z0=0.01, vs=0.1235):
#         self.Fc = Fc
#         self.Fs = Fs
#         self.v = v
#         self.vs = vs
#         self.sigma0 = sigma0
#         self.sigma1 = sigma1
#         self.sigma2 = sigma2
#         self.tinit = tinit
#         self.ts = ts
#         self.tspan = tspan
#         self.z0 = z0

#     def computeFrictionForce(self):
#         """
#         Compute friction force over the simulation time span.

#         > Returns:
#             - F (numpy.ndarray): Friction force for the given velocity.
#         """
#         t = np.arange(self.tinit, self.tspan + self.ts, self.ts)
#         N = len(t)
        
#         z = np.zeros(N, dtype=np.float64)
#         F = np.zeros(N, dtype=np.float64)
#         z[0] = self.z0

#         for idx in range(1, N):
#             v_safe = max(abs(self.vs), 1e-3)
#             sigma0_safe = max(abs(self.sigma0), 1e-6)
#             exp_input = -(self.v / v_safe) ** 2
#             exp_input_clipped = np.clip(exp_input, -1e6, 1e6)
#             gv = (self.Fc + (self.Fs - self.Fc) * np.exp(exp_input_clipped)) / sigma0_safe
#             gv = max(gv, 1e-4)  # Ensure gv does not become too small
            
#             z_dot = self.v - abs(self.v) * z[idx-1] / gv
#             z[idx] = z[idx-1] + z_dot * self.ts
#             if np.isnan(z[idx]) or np.isinf(z[idx]):
#                 z[idx] = 0
            
#             F[idx] = self.sigma0 * z[idx] + self.sigma1 * z_dot + self.sigma2 * self.v
#             if np.isnan(F[idx]) or np.isinf(F[idx]):
#                 F[idx] = 0

#         return F

#     def computeSteadyForce(self):
#         """
#         Compute the LuGre steady state friction force

#         Returns:
#             - Fss (float): Steady state friction force.
#         """
#         v_safe = max(np.abs(self.vs), 1e-6)
#         exp_input = -(self.v / v_safe) ** 2
#         exp_input_clipped = np.clip(exp_input, -1e6, 1e6)
#         Fss = self.Fc * np.sign(self.v) + (self.Fs - self.Fc) * \
#               np.exp(exp_input_clipped) * np.sign(self.v) + self.sigma2 * self.v
#         return Fss
    
    
    
# class Dahl:
#     """
#     Dahl friction Model class base definition.
#     The friction force is a hysteresis function, without memory of the x
#     Args:
#         - sigma0: Constant coefficient
#         - Fs    : Stribeck force coefficient
#     """
#     def __init__(self, sigma0, Fs, time_step=0.001) -> None:
#         assert sigma0 is not None and sigma0 != 0, \
#             "Viscoelastic Engine: coefficient sigma must be non-null float."
#         assert Fs is not None and Fs != 0, \
#             "Viscoelastic Engine: coefficient Fs must be non-null float."
#         self.sigma0    = sigma0 
#         self.Fs        = Fs     
#         self.time_step = time_step 

#     def computeFrictionForce(self, velocity:np.ndarray) -> np.ndarray:
#         """
#         Compute the friction force based on the Dahl model.
        
#         Args:
#             - velocity (np.ndarray): velocity values.
            
#         Returns:
#             np.ndarray: computed friction forces.
#         """
#         time_span = (velocity.size-1)* self.time_step
#         t = np.linspace(0, time_span, velocity.size)
#         F = np.zeros_like(velocity)
#         dFdt = np.zeros_like(F)
#         for i in range(1,velocity.size):
#             dFdt[i] = self.dahl(F[i-1],velocity[i])
#             F[i] = dFdt[i] *self.time_step + F[i-1]
#         return F
    
#     def dahl(self,F,v):
#         if v == 0:
#             dFdt = 0
#         else:
#             if self.Fs != 0 : 
#                 dFdt = self.sigma0 /v
#             else:
#                 dFdt = self.sigma0 /v*(1- F/self.Fs*np.sign(v))
#         return dFdt
# class Maxwell:
#     """
#     Maxwell-Voight contact model class.
    
#     Args:
#         sigma0 (float): Initial stress value.
#         eta (float): Viscosity parameter.
#         E (float): Elastic modulus.
#     """

#     def __init__(self, sigma0: float, eta: float, E: float) -> None:
#         """Initialize the Maxwell-Voight model parameters."""
#         self.sigma0 = sigma0
#         self.eta = eta
#         self.E = E

#     def stress(self, strain: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
#         """
#         Calculate the stress based on strain and strain rate using the Maxwell-Voight model.

#         Args:
#             strain (np.ndarray): Strain values.
#             strain_rate (np.ndarray): Strain rate values.

#         Returns:
#             np.ndarray: Calculated stress values.
#         """
#         return self.sigma0 + self.E * strain + self.eta * strain_rate

#     def simulate(self, time: np.ndarray, strain: np.ndarray) -> np.ndarray:
#         """
#         Simulate the stress response over time for a given strain history.

#         Args:
#             time (np.ndarray): Time values.
#             strain (np.ndarray): Strain values over time.

#         Returns:
#             np.ndarray: Stress values over time.
#         """
#         strain_rate = np.gradient(strain, time)
#         stress = self.stress(strain, strain_rate)
#         return stress

# class MaxwellSlip:
#     """
#     MaxwellSlip - Compute Maxwell Slip Friction Model.

#     Inputs:
#       n           - Number of Maxwell elements.
#       velocity    - Velocity (m/s)
#       k           - Stiffness of Maxwell elements (N/m)
#       c           - Damping coefficients of Maxwell elements (Ns/m)
#       sigma0      - Static friction force (N)
#       samplingRate- Sampling rate (Hz)

#     Returns:
#       t           - Simulation time vector.
#       F           - Friction Force for the given velocity
#     Note:
    
#     Ref:
#       Fundamentals Of Friction Modeling - Farid Al-Bender - 2010.
#     """

#     def __init__(self, n, velocity:np.ndarray, k, c, sigma0, samplingRate=1000):
#       assert samplingRate != 0,"Sampling frequency should not be null."
#       self.n = n
#       self.velocity = velocity
#       self.k = k
#       self.c = c
#       assert len(self.k) == n,\
#         "Length of stiffness coefficients (k) should be equal to the number of Maxwell elements."
#       assert len(self.c) == n,\
#         "Length of damping coefficients (c) should be equal to the number of Maxwell elements."
#       self.sigma0 = sigma0
#       self.samplingRate = samplingRate
      
#     def maxwell(self, y, t):
#         dydt = np.zeros(2*self.n)   
#         F = y[self.n:]
#         for i in range(self.n):
#             dxdt = np.mean(self.velocity) - F[i] / self.c[i]
#             dFdt = self.k[i] * dxdt
#             dydt[i] = dxdt
#             dydt[self.n + i] = dFdt
#         F_total = np.sum(F)
#         if np.abs(F_total) < self.sigma0:
#             dydt[self.n:] = 0
#         return dydt

#     def computeFrictionForce(self):
#         timeSpan = (len(self.velocity) - 1) / self.samplingRate
#         t = np.linspace(0, timeSpan, len(self.velocity))
#         initial_conditions = np.zeros(2*self.n)
#         y = odeint(self.maxwell, initial_conditions, t)
#         F = np.sum(y[:, self.n:], axis=1)
#         return F

#     def compute_friction_torques(self, qp:np.ndarray, q:np.ndarray):
#         """
#         Estimates the friction torque vector in robot joints given a 
#         constant joint velocity vector.
 
#         Args:
#             - qp       : Joints velocity vector  ( numSamples  * ndof )
#             - q        : Joints position vector  ( numSamples * ndof )
#             - tspan    : Simulation time duration (seconds)
#             - sampling : Sampling frequency        (Hz)
            
#         Returns:
#             tau_f      : Joints friction torques   ( numSamples * ndof )
#         """
#         sampling = self.params['simulation']['sampling_frequency']
#         frictionModel = self.params['robot_params']['friction'] 
        
#         NSamples, ndof = np.shape(qp)
#         tspan = ( NSamples -1 )/sampling
#         tau_f = np.zeros_like(qp)
    
#         assert ndof == self.model.nq,'Joints velocity data input msiamtch with model degree of freedom'
                
#         if frictionModel == 'lugre':
#             Fc = self.params['friction_params']['lugre']['Fc']
#             Fs = self.params['friction_params']['lugre']['Fs']
#             sigma0 = self.params['friction_params']['lugre']['sigma0']
#             sigma1= self.params['friction_params']['lugre']['sigma1']
#             sigma2= self.params['friction_params']['lugre']['sigma2']
#             qp_m = np.zeros_like(qp)
#             window_size = 10
#             for k in range(ndof):
#                 for t in range(1, len(qp), window_size):
#                     model = LuGre(Fc[k], Fs[k], qp[t,k],sigma0[k], sigma1[k], sigma2[k],\
#                         t,t/(2*window_size),max(t-window_size,0))
#                     F = model.computeFrictionForce() 
#                     tau_f[t:min(len(qp),t+window_size),k] = np.mean(F)
                    
                
#         elif frictionModel == 'maxwellSlip':
  
#             for j in range(int(q.shape[1])): 
#                 n = self.params['friction_params']['maxwellSlip']['n']
#                 k = self.params['friction_params']['maxwellSlip']['k']
#                 c = self.params['friction_params']['maxwellSlip']['c']
#                 sigma0 = np.array(self.params['friction_params']['maxwellSlip']['sigma0'])
#                 model = MaxwellSlip(n, q[:,j], k, c, sigma0[j],sampling)
#                 tau_f[:,j]= model.computeFrictionForce()
                
#         elif frictionModel == 'dahl':
#             for k in range(ndof):
#                 sigma0 = self.params['friction_params']['dahl']['sigma0']
#                 Fs = self.params['friction_params']['dahl']['Fs']
#                 model = Dahl(sigma0[k], Fs[k])
#                 tau_f[:,k]  = model.computeFrictionForce(qp[:,k])
                
#         elif frictionModel == 'viscous':
#             for k in range(ndof):
#                 Fc = self.params['friction_params']['viscous']['Fc']
#                 Fs = self.params['friction_params']['viscous']['Fs']
#                 tau_f[:,k] = computeViscousFrictionForce(qp[:,k],Fc[k],Fs[k])
#         else:
#             logger.error("Friction Model Not Supported Yet!")
#         return tau_f 