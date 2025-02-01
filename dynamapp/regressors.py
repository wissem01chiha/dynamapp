"""
Regressors module
=================
This module provide helpers functions for automatic 
derivation of the respective regressors matrixes for 
the concrete model or it's state reprsentation input-output
"""
import logging
import numpy as np

from .model import Model
from .model_state import ModelState

logger = logging.getLogger(__name__)

def dgenelizeTorques_d




# class Regressor:
#     """
#     basic class for computing the regressor matrix of the robot.
#     """
#     def __init__(self, robot:Robot=None) -> None:
#         if robot is None:
#             self.robot = Robot()
#         else:
#             self.robot= robot  
#         self.add_col = 4
#         self.param_vector_max_size = (10 + self.add_col) * self.robot.model.nv
        
#     def compute_basic_regressor(self,q:np.ndarray=None,v:np.ndarray=None,a:np.ndarray=None):

#         id_inertias=[]
#         for jj in range(len(self.robot.model.inertias.tolist())):
#             if self.robot.model.inertias.tolist()[jj].mass !=0 :
#                 id_inertias.append(jj)
#         nv= self.robot.model.nv
#         W = np.zeros((nv, (10+self.add_col)*nv))
#         W_mod = np.zeros((nv, (10+self.add_col)*nv))
    
#         W_temp = pin.computeJointTorqueRegressor(self.robot.model, self.robot.data, q, v, a)
#         for j in range(W_temp.shape[0]):
#             W[j, 0 : 10 * nv] = W_temp[j, :]

#             if self.robot.params['identification']['problem_params']['has_friction']:
#                 W[j, 10 * nv + 2 * j] = v[j]  # fv
#                 W[j, 10 * nv + 2 * j + 1] = np.sign(v[j])  # fs
#             else:
#                 W[j , 10 * nv + 2 * j] = 0  # fv
#                 W[j, 10 * nv + 2 * j + 1] = 0  # fs
#             if self.robot.params['identification']['problem_params']['has_actuator']:
#                 W[j, 10 * nv + 2 * nv + j] = a[j]  # ia
#             else:
#                 W[j, 10 * nv + 2 * nv + j] = 0  # ia
#             if self.robot.params['identification']['problem_params']['has_joint_offset']:
#                 W[j, 10 * nv + 2 * nv + nv + j] = 1  # off
#             else:
#                 W[j, 10 * nv + 2 * nv + nv + j] = 0  # off
#         for k in range(nv):
#             W_mod[:, (10 + self.add_col) * k + 9] = W[:, 10 * k + 0]  # m
#             W_mod[:, (10 + self.add_col) * k + 8] = W[:, 10 * k + 3]  # mz
#             W_mod[:, (10 + self.add_col) * k + 7] = W[:, 10 * k + 2]  # my
#             W_mod[:, (10 + self.add_col) * k + 6] = W[:, 10 * k + 1]  # mx
#             W_mod[:, (10 + self.add_col) * k + 5] = W[:, 10 * k + 9]  # Izz
#             W_mod[:, (10 + self.add_col) * k + 4] = W[:, 10 * k + 8]  # Iyz
#             W_mod[:, (10 + self.add_col) * k + 3] = W[:, 10 * k + 6]  # Iyy
#             W_mod[:, (10 + self.add_col) * k + 2] = W[:, 10 * k + 7]  # Ixz
#             W_mod[:, (10 + self.add_col) * k + 1] = W[:, 10 * k + 5]  # Ixy
#             W_mod[:, (10 + self.add_col) * k + 0] = W[:, 10 * k + 4]  # Ixx

#             W_mod[:, (10 + self.add_col) * k + 10] = W[:, 10 * nv + 2 * nv + k]       # ia
#             W_mod[:, (10 + self.add_col) * k + 11] = W[:, 10 * nv + 2 * k]            # fv
#             W_mod[:, (10 + self.add_col) * k + 12] = W[:, 10 * nv + 2 * k + 1]        # fs
#             W_mod[:, (10 + self.add_col) * k + 13] = W[:, 10 * nv + 2 * nv + nv + k]  # off
            
#         return W_mod 
        
        
#     def compute_full_regressor(self,q:np.ndarray=None,v:np.ndarray=None,a:np.ndarray=None):
#         """ 
#         Compute the Regressor matrix of the robot 
#         This function builds the basic regressor of the 10(+4) parameters
#         'Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m'+ ('ia','fs','fv') 
        
#         Args:
#             - q: (ndarray) a configuration position vector 
#             - v: (ndarray) a configuration velocity vector  
#             - a: (ndarray) a configutation acceleration vector
            
#         Returns:
#              - W_mod: (ndarray) basic regressor for 10(+4) parameters 
#                     ( NSamples * ndof, ( 10 + add_col ) * ndof) 
#         """
#         N = len(q) 
#         id_inertias=[]
#         for jj in range(len(self.robot.model.inertias.tolist())):
#             if self.robot.model.inertias.tolist()[jj].mass !=0 :
#                 id_inertias.append(jj)
#         nv= self.robot.model.nv
#         W = np.zeros((N*nv, (10+self.add_col)*nv))
#         W_mod = np.zeros([N*nv, (10+self.add_col)*nv])
#         for i in range(N):
#             W_temp = pin.computeJointTorqueRegressor(
#                 self.robot.model, self.robot.data, q[i, :], v[i, :], a[i, :]
#             )
#             for j in range(W_temp.shape[0]):
#                 W[j * N + i, 0 : 10 * nv] = W_temp[j, :]

#                 if self.robot.params['identification']['problem_params']['has_friction']:
#                     W[j * N + i, 10 * nv + 2 * j] = v[i, j]  # fv
#                     W[j * N + i, 10 * nv + 2 * j + 1] = np.sign(v[i, j])  # fs
#                 else:
#                     W[j * N + i, 10 * nv + 2 * j] = 0  # fv
#                     W[j * N + i, 10 * nv + 2 * j + 1] = 0  # fs
#                 if self.robot.params['identification']['problem_params']['has_actuator']:
#                     W[j * N + i, 10 * nv + 2 * nv + j] = a[i, j]  # ia
#                 else:
#                     W[j * N + i, 10 * nv + 2 * nv + j] = 0  # ia
#                 if self.robot.params['identification']['problem_params']['has_joint_offset']:
#                     W[j * N + i, 10 * nv + 2 * nv + nv + j] = 1  # off
#                 else:
#                     W[j * N + i, 10 * nv + 2 * nv + nv + j] = 0  # off
#         for k in range(nv):
#             W_mod[:, (10 + self.add_col) * k + 9] = W[:, 10 * k + 0]  # m
#             W_mod[:, (10 + self.add_col) * k + 8] = W[:, 10 * k + 3]  # mz
#             W_mod[:, (10 + self.add_col) * k + 7] = W[:, 10 * k + 2]  # my
#             W_mod[:, (10 + self.add_col) * k + 6] = W[:, 10 * k + 1]  # mx
#             W_mod[:, (10 + self.add_col) * k + 5] = W[:, 10 * k + 9]  # Izz
#             W_mod[:, (10 + self.add_col) * k + 4] = W[:, 10 * k + 8]  # Iyz
#             W_mod[:, (10 + self.add_col) * k + 3] = W[:, 10 * k + 6]  # Iyy
#             W_mod[:, (10 + self.add_col) * k + 2] = W[:, 10 * k + 7]  # Ixz
#             W_mod[:, (10 + self.add_col) * k + 1] = W[:, 10 * k + 5]  # Ixy
#             W_mod[:, (10 + self.add_col) * k + 0] = W[:, 10 * k + 4]  # Ixx

#             W_mod[:, (10 + self.add_col) * k + 10] = W[:, 10 * nv + 2 * nv + k]       # ia
#             W_mod[:, (10 + self.add_col) * k + 11] = W[:, 10 * nv + 2 * k]            # fv
#             W_mod[:, (10 + self.add_col) * k + 12] = W[:, 10 * nv + 2 * k + 1]        # fs
#             W_mod[:, (10 + self.add_col) * k + 13] = W[:, 10 * nv + 2 * nv + nv + k]  # off
            
#         return W_mod

#     def computeBasicSparseRegressor(self,q,v,a):
#         """ the torque of joint i do not depend on the torque of joint i-1 """
#         W = self.computeBasicRegressor(q,v,a)
#         for ii in range(W.shape[0]):
#             for jj in range(W.shape[1]):
#                 if ii < jj:
#                     W[ii,jj] = 0
#         return W
        
#     def computeReducedRegressor(self,q,v,a,tol=1e-6):
#         """ 
#         Eliminates columns which has L2 norm smaller than tolerance.
#         Args: 
#             - W: (ndarray) joint torque regressor
#             - tol_e: (float) tolerance
#         Returns: 
#             - Wred: (ndarray) reduced regressor
#         """
#         W = self.computeFullRegressor(q,v,a)
#         col_norm = np.diag(np.dot(np.transpose(W), W))
#         idx_e = []
#         for i in range(col_norm.shape[0]):
#             if col_norm[i] < tol:
#                 idx_e.append(i)
#         idx_e = tuple(idx_e)
#         Wred = np.delete(W, idx_e, 1)
#         return Wred 
    
#     def computeRegressionCriterion(self,torque,q,v,a,x)->float:
#         """ Compute the Regression error model : ε = τ - W.Θ """
#         if np.ndim(x) !=1:
#             logger.error('regression vector should be 1 dimeontional !')
#         if x.size != self.param_vector_max_size:
#             logger.error(f'x array length msismatch expected {self.param_vector_max_size}!')
#         if torque.size !=  self.robot.model.nq:
#             logger.error('error in torques size !')
#         W  = self.computeBasicRegressor(q,v,a)
#         reg_err = torque - np.dot(W,x)
#         return np.linalg.norm(reg_err)
        
    
#     def computeDifferentialRegressor(self, q, v, a, x,dx=1e-2):
#         """ 
#         This function differentiates the computeIdentificationModel of the class robot.
#         Assuming the model is not linear with respect to parameter vector x:
#             τ = f(q, qp, qpp, x)
#         Args:
#             - q: ndarray, joint positions
#             - v: ndarray, joint velocities
#             - a: ndarray, joint accelerations
#             - dx: float, small perturbation for finite difference
#         Returns: 
#             - W: ndarray, (NSamples*ndof, NParams)  regressor matrix
#         """
#         nx = np.size(x)
#         N = len(q)
#         W = np.zeros((N*self.robot.model.nq, nx)) 
#         self.robot.setIdentificationModelData(q, v, a)
#         tau = self.robot.computeIdentificationModel(x)
#         tau = tau.flatten()
    
#         for i in range(nx):
#             x_dx = np.copy(x)
#             x_dx[i] += dx
#             tau_dx = self.robot.computeIdentificationModel(x_dx)
#             tau_dx = tau_dx.flatten()
#             diff = (tau_dx - tau) / dx
    
#             if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
#                 diff[np.isnan(diff)] = 0
#                 diff[np.isinf(diff)] = 0
#             W[:, i] = diff
#         return W
    
#     def computeDifferentiationError(self, q, v, a, x,dx=1e-3):
#         """ retun the gradient differentiation error """
#         self.robot.setIdentificationModelData(q, v, a)
#         f = self.robot.computeIdentificationModel(x)
#         flin = self.computeDifferentialRegressor(q,v,a,x,dx) @ x
#         flin = np.reshape(flin, (-1, self.robot.model.nq))
#         err_per_time = RMSE(flin, f,axis=1)
#         return np.sqrt(np.mean(err_per_time**2))
    
#     def addJointOffset(self,q,v,a,param):
#         if self.robot.params['identification']['problem_params']["has_joint_offset"]:
#             logger.error('Dynamics Engine : Robot has no joint offsets. ')
#             return 
#         W = self.computeBasicRegressor(q,v,a)
#         N = len(q)  
#         nv = self.robot.model.nv
#         add_col = 4
#         for k in range(nv):
#             W[:, (10 + add_col) * k + 13] = 1
#         return W
    
#     def addActuatorInertia(self, q, v, a, param):
#         if self.robot.params['identification']['problem_params']["has_friction"]:
#             N = len(q)  
#         W = self.computeBasicRegressor(q,v,a)
#         nv = self.robot.model.nv
#         for k in range(nv):
#             W[:, (10 + self.add_col) * k + 10] = a
#         return W
    
#     def addFriction(self,q,v,a,param):
#         if self.robot.params['identification']['problem_params']["has_friction"]:
#             logger.error('Dynamics Engine : Robot joints has no friction.')
#             return 
#         W = self.computeBasicRegressor(q,v,a)
#         N = len(self.robot.model.q)  
#         nv = self.robot.model.nv
#         for k in range(nv):
#             W[:, (10 + self.add_col) * k + 11] = self.robot.model.v
#             W[:, (10 + self.add_col) * k + 12] = np.sign(self.robot.model.v)
            
#         return W
    
    
    
#     def computeDifferentialModel(self,q=None,qp=None,qpp=None, inertia_params=None):
        
#         if not(inertia_params is None):
#             self.updateInertiaParams(inertia_params)
#         M = self.computeMassMatrix(q)
#         C = self.computeCorlolisMatrix(qp,q)
#         G = self.computeGravityTorques(q)
#         tau_sim = np.dot(M,qpp)+ np.dot(C, qp) + G
        
#         return tau_sim
    
    
    

    
        
    
    
#     def computeIdentificationModel(self,x:np.ndarray):
#         """
#         This function require the setup up of the joints tarjectory parmters previlouslly 
#         ie self.q, v and a should be puted in the trajectoy or it will use the default.
#         initlize the robot structure with the trajectory data from begin.
#         Args:
#             x : paramters: [inertia, friction, stiffness, actuator, fext]
#                            [13n,     5n,       n,         10n,      6]
#         Returns:
#             tau : model output torque (Nsamples * n )
#         """ 
#         if (self.q is None) or  (self.v is None) or (self.a is None) or \
#         np.ndim(self.q)!=2 or np.ndim(self.v)!=2 or np.ndim(self.a) !=2 :
#             logger.error('Identification Data not set Run : setIdentificationModelData()') 
            
#         if np.ndim(x) != 1:
#             logger.error("X should be 1-dimensional array.")
            
#         fext = self.params['identification']['problem_params']['has_ external_forces']    
#         friction = self.params['identification']['problem_params']['has_friction']
#         motor = self.params['identification']['problem_params']['has_actuator']
#         stiffness = self.params['identification']['problem_params']['has_stiffness']
#         n = self.model.nq
#         if not(fext):
#             self.updateInertiaParams(x[0:13*n])
#             tau = self.computeTrajectoryTorques(self.q,self.v,self.a)
#             if friction:
#                 self.updateFrictionParams(x[13*n:18*n])
#                 tau_f = self.computeFrictionTorques(self.v,self.q)
#                 tau = tau + tau_f
#                 if stiffness:
#                     self.updateStiffnessParams(x[18*n:19*n])
#                     tau_s = self.computeStiffnessTorques(self.q)
#                     tau = tau+tau_f-tau_s
#                     if motor:
#                         self.updateActuatorParams(x[19*n:29*n])
#                         tau_m = self.computeActuatorTorques(self.q,self.v,self.a)
#                         tau = tau_m - tau_f-tau_s
#                 elif motor:
#                     self.updateActuatorParams(x[19*n:29*n])
#                     tau_m = self.computeActuatorTorques(self.q,self.v,self.a)
#                     tau = tau_m - tau_f
#             else:
#                 if stiffness:
#                     self.updateStiffnessParams(x[18*n:19*n])
#                     tau_s = self.computeStiffnessTorques(self.q)
#                     tau = tau -tau_s
#                     if motor:
#                         self.updateActuatorParams(x[19*n:29*n])
#                         tau_m = self.computeActuatorTorques(self.q,self.v,self.a)
#                         tau = tau_m -tau_s
#                 elif motor:
#                     self.updateActuatorParams(x[19*n:29*n])
#                     tau_m = self.computeActuatorTorques(self.q,self.v,self.a)
#                     tau = tau_m 
#         else:
#             self.updateInertiaParams(x[0:13*n])
#             self.updateExternalForces(x[29*n:29*n+6]) 
#             tau = self.computeTrajectoryTorques(self.q,self.v,self.a,self.fext)    
#             if friction:
#                 self.updateFrictionParams(x[13*n:18*n])
#                 tau_f = self.computeFrictionTorques(self.v,self.q)
#                 tau = tau + tau_f
#                 if stiffness:
#                     self.updateStiffnessParams(x[18*n:19*n])
#                     tau_s = self.computeStiffnessTorques(self.q)
#                     tau = tau+tau_f-tau_s
#                     if motor:
#                         self.updateActuatorParams(x[19*n:29*n])
#                         tau_m = self.computeActuatorTorques(self.q,self.v,self.a)
#                         tau = tau_m - tau_f-tau_s
#                 elif motor:
#                     self.updateActuatorParams(x[19*n:29*n])
#                     tau_m = self.computeActuatorTorques(self.q,self.v,self.a)
#                     tau = tau_m - tau_f
#             else:
#                 if stiffness:
#                     self.updateStiffnessParams(x[18*n:19*n])
#                     tau_s = self.computeStiffnessTorques(self.q)
#                     tau = tau -tau_s
#                     if motor:
#                         self.updateActuatorParams(x[19*n:29*n])
#                         tau_m = self.computeActuatorTorques(self.q,self.v,self.a)
#                         tau = tau_m -tau_s
#                 elif motor:
#                     self.updateActuatorParams(x[19*n:29*n])
#                     tau_m = self.computeActuatorTorques(self.q,self.v,self.a)
#                     tau = tau_m    
#         return tau
        
#     def genralizedInertiasParams(self):
#         """
#         Returns the genralized inertia paramters vector
#         Returns:
#             - inertia_vectors (numpy.ndarry) : concatenated inertia paramters of all 
#                 links
#         """
#         inertia_vectors = np.zeros((self.model.nq,13))
#         for i in range(self.model.nq):
#             m_i = self.model.inertias[i].mass
#             c_i = self.model.inertias[i].lever.flatten()
#             I_i =  self.model.inertias[i].inertia.flatten()
#             inertia_vector_i = np.concatenate(([m_i], I_i, c_i))
#             inertia_vectors[i,:] = inertia_vector_i
            
#         inertia_vectors.flatten().reshape(-1, 1)
#         return inertia_vectors
    
#     def updateInertiaParams(self, inertia_vector)->None:
#         """Update the inertia paramters vector"""
#         assert inertia_vector.size == 13 * self.model.nq, \
#             "The size of the inertia vector does not match the expected size."
#         idx = 0
#         for i in range(self.model.nq):
#             m_i = inertia_vector[idx]
#             c_i = inertia_vector[idx + 1:idx + 4]
#             I_i_flat = inertia_vector[idx + 4:idx + 13]
#             I_i = I_i_flat.reshape(3,3)
#             self.model.inertias[i].mass = m_i
#             self.model.inertias[i].lever = c_i
#             self.model.inertias[i].inertia = I_i
#             idx += 13
        
#     def setRandomInertiaParams(self)->None:
#         """set the robot links inertia paramters to random values."""
#         Xphi = self.genralizedInertiasParams()
#         randXphi = np.random.rand(np.shape(Xphi))
#         self.updateInertiaParams(randXphi)
        
#     def computeBaseInertiasParams(self):
#         """  
#         #TODO to implement 
#         Compute the manipulator Base inertial parameters 
#         Returns
#             base_params : numpy-ndarry 
#         """
#         base_params_vector = 1 
#         return base_params_vector
        
#     def updateFrictionParams(self, new_params)-> None:
#         """
#         update the robot friction parameters.
#         Args:
#             - new_params : ndarry of size min 14 (2n) max is 35(5n)
#         """   
#         friction_type= self.params['robot_params']['friction']
#         n =  self.model.nq
#         if friction_type == 'viscous': # array of 14
#             if new_params.size < 2 * n:
#                 logger.error(f"min parms number for friction model is {2 * n}")
#                 return 
#             else:
#                 new_params = new_params[0:2*n]
#             self.params['friction_params']['viscous']['Fc']= new_params[0:n]
#             self.params['friction_params']['viscous']['Fs']= new_params[n:2*n]
            
#         elif friction_type == 'lugre':
#             if new_params.size < 5 * n: # array of 35
#                 logger.error(f'min params number for friction model is {5 * n}')
#                 return
#             else:
#                 new_params = new_params[0:5*n]
#             self.params['friction_params']['lugre']['Fc']= new_params[0:n]
#             self.params['friction_params']['lugre']['Fs']= new_params[n:14]
#             self.params['friction_params']['lugre']['sigma0'] = new_params[14:21]
#             self.params['friction_params']['lugre']['sigma1'] = new_params[21:28]
#             self.params['friction_params']['lugre']['sigma2'] = new_params[28:35]
            
#         elif friction_type == 'maxwellSlip': # array of 13
#             if new_params.size < 6 + n:
#                 logger.error(f"min params number for friction model is {6 + n}")
#             else:
#                 new_params = new_params[0:13]
#             self.params['friction_params']['maxwellSlip']['k'] = new_params[0:3]
#             self.params['friction_params']['maxwellSlip']['c'] = new_params[3:6]
#             self.params['friction_params']['maxwellSlip']['sigma0'] = new_params[6:13]
            
#         elif friction_type == 'dahl': # array of 14
#             if new_params.size < 2*n:
#                 logger.error(f"min params number for friction model is {2*n}")
#             else:
#                 new_params = new_params[0:14]
#             self.params['friction_params']['dahl']['sigma0'] = new_params[0:7]
#             self.params['friction_params']['dahl']['Fs'] = new_params[7:14]
        
#     def updateStiffnessParams(self, new_params)-> None:
#         """
#         Update the joints stiffness paramters.
#         Args:
#             new_params (numpy ndarry)
#         """ 
#         assert new_params.size== self.model.nq,\
#             "stiffness inputs should be equal to robot joints number"
#         self.params['stiffness_params'] = new_params
        
#     def getStiffnessMatrix(self)->np.ndarray:
#         """
#         Return the diagonal stiffness matrix of the robot formed by each joint 
#         stiffness factor.
#         """
#         matrix = np.eye(self.model.nq)
#         for i in range(self.model.nq):
#             matrix[i,i] = self.params['stiffness_params'][i]
#         return matrix 
        
#     def computeActuatorTorques(self, q, qp, qpp):
#         """
#         Estimates the joints motors torque from position, velocity and acceleration. 
#         Args:
#             - q: Joints position (Nsamples * ndof)
#             - qp: Joints velocity (Nsamples * ndof)
#         Returns:
#             - tau_m : numpy.ndarry (Nsamples *ndof)
#         """
#         tau_m = np.zeros_like(q)
#         I = self.params['actuator_params']['inertia']
#         kt = self.params['actuator_params']['kt']
#         damping = self.params['actuator_params']['damping']
#         for i in range(self.model.nq):
#             motor_i = BLDC(I[i], kt[i], damping[i])
#             tau_m[:,i] = motor_i.computeOutputTorque(q[:,i], qp[:,i], qpp[:,i])
            
#         return tau_m
    
#     def updateActuatorParams(self, new_params:np.ndarray)->None:
#         """
#         Updates the joints actuator parameters.
#         Bounds for torque and current (Tmax, Imax) are exclued from update.
        
#         Args:
#             new_params [kt, inertia, damping, Ta, Tb, Tck] (numpy-ndarry) 1 * 10.ndof
#         """
#         n = self.model.nq 
#         assert new_params.size == 10 * n
#         self.params['actuator_params']['kt'] = new_params[0:n]
#         self.params['actuator_params']['inertia'] = new_params[n:2*n]
#         self.params['actuator_params']['damping'] = new_params[2*n:3*n]
#         self.params['actuator_params']['Ta'] = new_params[3*n:4*n]
#         self.params['actuator_params']['Tb'] = new_params[4*n:5*n]
#         self.params['actuator_params']['Tck'] = new_params[5*n:10*n].reshape((n, 5))
    
#     def getActuatorInertiasMatrix(self)->np.ndarray:
#         I = self.params['actuator_params']['inertia']
#         Im = np.eye(len(I))
#         for i in range(len(I)):
#             Im[i] = I[i]
#         return Im