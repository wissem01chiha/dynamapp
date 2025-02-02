"""
Identification Module
====================
"""
from .moesp import *
from .nfoursid import *
from .data_utils import *
from .regressors import *
from .reductions import *
from .kalman import *
from .math_utils import *

# class IDIM:
    
#     def __init__(self) -> None:
#         self.IDIM_LE 
#         pass
        
    
#     def set_upper_bounds(self):
#         pass
    
   
#     def set_lower_bounds(self):
#         pass

    
#     def set_maxeval(self):
#         pass

   
#     def set_set_min_objective(self,f:Callable[[np.ndarray],np.ndarray]):
#         pass

 
#     def optimize(self,x0:np.ndarray):
#         pass

#     def opt(self,method:str):
#         pass
    
    
#     def vizulizeOuput(self):
#         pass

    
#     def save(self,folder_path:str):
#         pass




# class IDIMMLE(IDIM):
#     """
#     Inverse dynamics identification with maximum likelihood estimation.
#     Ref:
#         Fourier-based optimal excitation trajectories for the dynamic identification of robots
#         Kyung.Jo Park - Robotica - 2006.
#     """
#     def __init__(self) -> None:
#         self.max_eval = 100
#         self.lower_bound = -np.finfo(float).eps
#         self.upper_bound = sys.float_info.max
        
#     def set_lower_bounds(self,lb):
#         self.lower_bound = lb
    
#     def set_upper_bounds(self,ub):
#         self.upper_bound=ub
    
#     def set_maxeval(self,niter):
#         self.max_eval = niter
    
#     def set_set_min_objective(self, f: Callable[[np.ndarray], np.ndarray]):
#         return super().set_set_min_objective(f)


# # inverse dynamics identification using neural network :
# # use the xeights saved from ghali training model and simulate them 

# import logging
# import numpy as np
# from typing import Callable
# import matplotlib.pyplot as plt
# import seaborn as sns 
# from scipy.optimize import least_squares, curve_fit 

# from utils import  clampArray, plotArray, plot2Arrays
# from dynamapp.math_utils import RMSE, computeCorrelation

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class IDIMNLS:
#     """
#     Inverse Dynamics Identification Methods with Non Linear Least Square Alogrithms.
#     The identification problem is formulated in a non linear optimisation problem :
    
#         Xopt = argmin(X) ||IDM(q, qdot, qddot, X)- tau||
        
#     Args:
#         - nVars : number of optimization variables in the X vector.
#         - output : desired output vector ( Nsamples * ndof )
#         - identificationModel : function that return the model computed torques.
#             of shape : ( Nsamples * ndof )
#     """
#     def __init__(self,nVars,output,identificationModel: Callable[[np.ndarray],np.ndarray],\
#         upperBound=2,lowerBound=-2,time_step=0.001) -> None:
        
#         if np.ndim(output) != 2 :
#             logger.error("Target output should be 2 dimentional")
            

#         self.time_step = time_step
#         self.output = output
#         self.nVars = nVars
#         self.upperBound = upperBound
#         self.lowerBound = lowerBound
#         self.identificationModel  = identificationModel
#         self.optimized_params = None
    
#     def __str__(self) -> str:
#         return (f"IDIMNLS Model with {self.nVars} optimization variables,"
#                 f"output shape: {self.output.shape}, "
#                 f"time step: {self.time_step}")
    

#     def computeLsCostFunction(self, x:np.ndarray):
#         """
#         The object function to be minimized with least squares.
#         Returns:
#             - cost : (float)
#         """
#         if self.nVars < x.size :
#             xnew = np.concatenate([x[0:self.nVars], np.zeros(x.size-self.nVars)])
#             xnew = clampArray(xnew,self.lowerBound,self.upperBound)
#             tau_s = self.identificationModel(xnew)
#         elif self.nVars == x.size:
#             x = clampArray(x,self.lowerBound,self.upperBound)
#             tau_s = self.identificationModel(x)
#         else:
#             logger.error(\
#         'Identification Engine: Optimisation Variables should be <= input vector size.')
#         rmse = RMSE(self.output, tau_s) 
#         cost = np.mean(rmse**2)
#         return cost 
    
#     def computeRelativeError(self, x:np.ndarray=None):
#         if x is None:
#             tau_s = self.identificationModel(self.optimized_params)
#         else:
#             tau_s = self.identificationModel(x)
#         n = min(self.output.shape[0],self.output.shape[1])
#         relative_error =np.zeros(n)
#         for i in range(n):
#             relative_error[i] =np.where(self.output[:,i]!=0, \
#                 np.abs(tau_s[:,i] -self.output[:,i] )/np.abs(self.output[:,i]),np.inf)
#         return relative_error
    
#     def evaluate(self)->None:
#         """Evaluate the model's performance using the current parameters."""
#         if self.optimized_params is None:
#             logger.error("No optimized parameters found. Run optimize() first.")
#             return
#         tau_s = self.identificationModel(self.optimized_params)
#         rmse = RMSE(self.output, tau_s)
#         mean_rmse = np.mean(rmse)
#         logger.info(f"Evaluation result - Mean RMSE: {mean_rmse:.4f}")
    
#     def optimize(self, x0:np.ndarray=None, method='least_square', tol=1e-4):
#         """
#         Optimize the cost function with NLS alorithms to mimize it value.
#         Args:
#             - x0 : numpy-ndarry : initial paramters values estimation.
#             - method : optimisation algorithm
#             - tol : optimization alorithm error stop tolerence.
#         """
#         if x0 is None:
#             x0 = clampArray(abs(np.random.rand(self.nVars)),self.lowerBound,self.upperBound)
#         xOpt = x0
#         if method == 'least_square':
#             try:
#                 xOpt = least_squares(self.computeLsCostFunction, x0, xtol=tol, verbose=1)
#                 self.optimized_params = clampArray(xOpt.x,self.lowerBound,self.upperBound)
#                 xOpt = self.optimized_params
#             except Exception as e:
#                 logger.error(f"An error occurred during optimization: {e}")
#         elif method == 'curve_fit':
#             init_params = np.zeros(self.nVars)
#             try:
#                 x_data = np.linspace(0, len(self.output) * self.time_step,\
#                     len(self.output))
#                 popt, _ = curve_fit(self.identificationModel, x_data, \
#                     self.output , p0=init_params,method='trf')
#                 self.optimized_params = clampArray(popt.x,self.lowerBound,self.upperBound)
#                 xOpt = clampArray(popt.x,self.lowerBound,self.upperBound)
#             except Exception as e:
#                 logger.error(f"An error occurred during optimization: {e}")
#         else:
#             logger.error('Optimisation method Not Supported!')
#         return xOpt
    
#     def visualizeError(self,title=None, ylabel=None)->None:
#         """Plot the root squred error between simulated and inputs"""
#         if self.optimized_params is None:
#             logger.error(\
#        "Identification Engine: No optimized parameters found.Run optimize().")
#             return
#         tau_s = self.identificationModel(self.optimized_params)
#         rmse = RMSE(self.output,tau_s,1)
#         plotArray(rmse,title,ylabel)
        
#     def visualizeResults(self, title=None, y_label=None)->None:
#         """Plot the simulated and real signals in one figure."""
#         if self.optimized_params is None:
#             logger.error("No optimized parameters found. Run optimize() first.")
#         tau_s = self.identificationModel(self.optimized_params)
#         plot2Arrays(tau_s,self.output,'simultion','true',title)
        
#     def visualizeRelativeError(self,x:np.ndarray=None):
#         """Plot the bar diagram of each joint relative error"""
#         plt.figure(figsize=(12, 6))
#         n = min(self.output.shape[0],self.output.shape[1])
#         relative_error = self.computeRelativeError(x)
#         sns.barplot(x= np.ones_like(range(n)), y=relative_error)
#         plt.xlabel('Joint Index',fontsize=9)
#         plt.title('Relative Joints Error',fontsize=9)
        
#     def visualizeCostFunction(self,points_number:int=1500)->None:
#         """Plot the cost function scalar variation with respect to ||x||"""
#         xi = np.zeros(self.nVars)
#         xlist= [xi] * points_number
#         xnorm = [0] * points_number
#         ylist = [0] * points_number
#         for i in range(len(xlist)):
#             xi = np.random.uniform(self.lowerBound, self.upperBound, self.nVars)
#             ylist[i] = self.computeLsCostFunction(xi)
#             xnorm[i] = np.linalg.norm(xi)
#             xlist[i]= xi
#         plt.figure(figsize=(12, 6))
#         plt.scatter(xnorm,ylist,marker='.',s=4)
#         if not (self.optimized_params is None):
#             optnorm = np.linalg.norm(self.optimized_params)
#             yopt= self.computeLsCostFunction(self.optimized_params)
#             plt.scatter([optnorm], [yopt], color='red', marker='.',s=10)
        
#         plt.title('Cost Function vs Paramter Vector Norm',fontsize=9)
#         plt.xlabel("Norm2 Values")
#         plt.ylabel("Loss Values")
 
# import logging 
# import matplotlib.pyplot as plt  

# import dynamapp.model as model, dynamapp.regressors as regressors


# class IDIMOLS:
#     """ 
#     Inverse Dynamics Identification Method Ordiany Least Square.
#     this class valid only when the friction 
#     Args:
#         - 
#     """
#     def __init__(self,robot ) -> None:
#         pass
    
#     def computeLsCostFunction(self):
#         """ """
#         reg= regressors.Regressor()
#         cost = 0 
#         return cost 
    
# import numpy as np 

# class IDIMWLS:
#     """
    
    
#     """
#     def __init__(self) -> None:
#         pass





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