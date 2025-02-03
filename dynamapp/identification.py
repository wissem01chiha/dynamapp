from .nfoursid import *
from .generators import *
from .regressors import *
from .reductions import *
from .kalman import *
from .solvers import *
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


