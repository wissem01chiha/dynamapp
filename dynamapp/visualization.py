import os
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .trajectory import *
from .model_state import *
from .model import *
from .data_utils import *


    

# def plot(self)->None:
#     sns.set(style="whitegrid")
#     fig, axes = plt.subplots(3, 3, figsize=(12, 6))
#     for i in range(7):
#         ax = axes[i // 3, i % 3]
#         sns.lineplot(ax=ax, x=np.arange(len(self.current[:, i])),\
#             y=self.current[:, i],linewidth=0.5)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        
        
# def visualizeTrajectory(self, ti,tf, q0=None, qp0=None, qpp0=None, savePath=None):
#     """Compute and plot a given trajectory, each variable in a single plot"""

#     q, qp, qpp = self.computeFullTrajectory(ti,tf,q0,qp0,qpp0)
#     plotArray(q,'Computed Trajectory Joints Positions')
#     plt.savefig(os.path.join(savePath,'computed_trajectory_positions'))
#     plotArray(qp,'Computed Trajectory Joints Velocity')
#     plt.savefig(os.path.join(savePath,'computed_trajectory_velocity'))
#     plotArray(qpp,'Computed Trajectory Joints Accelerations')
#     if not(savePath is None): 
#         plt.savefig(os.path.join(savePath,'computed_trajectory_accelaertions'))
            
#     def save2csv(self,ti:float,tf:float,file_path,q0=None,qp0=None,qpp0=None):
#         """ compute a given trajectory and save it to csv file.""" 

#         q, qp, qpp = self.computeFullTrajectory(ti,tf,q0,qp0,qpp0)
#         format = {'fmt': '%.2f', 'delimiter': ', ', 'newline': ',\n'}
#         traj_array = np.concatenate([q,qp,qpp])
#         np.savetxt(file_path, traj_array, **format)


#    def visualizeTrajectory(self):
#         time_slot = np.linspace(
#             0.0, (np.sum(self.Nf) + 1) * self.ts, num=(np.sum(self.Nf) + 1)
#         )
#         fig, axs = plt.subplots(3, 1)
#         for i in range(self.njoints):
#             axs[0].plot(time_slot, self.q[i])
#             axs[0].set_ylabel("q")
#             axs[1].plot(time_slot, self.qd[i])
#             axs[1].set_ylabel("qd")
#             axs[2].plot(time_slot, self.qdd[i])
#             axs[2].set_ylabel("qdd")
#             axs[2].set_xlabel("Time(s)")
#         x = [0, 10, 20, 30]
#         for j in range(3):
#             for xc in x:
#                 axs[j].axvline(x=xc, color="black", linestyle="dashed")
#         plt.show()
        
        
#     def visualizeTrajectory(self, ti, tf, Q0=None, Qp0=None, Qpp0=None):
#         """Visualizes the computed trajectory."""
#         q, qp, qpp = self.computeFullTrajectory(ti, tf, Q0, Qp0, Qpp0)
#         plotArray(q, 'Computed Trajectory Joints Positions')
#         plotArray(qp, 'Computed Trajectory Joints Velocities')
#         plotArray(qpp, 'Computed Trajectory Joints Accelerations')
    
#     def saveTrajectory2file(self, filename='trajectory.txt'):
#         """Saves the computed trajectory to a file."""
#         q, qp, qpp = self.computeFullTrajectory(0, 1)  # Assuming some default time range
#         with open(filename, 'w') as f:
#             for i in range(len(q)):
#                 f.write(f"{q[i]}\t{qp[i]}\t{qpp[i]}\n")
#         logger.info(f"Trajectory saved to {filename}")

    
#     def visualizeTrajectoryIdentifiability(self, ti, tf, torque, q, qp, qpp, x):    
#         """
#         Visualize the trajectory identifiability criteria
        
#         modeled by the probobilty function:
#         f(x) = P( abs((ε - εm)/σε) < 0.05 || abs((J - Jm)/σJ) < x)  where x > 0
#         """
#         J, eps = self.computeTrajectoryIdentifiability(ti, tf, torque, q, qp, qpp, x)
#         J_n = np.abs((J-np.mean(J))/np.std(J))
#         eps_n =np.abs((eps-np.mean(eps))/np.std(eps))
        
#         thresh = np.linspace(0,3,40)
#         P = np.zeros_like(thresh)
#         for i in range(len(thresh)):
#             P[i] = self.computeProba(eps_n,J_n,thresh[i])
        
#         plt.figure(figsize=(12, 6))
#         sns.lineplot(x=thresh, y= P,label='Propab vs thresh',linewidth=0.7)
#         plt.title('Trajectory Identifiability')
#         plt.xlabel('δ')
#         plt.ylabel('P')
#         plt.legend()
        

#     def computeTrajectoryIdentifiability(self,ti,tf,torque,q,qp,qpp,x):
#         """
#         Evaluate the regression criteria ε(qi, qpi, qppi, x) based on a given trajectory.

#         This function computes the regression criteria for a fixed system parameter vector `x`
#         and a trajectory `C(qi, qpi, qppi)` computed previously. The trajectory is considered
#         identifiable if the criteria is minimal for most of the time steps.

#         Args:
#             qi: Generalized position coordinates of the trajectory.
#             qpi: Generalized velocity coordinates of the trajectory.
#             qppi: Generalized acceleration coordinates of the trajectory.
#             x: Fixed system parameter vector.

#         Returns:
#             Regression criteria value ε for the given trajectory and system parameters.

#         Notes:
#             - The trajectory is identifiable if ε remains minimal over most time steps.
#             - TODO:
#              Investigate the relationship between the regression criteria evolution and
#             the trajectory `C` over time `t`.
#         """
#         reg = Regressor()
#         N = len(q)
#         J = np.empty(N)
#         eps = np.empty(N)
#         for i in range(N):
#             J[i] = self.computeTrajectoryCriterion(ti,tf,q[i,:],qp[i,:],qpp[i,:])
#             eps[i] = reg.computeRegressionCriterion(torque[i,:],q[i,:],qp[i,:],qpp[i,:],x)
#         return  J, eps
    


# def plot_array(array: np.ndarray,title=None,ylabel = None) -> None:
#     """
#     Given an ( n * m )  data array where n >> m, plot each coloum data 
#     in sperate subplots .

#     Args:
#         - array: numpy ndarray
#     """
#     N = array.shape[0]
#     if array.ndim ==1 :
#         fig = plt.figure(figsize=(12, 6))
#         ax = fig.add_subplot(111)
#         sns.lineplot(ax=ax, x=np.arange(N), y=array, linewidth=0.5, color='blue')
#         ax.set_xlabel("Time (ms)", fontsize=9)
#         if not(ylabel is None):
#             plt.ylabel(ylabel, fontsize=9)
#     elif array.ndim == 2:
#         ndof = min(array.shape[1],array.shape[0])
#         if not(ndof == array.shape[1]):
#             array = np.transpose(array)
#         fig, axes = plt.subplots(3, 3, figsize=(12, 6), dpi=100)
#         axes = axes.flatten()
#         for i in range(ndof):
#             ax = axes[i]
#             sns.lineplot(ax=ax, x=np.arange(N), y=array[:, i], linewidth=0.5,color='blue')
#             ax.set_xlabel("Time (ms)", fontsize=9)
#             if not(ylabel is None):
#                 ax.set_ylabel(ylabel, fontsize=9)
#             ax.set_title(f'Joint {i+1}', fontsize=9)

#         for j in range(ndof, len(axes)):
#             fig.delaxes(axes[j])
            
#     if title != None: 
#         fig.suptitle(title, fontsize=9)   
#     plt.tight_layout()
        
         
# def plot_arrays(array1: np.ndarray, array2: np.ndarray, legend1=None, legend2=None,title=None,
#                color1='red', color2='blue') -> None:
#     """
#     Given two (n * m) data arrays where n >> m, plot each column data
#     from both arrays in separate subplots. 
#     """
#     assert array1.shape == array2.shape, "Arrays should have the same shapes."
#     ndof = min(array1.shape[1],array1.shape[0])
#     if ndof == array1.shape[1]:
#         N = array1.shape[0]
#     else:
#         N = array1.shape[1]
#     fig, axes = plt.subplots(3, 3, figsize=(12, 6), dpi=100)
#     axes = axes.flatten()
    
#     for i in range(ndof):
#         ax = axes[i]
#         sns.lineplot(ax=ax, x=np.arange(N), y=array1[:, i], linewidth=0.5, color=color1, label=legend1)
#         sns.lineplot(ax=ax, x=np.arange(N), y=array2[:, i], linewidth=0.5, color=color2, label=legend2)
#         ax.set_xlabel("Time (ms)", fontsize = 9)
#         ax.set_title(f'Joint {i+1}', fontsize = 9)
#         ax.grid(True)
#         if legend1 or legend2:
#             ax.legend(fontsize = 6)
    
#     for j in range(ndof, len(axes)):
#         fig.delaxes(axes[j])
#     if title:
#         fig.suptitle(title, fontsize=9)
#     plt.tight_layout()
    
# def plot_arrays(array1: np.ndarray, array2: np.ndarray, array3: np.ndarray, 
#                     legend1=None, legend2=None, legend3=None, 
#                     title=None, color1='red', color2='blue', color3='green') -> None:
#     """
#     Given three (n * m) data arrays where n >> m, plot each column data
#     from all arrays in separate subplots. 
#     """
#     ndof = array1.shape[1]
#     N = array1.shape[0]
#     fig, axes = plt.subplots(3, 3, figsize=(12, 6), dpi=100)
#     axes = axes.flatten()
    
#     for i in range(ndof):
#         ax = axes[i]
#         sns.lineplot(ax=ax, x=np.arange(N), y=array1[:, i], linewidth=0.5, color=color1, label=legend1)
#         sns.lineplot(ax=ax, x=np.arange(N), y=array2[:, i], linewidth=0.5, color=color2, label=legend2)
#         sns.lineplot(ax=ax, x=np.arange(N), y=array3[:, i], linewidth=0.5, color=color3, label=legend3)
#         ax.set_xlabel("Time (ms)", fontsize=9)
#         ax.set_title(f'Joint {i+1}', fontsize=9)
#         ax.grid(True)
#         if legend1 or legend2 or legend3:
#             ax.legend(fontsize=6)
         
#     for j in range(ndof, len(axes)):
#         fig.delaxes(axes[j])
#     if title:
#         fig.suptitle(title, fontsize=9)
#     plt.tight_layout()
    
# def plot_element_wise_array(array:np.ndarray, title=None,xlabel=None,ylabel= None):
#     plt.figure(figsize=(12,6))
#     sns.barplot(x= np.ones_like(range(len(array)))+range(len(array)),y=array)
#     if not(xlabel is None):
#         plt.xlabel(xlabel,fontsize=9)
#     if not(ylabel is None): 
#         plt.ylabel(ylabel,fontsize=9)
#     if not(title is None):
#         plt.title(title,fontsize=9)
        
#     def visualizeRootLocus(self, q_or_x:np.ndarray,qp:np.ndarray=None)->None:
#         """ Plot the system root locus for a given trajectory."""
#         gain = np.linspace(0, 0.45, 35)
#         K = np.ones((7, 14))
#         A, B,_,_ = self.computeStateMatrices(q_or_x,qp)
#         poles = np.array([np.linalg.eigvals(A-np.dot(B,K*k)) for k in gain])
#         for j in range(poles.shape[1]):
#             plt.plot(np.real(poles[:,j]), np.imag(poles[:,j]), '.', markersize=3)
#         plt.title('Root Locus Plot')
#         plt.xlabel('Real')
#         plt.ylabel('Imaginary')
#         plt.axhline(0, color='black', linewidth=0.5)
#         plt.axvline(0, color='black', linewidth=0.5)
#         plt.grid(True)
         
#     def visualizeStatePoles(self,q,qp):
#         """Plot the system poles for a given trajectory"""
#         eigenvalues = self.getStateEigvals(q,qp)
#         plt.figure(figsize=(6, 4))
#         plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), marker='*', color='blue')
#         plt.axhline(0, color='black', linewidth=0.5)
#         plt.axvline(0, color='black', linewidth=0.5)
#         plt.xlabel('Real')
#         plt.ylabel('Imaginary')
#         plt.title('Pole Plot')
#         plt.grid(True)