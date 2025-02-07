import logging
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import figure as matplotlib_figure

from .trajectory import *
from .kalman import *
from .nfoursid import *
from .model_state import *
from .model import *
from .generators import *
from .jacobians import *
from .viscoelastic import *
from .identification import *
from .jacobians import *

class TrajectoryVisualizer:
    """
    A class for visualizing a custom computed trajectory.

    Args:
        - trajectory (Trajectory): An instance of a Trajectory subclass.
    """

    def __init__(self, trajectory):
        """
        Initializes the visualizer with a given trajectory.
        
        Args:
            trajectory (Trajectory): An instance of a Trajectory subclass.
        """
        self.trajectory = trajectory

    def plot(self, title="Trajectory Visualization", xlabel="Time (s)", ylabel="Position"):
        """
        Plots the computed trajectory.

        Args:
            - title (str): Title of the plot.
            - xlabel (str): Label for the x-axis.
            - ylabel (str): Label for the y-axis.
        """
        time = self.trajectory.time
        trajectory_values = self.trajectory.compute_full_trajectory()

        plt.figure(figsize=(10, 5))
        if trajectory_values.ndim == 1:
            plt.plot(time, trajectory_values, label="Trajectory", color="b")
        else:
            for i in range(trajectory_values.shape[1]):
                plt.plot(time, trajectory_values[:, i], label=f"DoF {i+1}")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
# def plot_filtered(self, fig: matplotlib_figure.Figure): 
#     """
#     The top graph plots the filtered output states of the Kalman filter and compares with the measured values.
#     The error bars correspond to the expected standard deviations.
#     The bottom graph zooms in on the errors between the filtered states and the measured values, compared with
#     the expected standard deviations.
#     """
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

#     df = self.to_dataframe()

#     top_legends, bottom_legends = [], []
#     for output_name in self.state_space.y_column_names:
#         actual_outputs = df[(output_name, self.actual_label, self.output_label)]
#         predicted_outputs = df[(output_name, self.filtered_label, self.output_label)]
#         std = df[(output_name, self.filtered_label, self.standard_deviation_label)]

#         markers, = ax1.plot(
#             list(range(len(actual_outputs))),
#             actual_outputs,
#             'x'
#         )
#         line, = ax1.plot(
#             list(range(len(actual_outputs))),
#             actual_outputs,
#             '-',
#             color=markers.get_color(),
#             alpha=.15
#         )
#         top_legends.append(((markers, line), output_name))

#         prediction_errorbar = ax1.errorbar(
#             list(range(len(predicted_outputs))),
#             predicted_outputs,
#             yerr=std,
#             marker='_',
#             alpha=.5,
#             color=markers.get_color(),
#             markersize=10,
#             linestyle='',
#             capsize=3
#         )
#         top_legends.append((prediction_errorbar, f'Filtered {output_name}'))

#         errors = actual_outputs - predicted_outputs
#         markers_bottom, = ax2.plot(
#             list(range(len(errors))),
#             errors,
#             'x'
#         )
#         lines_bottom, = ax2.plot(
#             list(range(len(errors))),
#             errors,
#             '-',
#             color=markers_bottom.get_color(),
#             alpha=.15
#         )
#         bottom_legends.append(((markers_bottom, lines_bottom), f'Error {output_name}'))
#         prediction_errorbar_bottom, = ax2.plot(
#             list(range(len(predicted_outputs))),
#             std,
#             '--',
#             alpha=.5,
#             color=markers.get_color(),
#         )
#         ax2.plot(
#             list(range(len(predicted_outputs))),
#             -std,
#             '--',
#             alpha=.5,
#             color=markers.get_color(),
#         )
#         bottom_legends.append((prediction_errorbar_bottom, rf'Filtered $\sigma(${output_name}$)$'))

#     lines, names = zip(*top_legends)
#     ax1.legend(lines, names, loc='upper left')
#     ax1.set_ylabel('Output $y$ (a.u.)')
#     ax1.grid()

#     lines, names = zip(*bottom_legends)
#     ax2.legend(lines, names, loc='upper left')
#     ax2.set_xlabel('Index')
#     ax2.set_ylabel(r'Filtering error $y-y_{\mathrm{filtered}}$ (a.u.)')
#     ax2.grid()
#     ax1.set_title('Kalman filter, filtered state')
#     plt.setp(ax1.get_xticklabels(), visible=False)

# def plot_predicted(
#         self,
#         fig: matplotlib_figure.Figure,
#         steps_to_extrapolate: int = 1
# ): 
#     """
#     The top graph plots the predicted output states of the Kalman filter and compares with the measured values.
#     The error bars correspond to the expected standard deviations.

#     The stars on the top right represent the ``steps_to_extrapolate``-steps ahead extrapolation under no further
#     inputs. The bottom graph zooms in on the errors between the predicted states and the measured values, compared
#     with the expected standard deviations.
#     """
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

#     df = self.to_dataframe()

#     extrapolation = self.extrapolate(steps_to_extrapolate)

#     top_legends, bottom_legends = [], []
#     for output_name in self.state_space.y_column_names:
#         actual_outputs = df[(output_name, self.actual_label, self.output_label)]
#         predicted_outputs = df[(output_name, self.next_predicted_corrected_label, self.output_label)]
#         std = df[(output_name, self.next_predicted_label, self.standard_deviation_label)]
#         last_predicted_std = std.iloc[-1]

#         markers, = ax1.plot(
#             list(range(len(actual_outputs))),
#             actual_outputs,
#             'x'
#         )
#         line, = ax1.plot(
#             list(range(len(actual_outputs))),
#             actual_outputs,
#             '-',
#             color=markers.get_color(),
#             alpha=.15
#         )
#         top_legends.append(((markers, line), output_name))

#         prediction_errorbar = ax1.errorbar(
#             list(range(1, len(predicted_outputs)+1)),
#             predicted_outputs,
#             yerr=std,
#             marker='_',
#             alpha=.5,
#             color=markers.get_color(),
#             markersize=10,
#             linestyle='',
#             capsize=3
#         )
#         top_legends.append((prediction_errorbar, f'Predicted {output_name}'))
#         extrapolation_errorbar = ax1.errorbar(
#             list(range(len(self.ys), len(self.ys) + steps_to_extrapolate)),
#             extrapolation[output_name].to_numpy(),
#             yerr=last_predicted_std,
#             marker='*',
#             markersize=9,
#             alpha=.8,
#             color=markers.get_color(),
#             linestyle='',
#             capsize=3
#         )
#         top_legends.append((extrapolation_errorbar, f'Extrapolation {output_name} (no input)'))

#         errors = actual_outputs.to_numpy()[1:] - predicted_outputs.to_numpy()[:-1]
#         markers_bottom, = ax2.plot(
#             list(range(1, len(errors)+1)),
#             errors,
#             'x'
#         )
#         lines_bottom, = ax2.plot(
#             list(range(1, len(errors)+1)),
#             errors,
#             '-',
#             color=markers_bottom.get_color(),
#             alpha=.15
#         )
#         bottom_legends.append(((markers_bottom, lines_bottom), f'Error {output_name}'))
#         prediction_errorbar_bottom, = ax2.plot(
#             list(range(1, len(predicted_outputs))),
#             std[:-1],
#             '--',
#             alpha=.5,
#             color=markers.get_color(),
#         )
#         ax2.plot(
#             list(range(1, len(predicted_outputs))),
#             -std[:-1],
#             '--',
#             alpha=.5,
#             color=markers.get_color(),
#         )
#         bottom_legends.append((prediction_errorbar_bottom, rf'Predicted $\sigma(${output_name}$)$'))

#     lines, names = zip(*top_legends)
#     ax1.legend(lines, names, loc='upper left')
#     ax1.set_ylabel('Output $y$ (a.u.)')
#     ax1.grid()

#     lines, names = zip(*bottom_legends)
#     ax2.legend(lines, names, loc='upper left')
#     ax2.set_xlabel('Index')
#     ax2.set_ylabel(r'Prediction error $y-y_{\mathrm{predicted}}$ (a.u.)')
#     ax2.grid()
#     ax1.set_title('Kalman filter, predicted state')
#     plt.setp(ax1.get_xticklabels(), visible=False)
    
# def plot_eigenvalues(self, ax: plt.axes):   
#     """
#     Plot the eigenvalues of the :math:`R_{32}` matrix, so that the order of the state-space model can be determined.
#     Since the :math:`R_{32}` matrix should have been calculated, this function can only be used after
#     performing ``self.subspace_identification``.
#     """
#     if self.R32_decomposition is None:
#         raise Exception('Perform subspace identification first.')

#     ax.semilogy(jnp.diagonal(self.R32_decomposition.eigenvalues), 'x')
#     ax.set_title('Estimated observability matrix decomposition')
#     ax.set_xlabel('Index')
#     ax.set_ylabel('Eigenvalue')
#     ax.grid()


# def plot_input_output(self, fig: matplotlib_figure.Figure):   
#     """
#     Given a matplotlib figure ``fig``, plot the inputs and outputs of the state-space model.
#     """
#     ax1 = fig.add_subplot(2, 1, 1)
#     ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

#     for output_name, outputs in zip(self.y_column_names, np.array(self.ys).squeeze(axis=2).T):
#         ax1.plot(outputs, label=output_name, alpha=.6)
#     ax1.legend(loc='upper right')
#     ax1.set_ylabel('Output $y$ (a.u.)')
#     ax1.grid()

#     for input_name, inputs in zip(self.u_column_names, np.array(self.us).squeeze(axis=2).T):
#         ax2.plot(inputs, label=input_name, alpha=.6)
#     ax2.legend(loc='upper right')
#     ax2.set_ylabel('Input $u$ (a.u.)')
#     ax2.set_xlabel('Index')
#     ax2.grid()

#     ax1.set_title('Inputs and outputs of state-space model')
#     plt.setp(ax1.get_xticklabels(), visible=False)
    

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