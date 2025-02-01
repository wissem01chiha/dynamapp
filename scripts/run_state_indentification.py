"""

in this script ---
kalman identification based on the augmented system 
lumberger identification based using state poles placement with window avarging 
LMI identiifcation on the 
"""
import argparse
import sys
import os
import matplotlib.pyplot as plt 
import numpy as np 
import nlopt
import logging
from scipy.signal import butter, filtfilt

parser = argparse.ArgumentParser(description=\
'simulate manipulator state space models with default paramters')

parser.add_argument('--v',type=bool)
parser.add_argument('--cutoff_frequency', type=float, default=3)
parser.add_argument('--show_figures', type=bool,default=False)
parser.add_argument('--data_file',type=str,default='b')
parser.add_argument('--iteration',type=int,default=10)
args = parser.parse_args()

base_dir = os.getcwd()
figure_path      = os.path.join(base_dir ,"pyDynaMapp/figure/kinova") 
config_file_path = os.path.join(base_dir,"pyDynaMapp/robot/kinova/config.yml")
state_poles_path = os.path.join(base_dir,"pyDynaMapp/autogen/state_poles.npy")  
data_file_path   = os.path.join(base_dir,"pyDynaMapp/data/kinova/identification_data/blast_traj.csv")
urdf_file_path   =  os.path.join(base_dir,"pyDynaMapp/robot/kinova/gen3.urdf")

from dynamics.robot import Robot
from dynamics.state_space import  StateSpace
from utils import RobotData,  plot2Arrays, yaml2dict, RMSE

dynamics_logger = logging.getLogger('dynamics')
dynamics_logger.setLevel(logging.ERROR)

mlogger  = logging.getLogger('matplotlib')
logging.basicConfig(level='INFO')
mlogger.setLevel(logging.WARNING)

cutoff_frequency  = args.cutoff_frequency
config_params  = yaml2dict(config_file_path)
data           = RobotData(data_file_path)
fildata        = data.lowPassfilter(cutoff_frequency)
kinova         = Robot(urdf_file_path,config_file_path)
q_f            = fildata ['position']
qp_f           = fildata['velocity']
qpp_f          = fildata['desiredAcceleration']
current_f      = fildata['current']
torque_f       = fildata['torque']
q              = data.position
qp             = data.velocity
qpp            = data.desiredAcceleration
current        = data.current
torque         = data.torque

kinova_ss = StateSpace(urdf_file_path,config_file_path)
tau_ss = torque


start = 0
step  = 2000
end = 30000
x0 = kinova_ss.getStateVector(qp_f[start,:],q_f[start,:])
max_iter = args.iteration
dim = 14

###################################################################
# define optimisation routines 
iteration_counter = 0
def optimize_poles(x,grad):
    """ given 3 parmters of polynme it compuytes the state-dependent system poles 
    like :
    k(x(t)) = α_0 + α_1 x(t) + α_2x(t)^2+ α_3 x(t)^3   
    """
    global kinova_ss, tau_ss, x0, iteration_counter, qp_f
    states = kinova_ss.simulate(x0,tau_ss[start:end:step],verbose=False,system_poles=x)
    rmse_time  = RMSE(np.transpose(states[7:14,:]), qp_f[start:end:step], axis=1)
    print(
        f"Iteration {iteration_counter}/{max_iter}: "
        f"RMSE = {np.sqrt(np.mean(rmse_time**2)):.5f}"
    )
    iteration_counter +=1
    return np.sqrt(np.mean(rmse_time**2))
    



opt = nlopt.opt(nlopt.LN_NEWUOA, dim) 
opt.set_min_objective(optimize_poles)
opt.set_maxeval(max_iter)  
opt.set_ftol_rel(1e-9)     
opt.set_xtol_rel(1e-9)

# Define bounds if necessary (optional)
lower_bounds = np.full(dim, -1)
upper_bounds = np.full(dim, 0)
opt.set_lower_bounds(lower_bounds)
opt.set_upper_bounds(upper_bounds)

# Initial guess for the optimization
if os.path.exists(state_poles_path):
    initial_guess = np.load(state_poles_path)
    print("Loaded initial guess from file.")
else:
    initial_guess = -np.abs(np.random.rand(dim))
    print("Using random initial guess.")
    
x_opt = opt.optimize(initial_guess)
min_value = opt.last_optimum_value()
result_code = opt.last_optimize_result()
print(f'paramters values : {x_opt}')
print(f'minimum value de la fonction objective: {min_value}')

# Save the optimized vector for future use 
np.save(state_poles_path, x_opt)
print("Saved optimized parameters to file.")
####################################################################################
# validation 
print('validation process : ...') 
states = kinova_ss.simulate(x0,tau_ss[start:end:step,:],verbose=True,system_poles=x_opt)
rmse_time  = RMSE(np.transpose(states[7:14,:]), qp_f[start:end:step], axis=1)
 

plot2Arrays(np.transpose(states[7:14,:]), qp_f[start:end:step,:],'state','true',\
    f'Joints Velocity State Model Simulation, RMSE =  {np.sqrt(np.mean(rmse_time**2)):.5f}')
plt.savefig(os.path.join(figure_path,'joints_velocity_state_model_simulation'))
 
if args.show_figures:
    plt.show()


###################################################
# run kalman filtering estimation on the augmented system dynamics
###################################################