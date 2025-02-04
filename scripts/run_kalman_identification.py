"""
kalman filter based identification of the robot state space system 
estimate : the default sytem --> velocity and poistion x(t)
estimate the augmented system state -> z(t) = [x(t) τ(t)]-> estimate the torque 

use the robust kalman for the fully state space estimation of evlocity ,
y=use the robust to the augmented 
use the simple filter kalmn for the linear model 
use the simple kalmn filter for the linear aumented model 

Note: 
P0  (Initial State Covariance): This matrix represents the initial uncertainty in the state estimate.
A higher value indicates more initial uncertainty.

Q (Process Noise Covariance): This matrix represents the uncertainty in the process model.
A higher value indicates that you trust the model less and the measurements more.

R (Measurement Noise Covariance): This matrix represents the uncertainty in the measurements. 
A higher value indicates that you trust the measurements less and the model more.
"""
import os
import sys
import argparse 
import numpy as np
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from identification import Kalman, RobustKalman
from dynamics import StateSpace, Robot
from utils import RobotData, plot2Arrays, plot3Arrays,  yaml2dict, smooth_columns, RMSE

base_dir = os.getcwd()
figure_path      = os.path.join(base_dir ,"pyDynaMapp/figure/kinova") 
config_file_path = os.path.join(base_dir,"pyDynaMapp/robot/kinova/config.yml")
state_poles_path = os.path.join(base_dir,"pyDynaMapp/autogen/state_poles.npy")  
data_file_path   = os.path.join(base_dir,"pyDynaMapp/data/kinova/identification_data/blast_traj.csv")
urdf_file_path   = os.path.join(base_dir,"pyDynaMapp/robot/kinova/gen3.urdf")

parser = argparse.ArgumentParser()
parser.add_argument('--v',type=bool,default=False)
parser.add_argument('--cutoff_frequency', type=float, default=3)
parser.add_argument('--show_figures', type=bool,default=False)
args = parser.parse_args()

cutoff_frequency = args.cutoff_frequency
config_params  = yaml2dict(config_file_path)
data           = RobotData(data_file_path)
q              = data.position
qp             = data.velocity
qpp            = data.desiredAcceleration
torque_cur     = data.torque_cur

fildata        = data.lowPassfilter(cutoff_frequency)
q_f            = fildata['position']
qp_f           = fildata['velocity']
qpp_f          = fildata['desiredAcceleration']
torque_f       = fildata['torque']
torque_cur_f   = fildata['torque_cur']

kinova         = Robot(urdf_file_path,config_file_path)
kinova_ss      = StateSpace(urdf_file_path,config_file_path)

ndof = kinova.model.nq
num_samples = data.numRows

A = np.zeros((2*ndof,2*ndof*num_samples))
B = np.zeros((2*ndof,ndof*num_samples))
H = np.zeros((2*ndof,2*ndof*num_samples))
P0 = np.zeros((2*ndof,2*ndof*num_samples))
R  = np.zeros((2*ndof,num_samples))
Q = np.zeros((2*ndof,2*ndof*num_samples))


start = 1
step = 100
stop = 29106

for i in range(start, stop,step):
    print(f'process sample = {i}/{num_samples}')
    Ai, Bi, _, _ = kinova_ss.computeStateMatrices(q_f[i, :], qp_f[i, :])
    
    if np.any(np.isnan(Ai)) or np.any(np.isnan(Bi)) or np.any(np.isinf(Ai)) or np.any(np.isinf(Bi)):
        print(f"Warning: NaN or Inf detected in Ai or Bi at sample {i}")
        Ai = np.nan_to_num(Ai)  
        Bi = np.nan_to_num(Bi)
    
    if np.any(np.abs(Ai) > 1e7) or np.any(np.abs(Bi) > 1e7):
        print(f"Warning: Large values detected in Ai or Bi at sample {i}")
    
    A[:, (i-1)*2*ndof:i*2*ndof] = Ai
    B[:, (i-1)*ndof:i*ndof] = Bi
    H[:, (i-1)*2*ndof:i*2*ndof] =  np.eye(2*ndof)
    Q[:, (i-1)*2*ndof:i*2*ndof] = np.diag([0.9] * 2*ndof)   
    R[:, i] = 0.45 * np.ones(2*ndof)   
    P0[:, (i-1)*2*ndof:i*2*ndof] = np.diag([0.1] * 2*ndof)
    
x0 = kinova_ss.getStateVector(qp_f[start,:],q_f[start,:])                            
sim_states = np.zeros((2*ndof,num_samples))
kf = RobustKalman(A, B, H, Q, R, P0, x0)
states_error = []
for k in range(start, stop,step):
    u = 1/40*data.torque[k,:] # 39 is the max torque give by any jont motor extraced form data sheeet is 39
    z = kinova_ss.getStateVector(qp[k,:],q[k,:])
    x, _ = kf.step(u, z, k)
    sim_states[:,k] = x
    states_error.append(np.sqrt(np.mean((x-z)**2))/np.sqrt(np.mean((z)**2)))
    print(f"Time {k}:, state relative RMSE {np.sqrt(np.mean((x-z)**2))/np.sqrt(np.mean((z)**2))}")

sim_vel = sim_states[7:14, start:stop:step]
true_vel= qp[start:stop:step,:]
scaled_sim_vel = sim_vel
scaled_sim_vel[4:7,:] = 1/5*scaled_sim_vel[4:7,:]
scaled_sim_vel[:4,:] = 1/1.3899*scaled_sim_vel[:4,:]
states_error = np.array(states_error)
avg_error = np.mean(states_error)
window_size = 10
avg_error_per_window = np.convolve(states_error, np.ones(window_size)/window_size, mode='valid')
filt_scaled_sim_vel =  smooth_columns(scaled_sim_vel.T)
global_rmse = RMSE(filt_scaled_sim_vel,true_vel)
rmse_moy =np.mean(global_rmse)
# plotting routines 
time_ms = np.arange(start, stop, step)
plt.figure(figsize=(10, 6))
plt.plot(time_ms, states_error, label='RMSE', color='blue')
plt.axhline(avg_error, color='red', linestyle='--', label='Average Error')
plt.plot(time_ms[window_size-1:], avg_error_per_window, color='green', linestyle='-.', label=f'Average Error (Window Size {window_size})')


plt.xlabel('Time (ms)')
plt.ylabel('RMSE')
plt.title('Robust kalman Filter RMSE over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot3Arrays(scaled_sim_vel.T,true_vel,filt_scaled_sim_vel ,'sim','true','filt',\
f'Robust kalman Filter Identification, cutoff={cutoff_frequency}, sampling ={data.samplingRate}, Mean RMSE={rmse_moy}')
plt.show()
