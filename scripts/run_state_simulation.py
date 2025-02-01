"""
in this script :
simulate the normale staete system, the augmented with state dpen pole plcement




"""
import os
import sys
import argparse
import numpy as np 
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt 
import logging
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import RobotData, plot2Arrays, plotArray, yaml2dict
from dynamics import Robot, StateSpace

parser = argparse.ArgumentParser(description=\
'simulate manipulator state space models with default paramters')

parser.add_argument('--v',type=bool,default=False)
parser.add_argument('--cutoff_frequency', type=float, default=3)
parser.add_argument('--show_figures', type=bool,default=False)
parser.add_argument('--data_file',type=str,default='blast_traj.csv')
parser.add_argument('--filter_output',type=bool,default=False)
args = parser.parse_args()

base_dir = os.getcwd()
figure_path      = os.path.join(base_dir ,"pyDynaMapp/figure/kinova") 
config_file_path = os.path.join(base_dir,"pyDynaMapp/robot/kinova/config.yml")
state_poles_path = os.path.join(base_dir,"pyDynaMapp/autogen/state_poles.npy")  
data_file_path   = os.path.join(base_dir,"pyDynaMapp/data/kinova/identification_data/blast_traj.csv")
urdf_file_path   =  os.path.join(base_dir,"pyDynaMapp/robot/kinova/gen3.urdf")

st = time.time()
logger = logging.getLogger(__name__)
if not(args.v):
    dynamics_logger = logging.getLogger('dynamics')
    dynamics_logger.setLevel(logging.ERROR)

cutoff_frequency = args.cutoff_frequency
config_params  = yaml2dict(config_file_path)
data           = RobotData(data_file_path)
fildata        = data.lowPassfilter(cutoff_frequency)
kinova         = Robot(urdf_file_path,config_file_path)
q_f            = fildata['position']
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
tau_ss    = torque
x0        = kinova_ss.getStateVector(qp_f[0,:],q_f[0,:])
start     = 0
step      = 1
end       = 300
 
states = kinova_ss.lsim(x0=x0,input=tau_ss[start:end:step,:],verbose=True)
nyquist_freq = 0.5 * data.samplingRate
 
if args.filter_output:
    normal_cutoff = cutoff_frequency / nyquist_freq
    b, a = butter(3, normal_cutoff, btype='low', analog=False)
    states =  filtfilt(b,a,states,axis=1)
    
#states[7:14,0:10]=0
# compute the error between the actual state velocity and the real ones :
vel_error = np.abs(np.transpose(states[7:14,:])-qp_f[start:end:step,:])
plotArray(vel_error,'absolute error velocity')

plot2Arrays(0.001*np.transpose(states[7:14,:]),qp_f[start:end:step,:],'state','true',\
f'Joints Velocity State Model Simulation, window = {step}, cutoff = {cutoff_frequency}, sampling ={data.samplingRate}')
plt.savefig(os.path.join(figure_path,'joints velocity state model simulation'))
if args.v :
    et = time.time()-st
    logger.info(f'elapsed time : {et}')

if args.show_figures:
    plt.show()
