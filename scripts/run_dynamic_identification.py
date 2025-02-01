"""
this files used for the 
following script puropose running: 
"""
import os 
import sys
import argparse
import numpy as np 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynamics.robot import Robot
from dynamics.regressor import Regressor
from utils import RobotData, plot2Arrays, plot3Arrays,  yaml2dict, MAE

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

ndof = kinova.model.nq
num_samples = data.numRows

# run ordiany least square identifcation for the linear of the qusqi linear system 
reg = Regressor(kinova)
x = np.random.rand(reg.param_vector_max_size)
W = reg.computeFullRegressor(q_f,qp_f,qpp_f) 
tau_sim = (W @ x).reshape((torque.shape[0],kinova.model.nq))