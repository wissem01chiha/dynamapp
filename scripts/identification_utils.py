import sys
import os
import matplotlib.pyplot as plt 
import numpy as np 
import logging

figureFolderPath = "/home/wissem/dynamic-identification/figure/kinova"
config_file_path = "/home/wissem/dynamic-identification/exemple/kinova/config.yml"

src_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '../src'))
sys.path.append(src_folder)

if not os.path.exists(figureFolderPath):
    os.makedirs(figureFolderPath)

from dynamapp import Robot, Regressor, StateSpace
from utils import RobotData, plot3Arrays, plot2Arrays, yaml2dict, RMSE, MAE

mlogger  = logging.getLogger('matplotlib')
logging.basicConfig(level='INFO')
mlogger.setLevel(logging.WARNING)

dynamics_logger = logging.getLogger('dynamics')
dynamics_logger.setLevel(logging.ERROR)

cutoff_frequency  = 3.5
config_params  = yaml2dict(config_file_path)
data           = RobotData(config_params['identification']['dataFilePath'])
fildata        = data.lowPassfilter(cutoff_frequency)
kinova         = Robot()
q_f            = fildata['position']
qp_f           = fildata['velocity']
qpp_f          = fildata['desiredAcceleration']
current_f      = fildata['current']
torque_f       = fildata['torque']
torque_cur_f   = fildata['torque_cur'] 
torque_rne_f   = fildata['torque_rne']
torque_rne     = data.torque_rne

q              = data.position
qp             = data.velocity
qpp            = data.desiredAcceleration
current        = data.current
torque         = data.torque

iteration_counter = 0
start = 0
end = 29000


def objective_function1(x, grad): 
    global kinova, iteration_counter, q_f, qp_f, qpp_f, torque_f
    kinova.setIdentificationModelData(q_f, qp_f, qpp_f)
    tau_sim = kinova.computeIdentificationModel(x)
    rmse_time  = RMSE(torque_f, tau_sim, axis=1)

    print(
        f"Iteration {iteration_counter}: "
        f"RMSE = {np.sqrt(np.mean(rmse_time**2)):.5f}"
    )
    iteration_counter += 1
    return np.sqrt(np.mean(rmse_time**2))
#######################################################################################
def objective_function2(x, grad): 
    global kinova, iteration_counter, q_f, qp_f, qpp_f, torque_f, torque_cur_f, torque_rne_f, start, end
    kinova.setIdentificationModelData(q_f[start:end,:], qp_f[start:end,:], qpp_f[start:end,:])
    tau_sim = kinova.computeIdentificationModel(x)
    rmse_time  = RMSE(torque_cur_f[start:end,:], tau_sim, axis=1)
    print(
        f"Iteration {iteration_counter}: "
        f"RMSE = {np.sqrt(np.mean(rmse_time**2)):.5f}"
    )
    iteration_counter += 1
    return np.sqrt(np.mean(rmse_time**2))

def validation(x):
    global kinova, q_f, qp_f, qpp_f, torque_f, figureFolderPath, torque_rne, torque_cur_f, start, end
    kinova.setIdentificationModelData(q_f[start:end,:],qp_f[start:end,:],qpp_f[start:end,:])
    tau_sim = kinova.computeIdentificationModel(x)
    rmse_time  = RMSE(torque_cur_f[start:end,:], tau_sim, axis=1)
    r = np.sqrt(np.mean(rmse_time**2))
    # rescale the torques values by the max values from datasehht that can the  [39 39 39 39 9 9 9]

    tau_sim[:,0]=1/39* tau_sim[:,0]
    torque_rne[:,0] = 1/39 * torque_rne[:,0]
    torque_cur_f[:,0] = 1/39 * torque_cur_f[:,0]

    tau_sim[:,1]=1/39* tau_sim[:,1]
    torque_rne[:,1] = 1/39 * torque_rne[:,1]
    torque_cur_f[:,1] = 1/39 * torque_cur_f[:,1]

    tau_sim[:,2]=1/39* tau_sim[:,2]
    torque_rne[:,2] = 1/39 * torque_rne[:,2]
    torque_cur_f[:,2] = 1/39 * torque_cur_f[:,2]

    tau_sim[:,3]=1/39* tau_sim[:,3]
    torque_rne[:,3] = 1/39 * torque_rne[:,3]
    torque_cur_f[:,3] = 1/39 * torque_cur_f[:,3]

    tau_sim[:,4]=1/9* tau_sim[:,4]
    torque_rne[:,4] = 1/9 * torque_rne[:,4]
    torque_cur_f[:,4] = 1/9 * torque_cur_f[:,4]

    tau_sim[:,5]=1/9* tau_sim[:,5]
    torque_rne[:,5] = 1/9 * torque_rne[:,5]
    torque_cur_f[:,5] = 1/9 * torque_cur_f[:,5]

    tau_sim[:,6]=1/9* tau_sim[:,6]
    torque_rne[:,6] = 1/9 * torque_rne[:,6]
    torque_cur_f[:,6] = 1/39 * torque_cur_f[:,6]
    # save the values of the torque simulted or comuted from the model to csv file 
    format = {'fmt': '%.4f', 'delimiter': ', ', 'newline': ',\n'}
    np.savetxt('/home/wissem/dynamic-identification/autogen/model_simulation_torques_current.csv', tau_sim, **format)
    torque_error = np.abs(torque_cur_f[start:end,:]- tau_sim)
    blast_torque_error =np.abs(torque_rne[start:end,:] - torque_cur_f[start:end,:])
    plot3Arrays(torque_cur_f[start:end,:],tau_sim,torque_rne[start:end,:],"current","simulation","blast",f"Manipulator Optimized Non Linear model NLopt-MaxNelder RMSE ={r}")
    plot2Arrays(torque_error,blast_torque_error,'simulation','blast', 'absloute error blast/new_model vs torque current')

    plt.savefig(os.path.join(figureFolderPath,'non_Linear_model_nlopt_best_poly_current'))
    plt.show()

