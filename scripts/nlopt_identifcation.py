import os  
import numpy as np
import nlopt 
import identification_utils as iu

initial_guess_path = "/home/wissem/dynamic-identification/autogen/initial_guess_nlopt_best_torque_current.npy"   

#####################################################################################
# optimisation routines 
#####################################################################################
# Initialize the optimizer
dim = 209  # Dimension of the input vector
max_iter = 3000
###################################################################################


opt = nlopt.opt(nlopt.LN_NEWUOA, dim) 
opt.set_initial_step([2.8] * dim)
# Set the objective function
opt.set_min_objective(iu.objective_function2)
# Set optimization parameters (optional)
opt.set_maxeval(max_iter)  
opt.set_ftol_rel(1e-7)    
opt.set_xtol_rel(1e-7)
# Define bounds if necessary (optional)
lower_bounds = np.full(dim,-1000)
upper_bounds = np.full(dim, 1000)
opt.set_lower_bounds(lower_bounds)
opt.set_upper_bounds(upper_bounds)
# Initial guess for the optimization
if os.path.exists(initial_guess_path):
    initial_guess = np.load(initial_guess_path)
    print("Loaded initial guess from file.")
else:
    initial_guess = np.random.rand(dim)
    print("Using random initial guess.")

try:
    x_opt = opt.optimize(initial_guess)
    min_value = opt.last_optimum_value()
    result_code = opt.last_optimize_result()
    print(f'Parameters values : {x_opt}')
    print(f'Minimum value of the objective function: {min_value}')
except KeyboardInterrupt:
    print("Optimization interrupted by user.")
    #print(f'Parameters values : {x_opt}')
    #np.save(initial_guess_path, x_opt)
    #print(f"Saved current optimized parameters to {initial_guess_path}.")

# Save the optimized vector for future use if not interrupted
if not np.isnan(x_opt).all():
    np.save(initial_guess_path, x_opt)
    print("Saved optimized parameters to file.")
else:
    print("Optimization did not produce a valid result.")
##############################################################################
# visulization
##############################################################################


iu.validation(x_opt)