# Example notebooks

## Introductory examples

If you haven't already, check out the [tutorials](../tutorials/index.md), which explain how to build and simulate models in Collimator.

### [Primitive blocks and composability](primitives.ipynb)

Shows how to build systems with primitive blocks and how to compose them into larger diagrams.

### [Bouncing ball](bouncing_ball.ipynb)

Shows hybrid dynamics modeling of a bouncing ball.

### [Linear Quadratic Regulator (LQR)](lqr.ipynb)

Demonstrates the LQR for a pendulum and a planar quadrotor model.

### [Energy shaping and LQR stabilization](energy_shaping_and_lqr.ipynb)

Demonstrates energy shaping control to swing a pendulum to the vertically 'up' orientation and then stabilize it in the 'up' orientation via LQR.

### [Linear Model Predictive Control (MPC)](linear_mpc.ipynb)

Demonstrates MPC on a linearized model of the Cessna Citation aircraft and a pendulum model.

### [Multi-layer perceptron (MLP)](MLP_training.ipynb)

Demonstrates training of a multi-layer perceptron (MLP), a class of feedforward artificial neural networks, for a regression task.

### [Interacting with the Dashboard](interacting_with_the_dashboard.ipynb)

Demonstrates how to interact with Collimator's Dashboard to upload your local models, import Dashboard models and run simulations on the cloud.

## Advanced examples

### [Trajectory optimization and stabilization](trajectory_optimization_and_stabilization.ipynb)

Shows trajectory optimization for the problem of swinging an Acrobot to the vertically 'up' orientation and then stabilizing the trajectory via finite-horizon LQR.

### [Robotic arm control](mujoco/pick_and_place.ipynb)

Implement a controller for a "pick-and-place" task with a robotic arm using MuJoCo as a multibody physics engine.
Download the necessary files from [here](mujoco/assets/franka_emika_panda.zip).

### [Automatic tuning of a PID controller](pid_tuning.ipynb)

Demonstrates automatic differentiation and optimization capabilities of Collimator to automatically tune the gains of a discrete-time PID controller.

### [Finding limit cycles](limit_cycles.ipynb)

Demonstrates how to find limit cycles and assess their stability by leveraging the automatic differentiation capabilities of Collimator.

### [Kalman Filters: linear and nonlinear extensions](state_estimation_with_Kalman_filters.ipynb)

Demonstrates the use of Kalman filters (finite and infinite-horizon) and nonlinear extensions (Extended Kalman Filter and Unscented Kalman Filter) for state estimation in a pendulum model. Where necessary, the nonlinear Pendulum plant is automatically linearized and discretized by Collimator for the construction of the filters.

### [Universal Differential Equations (UDEs) and symbolic regression (SR)](ude_and_sr_lotka_volterra.ipynb)

Demonstrates training a Universal Differential Equation (UDE) to fit the observations produced by the Lotka-Volterra predator-prey system. Subsequently, the UDE is symbolically regressed to learn a closed-form model.

### [Nonlinear MPC](#3d-quadcopter-modeling-and-control)

See thematic series on modeling and control of 3D quadcopter [below](#3d-quadcopter-modeling-and-control), which showcases trajectory tracking by nonlinear MPC.

### [Submodels](dashboard_submodel.ipynb)

Demonstrates how to download a submodel defined in the Dashboard, incorporate it in a local model to run an optimization workflow and upload the result to the Dashboard.

## Thematic examples

### Battery modeling

1. [Equivalent circuit model (ECM) for a battery](part_1_battery_ecm_model.ipynb)
2. [ECM parameter estimation: synthetic data](part_2_parameter_estimation_synthetic_data.ipynb)
3. [ECM parameter estimation: experimental data](part_3_parameter_estimation_real_data.ipynb)
4. [Data-driven modeling: Dynamic Mode Decomposition (DMD)](part_4_data_driven_battery_models_DMDc.ipynb)
5. [Data-driven modeling: Extended DMD](part_5_data_driven_battery_models_eDMDc.ipynb)
6. [Data-driven modeling: SINDy with control](part_6_data_driven_battery_models_SINDyc.ipynb)
7. [Data-driven modeling: Neural Networks](part_7_data_driven_battery_models_Neural_Networks.ipynb)

<a id="nmpc"></a>

### 3D quadcopter modeling and control

1. [3D quadcopter modelling](01_quadcopter_modelling.ipynb)
2. [Trajectory generation through differentially flat outputs](02_quadcopter_trajectory_generation.ipynb)
3. [Control with nonlinear MPC](03_quadcopter_nonlinear_mpc.ipynb)

### Quanser Qube Servo hardware control

1. [Qube Servo modeling](quanser/01-plant-model.ipynb)
2. [Linear control](quanser/02-lqg.ipynb)
3. [Nonlinear swing-up control](quanser/03-energy-shaping.ipynb)
4. [Trajectory optimization](quanser/04-trajopt.ipynb)
5. [Neural network control](quanser/05-nn-control.ipynb)
