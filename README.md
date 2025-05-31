DynaMapp 
======  

 
[![PyLint](https://github.com/wissem01chiha/dynamapp/actions/workflows/pylint.yml/badge.svg)](https://github.com/wissem01chiha/dynamapp/actions/workflows/pylint.yml)
[![Deploy static content to Pages](https://github.com/wissem01chiha/dynamapp/actions/workflows/static.yml/badge.svg)](https://github.com/wissem01chiha/dynamapp/actions/workflows/static.yml)
[![Version](https://anaconda.org/chiha/pydynamapp/badges/version.svg)](https://anaconda.org/chiha/pydynamapp)
[![Release Date](https://anaconda.org/chiha/pydynamapp/badges/latest_release_date.svg)](https://anaconda.org/chiha/pydynamapp)
[![Platforms](https://anaconda.org/chiha/pydynamapp/badges/platforms.svg)](https://anaconda.org/chiha/pydynamapp)
[![License](https://anaconda.org/chiha/pydynamapp/badges/license.svg)](https://anaconda.org/chiha/pydynamapp)
[![Downloads](https://anaconda.org/chiha/pydynamapp/badges/downloads.svg)](https://anaconda.org/chiha/pydynamapp)

DynaMapp is a Python software package designed for the modeling, identification, and calibration of dynamic models of serial robotic manipulators. 
The primary goal of this project is to offer a flexible framework for modeling and identifying serial manipulators, including the incorporation of nonlinear effects such as friction, stiffness, and backlash in robot joints. Additionally, a predictive feedback control mechanism is being developed to compensate these effects, making the framework suitable for collaborative robotics applications.

The project aims to explore subspace identification methods for manipulators. Given the nonlinear nature of the problem, the state-space representation parameters are state-dependent, requiring more advanced mathematical tools to analyze such systems effectively.  

Table of Contents
---
- [DynaMapp](#dynamapp)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Package Structure](#package-structure)
  - [Installation](#installation)
      - [Supported Platform](#supported-platform)
  - [Exemples](#exemples)
      - [Create Multiple Robot Model Representations](#create-multiple-robot-model-representations)
        - [Analyze Robot Stability and Control Design Using Common Tools](#analyze-robot-stability-and-control-design-using-common-tools)
      - [Model Identification Routines](#model-identification-routines)
        - [Visualizing Different Models](#visualizing-different-models)
  - [Documentation](#documentation)
      - [Build](#build)
  - [Recent Releases](#recent-releases)
  - [References](#references)
  - [Contributing](#contributing)
  - [License](#license)


Prerequisites
----
DynaMapp relies heavilly for it's internal rigid body computation on Pinocchio python bindings, for more information, refer to:
  - [Pinocchio Documentation](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/)
  - [Pinocchio's GitHub Page](https://github.com/stack-of-tasks/pinocchio)
  - [Pinocchio Cheat Sheet](docs/pinocchio_sheet.pdf)

also it uses, as an entry point  to represent the rigid body tree of the robot reprsenttaion [URDF](https://wiki.ros.org/urdf),


Package Structure
---
- **`script`** folder contains executable identification scripts. Settings for each algorithm are passed through command-line arguments.
- **`dynamics`** folder contains all direct and inverse dynamics model computations, state-space representation, and regressor formulation.
- **`viscoelastic`** folder contains all base classes for contact models.
- **`models`** folder contains all electromechanical joint component models implementations.
- **`optimisation`** folder contains general optimisation routines tailored for the identification problem.
- **`robot`** folder contains common robot description files.
- **`identification`** folder contains custom identification base classes and helper functions.
- **`extern`** folder contains third-party software used by DynaMapp. This folder has other license terms than the project, so please review them individually.


Installation 
----
The code has been tested successfully on the following platform:
#### Supported Platform

- **Windows 11**:
  - Python 3.12.4
  - Miniconda 4.5.0
  - CUDA 12.5 (optional)

for installation on other platforms or additional setup instructions, refer to the [INSTALL.md](INSTALL.md) guide.

Exemples
----
This section provides an overview of examples demonstrating the usage of DynaMapp for tuning the Kinova Gen3 robot parameters to estimate the joint torques. Currently, the documentation lacks detailed and meaningful examples, and these examples do not cover all software functions.

#### Create Multiple Robot Model Representations
Use a URDF file along with a configuration file. The configuration file should follow the same layout as the [Kinova config file](https://github.com/wissem01chiha/dynamic-identification/blob/main/pyDynaMapp/robot/kinova/config.yml).

```python
from pyDynaMapp.dynamics import Robot

my_robot = Robot(urdf_file_path, configuration_file_path)

# Example: Compute mass and Coriolis matrices for the home position
C = my_robot.computeCoriolisMatrix()
M = my_robot.computeMassMatrix()

# Example: Compute the joint torques vector for the home state
T = my_robot.computeGeneralizedTorques()

# Example: Compute the joint torque vector for a given state
q = [1.25, 1, 1.25, 1, 1, 1, 1]          # Position
qp = [0.25, 1.5, 1.36, 0.25, 1, 1, 1]    # Velocity  
qpp = [0.05, 0, 0, 1.25, 0, 0.2, 0.3]    # Acceleration       

T = my_robot.computeGeneralizedTorques(q, qp, qpp)

# Example: Compute joint gravity torques for the home state
G = my_robot.computeGravityTorques()
```

Compute the Robot Regressor Matrix

$$
\tau = Y_{\chi}(q, \dot{q}, \ddot{q}) \chi + e 
$$


  ```python
  import numpy as np
  from pyDynaMapp.dynamics import Regressor

  reg = Regressor()
  q = np.random.rand(100,7)
  qp = np.random.rand(100,7)
  qpp = np.random.rand(100,7)

  #Exemple: compute the full regressor matrix 
  W = reg.computeFullRegressor(q, qp, qpp)

  #Exemple : compute the reduced rgressor matrix
  W_ = reg.computeReducedRegressor(q, qp, qpp,1e-6)
  ```
Create a State-Space Representation

$$
\begin{aligned}
    \dot{x} &= \mathcal{A}(x) x + \mathcal{B}(x) u \\
    y &= \mathcal{\hat{C}} x 
\end{aligned}
$$

  ```python
    from pyDynaMapp.dynamics import StateSpace
    import numpy as np

    robot_ss = StateSpace(urdf_file_path, configuration_file_path)
    
    # Example: Compute the state-space matrices at a given configuration using a state vector x
    x = np.random.rand(14)
    A, B, C, D = robot_ss.model.computeStateMatrices(x)
    
    # Example: Compute the state-space matrices at a random configuration using position and velocity vectors
    q = np.random.rand(7)
    qp = np.random.rand(7)
    A, B, C, D = robot_ss.model.computeStateMatrices(q, qp)
  ```
The last model computation serves as a bridge to state-space identification techniques, subspace identification methods, and other fields.

> **Note:**  The stability of the computed matrices is not guaranteed. The intensive computations of derivatives are error-prone, so a new computation method is needed!

##### Analyze Robot Stability and Control Design Using Common Tools
  
```python
import matplotlib.pyplot as plt
from pyDynaMapp.dynamics import StateSpace
model = StateSpace("urdf/file/path","configuration/file/path")

#Example: plotting the system poles state for a given position and velocity configurations
q     = [1.25, 1, 125, 1, 1, 1, 1]                  
qp   = [0.25, 1.5, 1.36, 0.25, 1, 1, 1]
model.visualizeStatePoles(q,qp)

#Exemple: plotting the system root-locus for a given position and velocity configurations
model.visualizeRootLocus()
plt.show()
```
this can optimise the desgin of a state-depend gain function $$K(q,qp)$$ to achieve desired varaible-poles location.



#### Model Identification Routines
All DynaMapp models depend on user-defined parameters modeling physical aspects of the manipulators (inertia, friction, stiffness, etc.). These parameters are concatenated to form a one-dimensional vector, and the identification problem is turned into an optimization-based problem:

$$
\begin{equation}
    \hat{\beta} = argmin_{\beta}\left\| IDM(\beta,q,\dot{q},\ddot{q})-\tau_s \right\|_2
\end{equation}
$$

Where:
- $\tau_s \in \mathbb{R}^{N}$: Torque measured from joint sensors.
- $IDM(\beta, q, \dot{q}, \ddot{q}) \in \mathbb{R}^{N}$: The inverse nonlinear flexible joint dynamics model with friction, actuator, and stiffness for a given fixed trajectory.
- $\beta \in \mathbb{R}^{n \times m}$: The global parameter vector.

> **Warning** Currently, only NLOPT-based optimizers and Kalman  
> filter optimizers are supported.

- Using NLOPT 
 ```shell
  cd pyDynaMapp/script
  python3 nlopt_identification 
 ```
This uses the last computed parameters and starts the optimizer. All problem settings can be adjusted in the [utils](pyDynaMapp/script/identification_utils.py) script.
  
##### Visualizing Different Models
   
  - Given a data file path:

  ```shell
  python3 run_dynamic_simulation -v= True -data_file_path="path/to/example.csv" --show_figures= True --cutoff_frequency=3
  ```

- Using Kalman Filter
```shell
python3 run_kalman_identification -v=True --cutoff_frequency=3 --show_figures= True
```

Documentation
----
#### Build 
We use [Sphinx](https://www.sphinx-doc.org/en/master/) to generate the software documentation. Assuming that Sphinx is installed, you can generate and build the documentation using the following steps:


to generate and build the documentation, run from the root of the project: 

```shell
sphinx-build -b html docs/source docs/build/html

```

The official documentation is hosted on this [link](https://wissem01chiha.github.io/dynamic-identification/).

The source code is hosted on the  [pyDynaMapp-Anaconda](https://anaconda.org/chiha/pydynamapp) page, 
for detailed reports and figures visit [figshare](https://figshare.com/articles/thesis/Modeling_and_Identification_of_Robotic_Manipulators_Dynamics/27215562) 

Recent Releases
---
- **[pyDynaMapp-v0.1.0](https://github.com/wissem01chiha/dynamic-identification/releases/tag/v0.1.0)** — August 2024: First release

References
----

- **A three-loop physical parameter identification method of robot manipulators considering physical feasibility and nonlinear friction mode**, *Tangzhong Song, Lijin Fang,Guanghui Liu, Hanyu Pang*, 2024
- **Comprehensive modeling and identification of nonlinear joint dynamics for collaborative industrial robot manipulators**, *Emil Madsen, Oluf Skov Rosenlund, David Brandt, Xuping Zhang*, 2020
- **Robotics Modelling Planning and Control**, *Bruno Siciliano, Lorenzo Sciavicco Luigi Villani,Giuseppe Oriolo*, 
- **Global Identification of Joint Drive Gains and Dynamic Parameters of Robots** , *Maxime Gautier*, *Sebastien Brio*, 2014
- **Direct Calculation of Minimum Set of Inertial Parameters of Serial Robots**, *Maxime Gautier, Wisama khalil*, 1990
- **Efficient Dynamic Computer Simulation of Robotic Mechanisms**, *M.W.Walker*, *D. E. Orin*, 1982  
- **Inertial Parameter Identification in Robotics: A Survey**, *Quentin Leboutet, Julien Roux, Alexandre Janot, Julio Rogelio, Gordon
Cheng*, 2021
- **Practical Modeling and Comprehensive System Identification of a BLDC Motor**, *Changle Xiang, Xiaoliang Wang, Yue Ma, and Bin Xu*, 2015
- **Identiable Parameters and Optimum Congurations for Robots Calibration**, *W. Khalil, M. Gautier and Ch. Enguehard*, 2009, *Robotica*
- **Recursive identification of certain structured time varying state-space models**, *M.H. Moazzam T. Hesketh, D.J.Clements*, 1997 
- **Comparison Between the CLOE Method and the DIDIM Method for Robots Identification**, *Alexandre Janot, Maxime Gautier, Anthony Jubien, and Pierre Olivier Vandanjon*, 2014
- **Robot Joint Modeling and Parameter Identification Using the Clamping Method**, *Christian Lehmann ∗ Bjorn Olofsson, Klas Nilsson, Marcel Halbauer, Mathias Haage, Anders Robertsson, Olof Sornmo, Ulrich Berger*, 2013
- **Fundamentals of friction modeling**, *Farid Al-Bender*, 2015
- **Constrained State Estimation - A Review**, *Nesrine Amor, Ghualm Rasool, and Nidhal C. Bouaynaya*, arXiv, 2022
- **The Pinocchio C++ library**,*J.Carpentier, G.Saurel, G.Buondonno, J.Mirabel, F.Lamiraux, O.Stasse, N.Mansard*, 2019

Contributing
----
Before contributing, please take a moment to review the following:  
- The [CHANGELOG](CHANGELOG.md) for an overview of updates and changes.    
- The [CONTRIBUTING](CONTRIBUTING.md) guide for detailed instructions on how to contribute.  

This is an early-stage research software, and contributions are highly welcomed!  

If you have any questions or need assistance, feel free to reach out via [email](mailto:chihawissem08@gmail.com).  

License
----
This project is currently licensed under the **CC0 1.0 Universal License**.  
For more details, see the [LICENSE](LICENSE.txt) file. 


https://github.com/TaISLab/simulink_arduino_dynamixel
https://github.com/geoand-robotdesign/Dynamixel_Library_MATLAB_Simulink
https://github.com/marciska/vpc-library
https://github.com/alperunlu/AutoLIB
https://github.com/odufour-mw/Simulink-Target-support-for-ED247
https://github.com/SysensoSystems/SimExcelBlocks