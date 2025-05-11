⚠️ **Note : This repository and all related projects are undergoing extensive refactoring in private development. A revised version will be released in the future.Not recommended for use at this time.**


<!-- omit in toc -->
DynaMapp 
======= 


[![Tests](https://github.com/wissem01chiha/dynamapp/actions/workflows/tests.yml/badge.svg)](https://github.com/wissem01chiha/dynamapp/actions/workflows/tests.yml)
[![PyLint](https://github.com/wissem01chiha/dynamapp/actions/workflows/pylint.yml/badge.svg)](https://github.com/wissem01chiha/dynamapp/actions/workflows/pylint.yml)
[![build-docs](https://github.com/wissem01chiha/dynamapp/actions/workflows/build-docs.yml/badge.svg)](https://github.com/wissem01chiha/dynamapp/actions/workflows/build-docs.yml)
![GitHub License](https://img.shields.io/github/license/wissem01chiha/dynamapp)
![GitHub last commit](https://img.shields.io/github/last-commit/wissem01chiha/dynamapp)
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/t/wissem01chiha/dynamapp/main)

DynaMapp is a lightweight Python software package designed for the representation and identification of multibody systems. It is optimized for efficient computation and visualization of how static input parameters (such as inertial, geometric, and electromechanical) affect the behavior of these systems.
 
<!-- omit in toc -->
## Table of Contents
- [🚀 About](#-about)
- [📝 Installation](#-installation)
- [💻 Examples](#-examples)
- [📚 Documentation](#-documentation)
- [📦 Releases](#-releases)
- [🤝 Contributing](#-contributing)
- [📃 License](#-license)

## 🚀 About

The primary goal of **DynaMapp** is to offer an implementation of common rigid body dynamics algorithms and their derivatives using [JAX](https://jax.readthedocs.io/en/latest/quickstart.html).

It provides tools for computing the [state-space](https://en.wikipedia.org/wiki/State-space_representation) representation of these systems.

- Compute the Jacobian tensors of the joint torque vector as a function of the system parameters vector using [automatic differentiation](https://jax.readthedocs.io/en/latest/automatic-differentiation.html).
- Compute the Jacobian of other quantities (such as the global inertia matrix and Coriolis matrix) with respect to the input parameters.
- Implement common identification algorithms for optimizing the multibody system parameters.

The package does not rely on any rigid body dynamics libraries.

> **Note**:  
 This is an early-stage research software, and many  parts still require focus and further implementation.
>



## 📝 Installation  

See the [INSTALL](INSTALL.md) file.

## 💻 Examples

This section provides an overview of examples demonstrating the usage of the package's basic functions and mathematical notations. Currently, the documentation lacks detailed and meaningful examples, and these examples do not cover all software functions.

All guidelines will be available in the [Tutoriel](https://wissem01chiha.github.io/dynamapp/TUTORIAL.html).


<!-- omit in toc -->
#### Example 1: Creating a Model Instance

```python
from dynamapp.model import Model
# Define the Inertia matrices (list of 6x6 matrices)
Imats = [jnp.eye(6) for _ in range(3)]
# Define the Denavit-Hartenberg parameters (theta, d, a, alpha)
dhparams = [
    [0.0, 0.5, 0.5, jnp.pi / 2],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5, jnp.pi / 2]
]
# Define the gravity vector
gravity = -9.81
# Define damping coefficients
dampings = [0.1, 0.1, 0.1]
# Create the model instance
model = Model(Imats, dhparams, gravity, dampings)
# Check the generalized torques at initial joint configurations (q, qp, qpp = 0)
torques = model.generalized_torques()
```
<!-- omit in toc -->
#### Example 2: Computing the Generalized Torques and Inertia Matrix

```python
from dynamapp.model import Model
# Define joint positions, velocities, and accelerations
q = jnp.array([0.0, 0.0, 0.0])  # Joint positions (rad)
qp = jnp.array([0.0, 0.0, 0.0]) # Joint velocities (rad/s)
qpp = jnp.array([0.0, 0.0, 0.0]) # Joint accelerations (rad/s^2)
# Define the Inertia matrices, DH parameters, and damping coefficients
Imats = [jnp.eye(6) for _ in range(3)]
dhparams = [
    [0.0, 0.5, 0.5, jnp.pi / 2],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5, jnp.pi / 2]
]
gravity = -9.81
dampings = [0.1, 0.1, 0.1]
# Create the model instance
model = Model(Imats, dhparams, gravity, dampings)
# Compute the Generalized Torques at the current joint configuration
generalized_torques = model.generalized_torques(q, qp, qpp)
# Compute the Inertia Matrix at the current joint configuration
inertia_matrix = model.inertia_tensor(q)
```
<!-- omit in toc -->
#### Example 3: Computing State Matrices (A, B, C, D)

$$
\begin{aligned}
    \dot{x} &= \mathcal{A}(x) x + \mathcal{B}(x) u \\
    y &= \mathcal{\hat{C}} x 
\end{aligned}
$$

```python
from dynamapp.model_state import ModelState
# Define system parameters (Inertia matrices, DH parameters, etc.)
Imats = [jnp.eye(6) for _ in range(3)]  # Example inertia matrices
dhparams = [
    [0.0, 0.5, 0.5, jnp.pi / 2],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5, jnp.pi / 2]
]
# Initialize the ModelState object
model_state = ModelState(Imats, dhparams)
# Define an example state vector (x)
x = jnp.zeros((2 * model_state.model.ndof, 1))  # Example state vector (2 * ndof)
# Compute the state-space matrices
model_state._compute_matrices(x)
# Access and print the state-space matrices A, B, C, D
print(model_state.model_state_space.a)
print(model_state.model_state_space.b)
print(model_state.model_state_space.c)
print(model_state.model_state_space.d) 
```
The last model computation serves as a bridge to state-space identification techniques, subspace identification methods, and other fields.

> **Note:**  The stability of the computed matrices is not guaranteed. The intensive computations of derivatives are error-prone, so a new computation method is needed!
> 
<!-- omit in toc -->
#### Example 4: Advanced Analysis — Stability, Controllability, and Observability
  
```python
from dynamapp.model_state import ModelState
Imats = [jnp.eye(6) for _ in range(3)]  # Example inertia matrices
dhparams = [
    [0.0, 0.5, 0.5, jnp.pi / 2],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5, jnp.pi / 2]
]
# Initialize the ModelState object
model_state = ModelState(Imats, dhparams)
# Define an example state vector (x)
x = jnp.zeros((2 * model_state.model.ndof, 1))  # Example state vector (2 * ndof)
# Check if the system is stable
is_stable = model_state._is_stable(x)
eigenvalues = model_state.compute_eigvals(x)
controllability_matrix = model_state.compute_ctlb_matrix(x)
# Compute the observability matrix
observability_matrix = model_state.compute_obs_matrix(x)
```

<!-- omit in toc -->
####  Example 4: Torques Derivatives with Respect to Inertia

$$J = \frac{\partial \tau}{\partial I}$$

```python
m = Model(...)  # A Model object
q = jnp.array([0.5, 1.0, -0.3])  # Generalized positions (q)
v = jnp.array([0.1, -0.2, 0.3])  # Generalized velocities (v)
a = jnp.array([0.05, 0.1, -0.15])  # Generalized accelerations (a)
t = generalized_torques_wrt_inertia(m, q, v, a)
```
<!-- omit in toc -->
#### Example 5: Torques Derivatives with Respect to DH Parameters
$$J = \frac{\partial \tau}{\partial \theta}
$$
```python
# Example Usage
q = jnp.array([0.5, 1.0, -0.3])  # Generalized positions (q)
v = jnp.array([0.1, -0.2, 0.3])  # Generalized velocities (v)
a = jnp.array([0.05, 0.1, -0.15])  # Generalized accelerations (a)
# Compute the Jacobian of generalized torques with respect to DH parameters
torques_wrt_dhparams = generalized_torques_wrt_dhparams(m, q, v, a)
```
<!-- omit in toc -->
#### Example 6: Torques Derivatives with Respect to Damping
$$J = \frac{\partial \tau}{\partial c}
$$
```python
q = jnp.array([0.5, 1.0, -0.3])  # Generalized positions (q)
v = jnp.array([0.1, -0.2, 0.3])  # Generalized velocities (v)
a = jnp.array([0.05, 0.1, -0.15])  # Generalized accelerations (a)
torques_wrt_damping = generalized_torques_wrt_damping(m, q, v, a)
```

## 📚 Documentation 
 
The official documentation is avalible at [link](https://wissem01chiha.github.io/dynamapp/README.html)


## 📦 Releases
- **[v1.0.0](https://github.com/wissem01chiha/dynamapp/tree/main)** - Jan 2025, current release
- **[v0.1.0](https://github.com/wissem01chiha/dynamapp/tree/dev)** — august 2024: first release.


## 🤝 Contributing

please review the following:  
- The [CHANGELOG](CHANGELOG.md) for an overview of updates and changes.    
- The [CONTRIBUTING](CONTRIBUTING.md) guide for detailed instructions on how to contribute.  

This is an early-stage research software, and contributions are highly welcomed!  

If you have any questions or need assistance, feel free to reach out via [email](mailto:chihawissem08@gmail.com).  

## 📃 License

See the [LICENSE](LICENSE.txt) file. 

[Back to top](#top)


 
 
