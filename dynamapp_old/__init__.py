"""
DynaMapp
====================
A Differentiable Package for State Representation 
And Identification of Multibody Dynamics

Usage:
------
>>> import dynamapp as dymp
"""
import warnings
from .version import __version__

from .trajectory import (
    SplineTrajectory,
    TrapezoidalTrajectory,
    PeriodicTrajectory,
    StepTrajectory
)
from .regressors import (
    generalized_torques_wrt_inertia,
    inertia_tensor_wrt_inertia,
    generalized_torques_wrt_dhparams,
    generalized_torques_wrt_damping,
    full_torques_wrt_inertia,
    full_torques_wrt_friction,
    eigvals_wrt_inertia,
    eigvals_wrt_dhparams,
    eigvals_wrt_damping,
    state_matrix_a_wrt_inertia,
    state_matrix_a_wrt_state
)
from .generators import (
    ModelDataGenerator,
    ModelStateDataGenerator
)
from .model import Model
from .model_state import ModelState
from .state_space import StateSpace
from .reductions import PCA, LDA
from .identification import *
from .viscoelastic import (
    compute_coulomb_friction_force,
    compute_friction_force
)
from .visualization import *
from .kalman import Kalman
from .nfoursid import NFourSID

__all__ = [
    "Model",
    "ModelState",
    "Kalman", "NFourSID",
    "StateSpace",
    "StateSpace",
    "PCA","LDA",
    "SplineTrajectory",
    "TrapezoidalTrajectory",
    "PeriodicTrajectory",
    "StepTrajectory",
    "ModelDataGenerator",
    "ModelStateDataGenerator",
    "generalized_torques_wrt_inertia",
    "inertia_tensor_wrt_inertia",
    "generalized_torques_wrt_dhparams",
    "generalized_torques_wrt_damping",
    "full_torques_wrt_inertia",
    "full_torques_wrt_friction",
    "eigvals_wrt_inertia",
    "eigvals_wrt_dhparams",
    "eigvals_wrt_damping",
    "state_matrix_a_wrt_inertia",
    "state_matrix_a_wrt_state",
    "compute_coulomb_friction_force",
    "compute_friction_force"
]

_internal_modules = {"solvers", "math_utils","version"}

def __getattr__(name):
    if name in _internal_modules:
        warnings.warn(f"Module 'dynamapp.{name}' is for internal \
        use only and should not be imported.", UserWarning)
        return None 
    raise AttributeError(f"module 'dynamapp' has no attribute '{name}'")