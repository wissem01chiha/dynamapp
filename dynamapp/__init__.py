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
from .jacobians import (
    ModelJacobian, ModelStateJacobian
)
from .generators import (
    ModelDataGenerator,
    ModelStateDataGenerator
)
from .model import Model
from .model_state import ModelState
from .state_space import StateSpace
from .identification import *
from .viscoelastic import (
    coulomb_friction_force,
    friction_force
)
from .visualization import (
    TrajectoryVisualizer
)
from .kalman import Kalman
from .nfoursid import NFourSID

__all__ = [
    "Model", "ModelState",
    "Kalman", "NFourSID",
    "StateSpace", "StateSpace",
    "SplineTrajectory", "TrapezoidalTrajectory",
    "PeriodicTrajectory", "StepTrajectory",
    "TrajectoryVisualizer",
    "ModelDataGenerator","ModelStateDataGenerator",
    "ModelJacobian","ModelStateJacobian",
    "coulomb_friction_force","friction_force"
]

_internal_modules = {"solvers", "math_utils","version"}

def __getattr__(name):
    if name in _internal_modules:
        warnings.warn(f"Module 'dynamapp.{name}' is for internal \
        use only and should not be imported.", UserWarning)
        return None 
    raise AttributeError(f"module 'dynamapp' has no attribute '{name}'")