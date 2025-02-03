"""
DynaMapp
====================
A differentiable package for representation and 
identification of multibody dynamics.

Usage:
------
>>> import dynamapp as dymp
"""
from .version import __version__

from .trajectory import (
    SplineTrajectory,
    TrapezoidalTrajectory,
    PeriodicTrajectory,
    StepTrajectory
)
from .regressors import *
from .reductions import *
from .generators import *
from .actuators import *
from .model import *
from .identification import *
from .viscoelastic import *
from .visualization import *

__all__ = [
    "SplineTrajectory",
    "TrapezoidalTrajectory",
    "PeriodicTrajectory",
    "StepTrajectory",
]

