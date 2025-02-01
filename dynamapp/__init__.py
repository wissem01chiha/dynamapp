"""
DynaMapp
=======
Provides:
    1. .
Usage:
------
>>> import dynamapp as dp
"""
from .version import __version__

from .trajectory import * 
from .regressors import *
from .reduction import *
from .data_utils import *
from .actuators import *
from .model import *
from .kalman import *
from .solvers import *
from .moesp import *
from .visualization import *

__all__ = [
    "trajectory",
    "regressors",
    "data_utils",
    "reduction",
    "actuators",
    "visualization",
    "identification",
    "kalman",
    "model",
    "solvers",
    "moesp",
]