# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

from typing import TYPE_CHECKING
import numpy as np

from .python_functions import (
    cond,
    scan,
    while_loop,
    fori_loop,
    callback,
    jit,
    astype,
    interp2d,
)
from .ode_solver import ODESolver
from .results_data import NumpyResultsData

from ...lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation
else:
    scipy = LazyLoader("scipy", globals(), "scipy")
    Rotation = LazyModuleAccessor(scipy, "spatial.transform.Rotation")

__all__ = ["numpy_functions", "numpy_constants"]

numpy_functions = {
    "astype": astype,
    "cond": cond,
    "scan": scan,
    "while_loop": while_loop,
    "fori_loop": fori_loop,
    "jit": jit,
    "io_callback": callback,
    "pure_callback": callback,
    "ODESolver": ODESolver,
    "interp2d": interp2d,
}

numpy_constants = {
    "lib": np,
    "Rotation": Rotation,
    "ResultsDataImpl": NumpyResultsData,
}
