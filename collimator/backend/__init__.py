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

from .backend import (
    DEFAULT_BACKEND,
    IS_JAXLITE,
    dispatcher,
    asarray,
    array,
    zeros,
    zeros_like,
    reshape,
    Rotation,
    cond,
    scan,
    while_loop,
    fori_loop,
    jit,
    io_callback,
    pure_callback,
    ODESolver,
    ResultsData,
    inf,
    nan,
)

from .ode_solver import ODESolverOptions, ODESolverState

# Alternate name for clear imports `from collimator.backend import numpy_api`
numpy_api = dispatcher

__all__ = [
    "DEFAULT_BACKEND",
    "IS_JAXLITE",
    "dispatcher",
    "asarray",
    "array",
    "zeros",
    "zeros_like",
    "reshape",
    "Rotation",
    "cond",
    "scan",
    "while_loop",
    "fori_loop",
    "jit",
    "io_callback",
    "pure_callback",
    "ODESolver",
    "ODESolverOptions",
    "ODESolverState",
    "ResultsData",
    "inf",
    "nan",
]
