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

"""JAX-backend ODE solvers for continuous-time simulation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .dopri5 import Dopri5Solver
from .bdf import BDFSolver

from ..ode_solver import ODESolverBase, ODESolverOptions

if TYPE_CHECKING:
    from collimator.framework import SystemBase

__all__ = ["ODESolver"]


def ODESolver(
    system: SystemBase,
    options: ODESolverOptions = None,
) -> ODESolverBase:
    """Create an ODE solver used to advance continuous time in hybrid simulation.

    Args:
        system (SystemBase):
            The system to be simulated.
        options (ODESolverOptions, optional):
            Options for the ODE solver.  Defaults to None.
        enable_tracing (bool, optional):
            Whether to enable tracing of time derivatives for JAX solvers.  Defaults
            to True.

    Returns:
        ODESolverBase:
            An instance of a class implementing the `ODESolverBase` interface.
            The specific class will depend on the system and options.
    """

    if options is None:
        options = ODESolverOptions()
    options = dataclasses.asdict(options)

    method = options.pop("method", "auto")

    # legacy support
    if method == "default":
        method = "auto"

    if method == "auto":
        method = "bdf" if system.has_mass_matrix else "dopri5"

    Solver = {
        "stiff": BDFSolver,
        "non-stiff": Dopri5Solver,
        "rk45": Dopri5Solver,
        "dopri5": Dopri5Solver,
        "bdf": BDFSolver,
    }[method.lower()]

    return Solver(system, **options)
