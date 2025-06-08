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
import jax
from jax import lax
import jax.numpy as jnp
from .python_functions import interp2d

from .ode_solver import ODESolver
from .results_data import JaxResultsData

from ...lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    from jax.scipy.spatial.transform import Rotation
else:
    # NOTE: spatial and transform seem to be lazy loaded in jax.scipy too
    transform = LazyLoader("transform", globals(), "jax.scipy.spatial.transform")
    Rotation = LazyModuleAccessor(transform, "Rotation")

__all__ = ["jax_functions", "jax_constants"]


jax_functions = {
    "cond": lax.cond,
    "scan": lax.scan,
    "while_loop": lax.while_loop,
    "fori_loop": lax.fori_loop,
    "jit": jax.jit,
    "io_callback": jax.experimental.io_callback,
    "pure_callback": jax.pure_callback,
    "ODESolver": ODESolver,
    "interp2d": interp2d,
}

jax_constants = {
    "lib": jnp,
    "Rotation": Rotation,
    "ResultsDataImpl": JaxResultsData,
}
