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

from __future__ import annotations
import os
from typing import Callable

import numpy as np
from jax.tree_util import register_pytree_node_class
import jax

from ._numpy import numpy_functions, numpy_constants
from .results_data import AbstractResultsData

# IS_JAXLITE is used for the pyodide build where we only have numpy and jaxlite
IS_JAXLITE = os.environ.get("JAXLITE", "0") == "1"
USE_TORCH_BACKEND = os.environ.get("USE_TORCH_BACKEND", "0") == "1"
DEFAULT_BACKEND = "jax" if not IS_JAXLITE else "numpy"
if not IS_JAXLITE:
    from ._jax import jax_functions, jax_constants
    from ._torch import torch_functions, torch_constants


def _make_backend(name, functions, constants):
    # Create a new class with the given name and attributes
    static_functions = {
        name: staticmethod(function) for name, function in functions.items()
    }
    attrs = {**static_functions, **constants}
    return type(name, (), attrs)()


class MathDispatcher:
    """Singleton class for calling out to the appropriate backend."""

    _active_backend = DEFAULT_BACKEND
    _disable_x64 = False

    _backends = {
        "numpy": _make_backend("NumpyBackend", numpy_functions, numpy_constants),
    }

    if not IS_JAXLITE:
        _backends["jax"] = _make_backend("JaxBackend", jax_functions, jax_constants)
        if USE_TORCH_BACKEND:
            _backends["torch"] = _make_backend(
                "TorchBackend", torch_functions(), torch_constants
            )

    def __init__(self) -> None:
        if IS_JAXLITE:
            return

        # FIXME can't switch to 32 bits after init
        enable_x64 = os.environ.get("JAX_ENABLE_X64", "true").lower() != "false"
        jax.config.update("jax_enable_x64", enable_x64)
        self._disable_x64 = not enable_x64

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def set_backend(self, backend: str):
        if backend not in self._backends:
            raise ValueError(f"Backend {backend} not supported")
        self._active_backend = backend

    @property
    def intx(self):
        """
        Defines native int bit size (32 or 64), by default we want 64 bits.

        intx is similar to intp (and intc, int_) except that the choice
        depends on runtime variables, not on the platform/OS. Set the env
        var "JAX_ENABLE_X64" to "false" to disable x64.
        """
        return self.int32 if self._disable_x64 else self.int64

    def __getattr__(self, name):
        backend = self._backends[self.active_backend]
        # First look for the attribute in the backend in case the default is overridden
        if hasattr(backend, name):
            return getattr(self._backends[self.active_backend], name)
        # Else try to get it from the underlying lib
        if hasattr(backend.lib, name):
            return getattr(backend.lib, name)
        raise AttributeError(f"Backend {self.active_backend} has no attribute {name}")

    def function(self, name: str) -> Callable:
        # These seem to have to be wrapped in a function to avoid fixing
        # the backend at the time of definition.
        def _call(*args, **kwargs):
            return getattr(self, name)(*args, **kwargs)

        return _call

    @property
    def Rotation(self):
        return self._backends[self.active_backend].Rotation

    @property
    def ResultsDataImpl(self) -> AbstractResultsData:
        return self._backends[self.active_backend].ResultsDataImpl


dispatcher = MathDispatcher()

# TODO: Do we need to be specific about which version of these constants gets used?
inf = np.inf
nan = np.nan

# Alias some core functions for convenience
asarray = dispatcher.function("asarray")
array = dispatcher.function("array")
zeros = dispatcher.function("zeros")
zeros_like = dispatcher.function("zeros_like")
reshape = dispatcher.function("reshape")
cond = dispatcher.function("cond")
scan = dispatcher.function("scan")
while_loop = dispatcher.function("while_loop")
fori_loop = dispatcher.function("fori_loop")
jit = dispatcher.function("jit")
io_callback = dispatcher.function("io_callback")
pure_callback = dispatcher.function("pure_callback")
Rotation = dispatcher.Rotation
interp2d = dispatcher.function("interp2d")

ODESolver = dispatcher.function("ODESolver")


@register_pytree_node_class
class ResultsData:
    def __init__(self, solution_data: AbstractResultsData):
        self._solution_data = solution_data

    @property
    def time(self):
        return self._solution_data.time

    @property
    def outputs(self):
        return self._solution_data.outputs

    @staticmethod
    def initialize(*args, **kwargs) -> ResultsData:
        solution = dispatcher.ResultsDataImpl.initialize(*args, **kwargs)
        return ResultsData(solution)

    def update(self, *args, **kwargs) -> ResultsData:
        solution = self._solution_data.update(*args, **kwargs)
        return ResultsData(solution)

    def finalize(self):
        return self._solution_data.finalize()

    @classmethod
    def _scan(cls, *args, **kwargs):
        return scan(*args, **kwargs)

    def tree_flatten(self):
        return (self._solution_data,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (solution_data,) = children
        return ResultsData(solution_data)
