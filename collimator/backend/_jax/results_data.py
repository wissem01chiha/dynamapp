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
from typing import TYPE_CHECKING
import dataclasses

import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import io_callback

from ..results_data import AbstractResultsData

if TYPE_CHECKING:
    from ...framework import SystemCallback, ContextBase
    from ..typing import Array


__all__ = ["JaxResultsData"]


def _raise_buffer_overflow(n_steps, buffer_length):
    # FIXME (WC-291): Should this be a warning, a RuntimeError, or a custom exception?
    # Ideally this would actually just trigger dumping the results to NumPy arrays and
    # clearing the buffers.
    if n_steps > buffer_length:
        raise RuntimeError(
            f"Results buffer overflow: {n_steps} > {buffer_length} steps. "
            "Increase the buffer size to store all results"
        )


@jax.jit
def error_buffer_overflow(n_steps, buffer_length):
    jax.debug.callback(_raise_buffer_overflow, n_steps, buffer_length)


def _make_empty_solution(
    context: ContextBase,
    recorded_signals: dict[str, SystemCallback],
    buffer_length: int,
) -> JaxResultsData:
    """Create an empty "buffer" solution data object with the correct shape.
    For each source in "recorded_signals", determine what the signal data type is
    and create an empty vector that can hold enough data to max out the simulation
    buffer.
    """

    def _expand_template(source: SystemCallback):
        # Determine the data type of the signal (shape and dtype)
        x = source.eval(context)
        if jnp.isscalar(x):
            x = jnp.asarray(x)
        # Create a buffer that can hold the maximum number of (major, minor) steps
        return jnp.zeros((buffer_length, *x.shape), dtype=x.dtype)

    signals = {
        key: _expand_template(source) for key, source in recorded_signals.items()
    }
    # The time vector is used to determine the number of steps taken by the ODE solver
    # since diffrax will return inf for unused buffer entries. Then we can use isfinite
    # to trim the unused buffer space.  For this reason, initialize to inf rather
    # than zero.
    times = jnp.full((buffer_length), jnp.inf)
    return JaxResultsData(
        source_dict=recorded_signals,
        outputs=signals,
        time=times,
        buffer_length=buffer_length,
    )


def _trim(solution: JaxResultsData) -> tuple[Array, dict[str, Array]]:
    """Remove unused entries from the buffer and return flattened arrays.

    See `JaxResultsData.finalize` for more details.
    """
    # Adaptive ODE solvers should return inf for unused buffer entries.
    # Then we can use isfinite to trim the unused buffer space.
    valid_idx = jnp.isfinite(solution.time)
    time = np.array(solution.time[valid_idx])

    outputs = {}
    for key, y in solution.outputs.items():
        outputs[key] = np.array(y[valid_idx])

    # If there is stored NumPy data in the solution, add the buffer data to it.
    if solution.np_data.time is not None:
        time = np.append(solution.np_data.time, time, axis=0)
        solution.np_data.time = None
        for key, value in outputs.items():
            outputs[key] = np.append(solution.np_data.outputs[key], value, axis=0)

    return time, outputs


@dataclasses.dataclass
class _NumpyData:
    """Class to store the solution data in NumPy arrays when the buffer is full.

    This doesn't seem like it should merit its own class, but this seems to be the
    only way to successfully store the data from within the JIT-compiled function.
    """

    time: np.ndarray = None
    outputs: dict[str, np.ndarray] = None

    def dump_buffer(self, buffer_full: bool, time: Array, outputs: dict[str, Array]):
        """If the solution buffer is full, store the results in NumPy arrays."""
        if not buffer_full:
            return

        # Dump the buffer to NumPy arrays
        if self.time is None:
            self.time = np.asarray(time)
            self.outputs = {key: np.asarray(value) for key, value in outputs.items()}

        else:
            self.time = np.append(self.time, np.asarray(time), axis=0)
            for key, value in outputs.items():
                self.outputs[key] = np.append(
                    self.outputs[key], np.asarray(value), axis=0
                )


# Inherits docstring from `AbstractResultsData`
@dataclasses.dataclass
class JaxResultsData(AbstractResultsData):
    n_steps: int = 0  # Number of saved time stamps

    # Index of the current buffer position. Will get reset when buffer is full and
    # the contents are dumped to NumPy arrays.
    buffer_index: int = 0
    buffer_length: int = None  # Maximum number of time stamps to save

    # Data stored in numpy arrays as the buffer fills up
    np_data: _NumpyData = dataclasses.field(default_factory=_NumpyData)

    @staticmethod
    def initialize(
        context: ContextBase,
        recorded_signals: dict[str, SystemCallback],
        buffer_length: int,
    ) -> JaxResultsData:
        return _make_empty_solution(context, recorded_signals, buffer_length)

    def update(self, context: ContextBase) -> JaxResultsData:
        """Update the simulation solution with the results of a simulation step.

        This stores the current state of the system in the buffer arrays corresponding
        to the recorded signals.

        Args:
            context (ContextBase):
                The simulation context at the end of the simulation step.

        Returns:
            JaxResultsData: The updated simulation solution data.
        """

        # Index of the current major step in the solution data buffer.
        index = self.buffer_index

        # initialize the time buffer
        self.time = jnp.where(index == 0, jnp.full_like(self.time, jnp.inf), self.time)

        # In this case we only need to get the signal at the current step,
        # since there are no intermediate steps from the ODE solver.
        y = self.eval_sources(context)

        outputs = {
            key: self.outputs[key].at[index].set(y[key]) for key in self.source_dict
        }

        # Set the first entry of the time vector to the current time.
        # The rest will still be inf, indicating unused buffer entries.
        time = self.time.at[index].set(context.time)

        buffer_index = index + 1

        # If the buffer is full:
        # - dump the buffer to NumPy arrays
        # - reset the time buffer array to `inf`.  This will signal unused buffer entries.
        # - reset the buffer index
        buffer_full = buffer_index >= self.buffer_length
        io_callback(self.np_data.dump_buffer, None, buffer_full, time, outputs)
        buffer_index = jnp.where(buffer_full, 0, buffer_index)

        return dataclasses.replace(
            self,
            outputs=outputs,
            time=time,
            n_steps=self.n_steps + 1,
            buffer_index=buffer_index,
        )

    def finalize(self) -> tuple[Array, dict[str, Array]]:
        """Trim unused buffer space from the solution data.

        The raw solution data contains the full 'buffer' of simulation steps. This function
        trims the unused buffer space from the solution data.

        Because this returns variable-length arrays depending on the results of the solver
        calls, it cannot be called from a JAX jit-compiled function.  Instead, call as part
        of a 'postprocessing' step after simulation is complete.  This is done by default
        if the simulation is invoked via the `simulate` function.
        """
        return _trim(self)

    @classmethod
    def _scan(cls, *args, **kwargs):
        return lax.scan(*args, **kwargs)


#
# Register as custom pytree nodes
#    https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
#
def _solution_flatten(solution: JaxResultsData):
    """Flatten the solution data for tracing."""
    children = (
        solution.time,
        solution.outputs,
        solution.n_steps,
        solution.buffer_index,
    )
    aux_data = (
        solution.source_dict,
        solution.buffer_length,
        solution.np_data,
    )
    return children, aux_data


def _solution_unflatten(aux_data, children):
    """Unflatten the solution data after tracing."""
    time, outputs, n_steps, buffer_index = children
    source_dict, buffer_length, np_data = aux_data
    return JaxResultsData(
        source_dict=source_dict,
        time=time,
        outputs=outputs,
        n_steps=n_steps,
        buffer_index=buffer_index,
        buffer_length=buffer_length,
        np_data=np_data,
    )


jax.tree_util.register_pytree_node(
    JaxResultsData,
    _solution_flatten,
    _solution_unflatten,
)
