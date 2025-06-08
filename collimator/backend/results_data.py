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
import abc
from typing import TYPE_CHECKING
import dataclasses
import numpy as np

from jax import tree_util


if TYPE_CHECKING:
    from ..framework import SystemCallback, ContextBase
    from ..framework.state import StateComponent
    from ..backend.typing import Array


__all__ = [
    "AbstractResultsData",
    "eval_source",
]


def _get_at_index(states: StateComponent, index: int) -> StateComponent:
    """Isolate the state values at a given time step"""
    return tree_util.tree_map(lambda x: x[index], states)


def eval_source(
    source: SystemCallback,
    context: ContextBase,
    xc: StateComponent,
    t: Array,
    index: int,
) -> Array:
    """Evaluate a callback (e.g. output port) at a given time step"""
    context = context.with_continuous_state(_get_at_index(xc, index))
    context = context.with_time(t[index])
    result = source.eval(context)
    return result


@dataclasses.dataclass
class AbstractResultsData(metaclass=abc.ABCMeta):
    """Simulation results data stored in buffer arrays.

    This is an intermediate storage object and should not normally need
    to be interacted with directly.  The actual post-processed results
    will be available in the `SimulationResults` object.

    Attributes:
        source_dict (dict[str, SystemCallback]):
            Dictionary of system callbacks, indexed by the signal name as defined
            by `recorded_signals` and passed to either `simulate` or
            `SimulationOptions`. Typically these sources will be output ports.
        outputs (dict[str, Array]):
            Dictionary of simulation outputs, indexed by the signal name as defined
            by `recorded_signals` and passed to either `simulate` or
            `SimulationOptions`. Each array has shape (max_major_steps, max_minor_steps+1, ...)
            where max_major_steps is the maximum number of time intervals (major steps)
            in the simulation and max_minor_steps is the maximum number of steps taken by the
            ODE solver in each interval.  The first entry in the array is the initial
            value of the signal, and the remaining entries are the values of the signal
            at each step taken by the ODE solver in the corresponding time interval.
            Unused data is filled with `inf`.
        time (Array):
            The time steps at which the ODE solver was evaluated.  This is a 2D array
            with shape (max_major_steps, max_minor_steps+1) where max_major_steps is the
            maximum number of time intervals (major steps) in the simulation and
            max_minor_steps is the maximum number of steps taken by the ODE solver in each
            interval.  The first entry in the array is the initial time, and the
            remaining entries are the times at which the ODE solver was evaluated in
            the corresponding time interval.  Unused data is filled with `inf`.
    Unused entries in the buffer arrays can be stripped with the `trim` method,
    although this cannot be called from within a JAX jit-compiled function. Instead,
    call `postprocess` on the final `SimulationResults` object.
    """

    # Dictionary of system callbacks, indexed by the signal name as defined
    # by `recorded_signals` and passed to either `simulate` or `SimulationOptions`.
    # Typically these sources will be output ports.
    source_dict: dict[str, SystemCallback]

    time: Array = None
    outputs: dict[str, Array] = dataclasses.field(default_factory=dict)

    @staticmethod
    @abc.abstractmethod
    def initialize(
        context: ContextBase,
        recorded_signals: dict[str, SystemCallback],
        buffer_length: int = None,
    ) -> AbstractResultsData:
        pass

    @abc.abstractmethod
    def update(self, context: ContextBase) -> AbstractResultsData:
        """Update the simulation solution with the results of a simulation step.

        This stores the results of a single major step in a modified solution buffer
        (so that it acts as a pure function for the purposes of JAX tracing).

        Args:
            context (ContextBase):
                The simulation context at the end of the simulation step.

        Returns:
            ResultsData: The updated simulation solution data.
        """
        pass

    @abc.abstractmethod
    def finalize(self) -> tuple[Array, dict[str, Array]]:
        """Finalize the simulation solution and return the time and outputs."""
        pass

    def eval_sources(self, context: ContextBase) -> dict[str, np.ndarray]:
        """Evaluate all the signals with the current context."""
        return {key: source.eval(context) for key, source in self.source_dict.items()}

    def _eval_scan_fun(self, carry, i):
        """Collect all the signals at a single ODE solver step"""
        signals = {
            key: eval_source(source, *carry, i)
            for key, source in self.source_dict.items()
        }
        return carry, signals

    @classmethod
    @abc.abstractmethod
    def _scan(cls, *args, **kwargs):
        """Dispatch to a backend-appropriate scan implementation."""
        pass
