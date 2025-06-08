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

from ..results_data import AbstractResultsData
from .python_functions import scan

if TYPE_CHECKING:
    from ..typing import Array
    from ...framework import ContextBase, SystemCallback


__all__ = ["NumpyResultsData"]


# Inherits docstring from `AbstractResultsData`
@dataclasses.dataclass
class NumpyResultsData(AbstractResultsData):
    @staticmethod
    def initialize(
        context: ContextBase,
        recorded_signals: dict[str, SystemCallback],
        max_major_steps: int = None,
        max_minor_steps_per_major_step: int = None,
    ) -> NumpyResultsData:
        return NumpyResultsData(recorded_signals)

    def update(self, context: ContextBase) -> NumpyResultsData:
        """Update the simulation solution with the results of a simulation step.

        This stores the results of a single major step in a solution array.
        It will loop over all "minor" steps in the ODE and reconstruct the signals
        at each step, storing the results in the solution buffer.  If a pure discrete
        system is being simulated, then only a single data point will be saved per
        major step.

        Args:
            context (ContextBase):
                The simulation context at the end of the simulation step.

        Returns:
            NumpyResultsData: The updated simulation solution data.
        """
        signals = self.eval_sources(context)

        # Add an empty leading dimension to the signals for concatenation
        ts = np.expand_dims(context.time, axis=0)
        for key in self.source_dict:
            signals[key] = np.expand_dims(signals[key], axis=0)

        if self.time is None:
            # Need to initialize the results
            time = ts
            outputs = {key: np.asarray(y) for key, y in signals.items()}

        else:
            # Extend the existing arrays
            time = np.concatenate([self.time, ts], axis=0)
            outputs = {
                key: np.concatenate([self.outputs[key], y], axis=0)
                for key, y in signals.items()
            }

        return dataclasses.replace(self, outputs=outputs, time=time)

    def finalize(self) -> tuple[Array, dict[str, Array]]:
        return self.time, self.outputs

    @classmethod
    def _scan(cls, *args, **kwargs):
        return scan(*args, **kwargs)
