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

from typing import Any

import numpy as np


class RandomDistribution:
    """Represents a random distribution to be used in monte-carlo simulations.
    Must be a valid numpy random distribution.
    """

    def __init__(self, distribution: str, **parameters):
        if distribution not in np.random.__dict__:
            raise ValueError(f"Unknown distribution: {distribution}")
        self.distribution = distribution
        self.parameters = parameters


class SweepValues:
    """Represents a list of values to sweep over."""

    def __init__(self, values: list[Any]):
        self.values = values
