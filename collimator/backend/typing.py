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

"""Aliases for type hints."""

from __future__ import annotations

from typing import Any, Tuple, Union

from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "Array",
    "ArrayLike",
    "ShapeLike",
    "DTypeLike",
    "Scalar",
]

Scalar = ArrayLike  # Specifically, this should be a float, int, or array with shape ()

ShapeLike = Union[Tuple[int, ...], int]
DTypeLike = Any
