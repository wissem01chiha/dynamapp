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

from . import numpy_api as cnp
from .typing import (
    Array,
    DTypeLike,
    ShapeLike,
)


def make_array(
    default_value: Array, dtype: DTypeLike = None, shape: ShapeLike = None
) -> Array:
    assert not (
        shape is None and default_value is None
    ), "Must provide either shape or default_value"

    if default_value is not None:
        default_value = cnp.array(default_value, dtype=dtype)
    else:
        default_value = cnp.zeros(shape, dtype=dtype)

    # JAX doesn't support non-numeric arrays, so for consistency we will
    # ensure that no backend does.  This will mimic the error that JAX raises
    # when trying to convert a non-numeric value to a JAX array.
    if not cnp.issubdtype(default_value.dtype, cnp.number) and not (
        default_value.dtype == bool
    ):
        msg = (
            f"Parameter values must be numeric.  Got: {default_value} with "
            f"dtype {default_value.dtype}"
        )
        raise TypeError(msg)

    return default_value
