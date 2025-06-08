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

import jax.numpy as jnp

from ..framework import LeafSystem
from .primitives import FeedthroughBlock

__all__ = ["ode_block", "feedthrough_block"]


def ode_block(state_dim, dtype=jnp.float64, num_inputs=0, name=None):
    template_vector = jnp.zeros(state_dim, dtype=dtype)

    def _wrapper(func):
        block_name = name if name is not None else func.__name__
        block = LeafSystem(name=block_name)

        for i in range(num_inputs):
            block.declare_input_port(name=f"{block.name}:input[{i}]")

        block.declare_continuous_state(default_value=template_vector, ode=func)
        block.declare_continuous_state_output(
            name=f"{block.name}:output"
        )  # One vector-valued output
        return block

    return _wrapper


def feedthrough_block(func):
    return FeedthroughBlock(func, name=func.__name__)
