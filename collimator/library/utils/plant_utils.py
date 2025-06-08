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

"""Utilities to generate a function that evaluates ODE RHS for a plant"""

import jax
import jax.numpy as jnp

from jax.flatten_util import ravel_pytree


def make_ode_rhs(plant, nu):
    """
    Return a function that evaluate the plant's ODE RHS at state `x`, control input
    `u`, and time `t`: ode_rhs(x, u, t) -> dx/dt

    Parameters:
        plant : LeafSystem or Diagram
                The plant for which to generate the ODE RHS function. If Diagram, then
                enities within the diagram should have only one vector-valued
                Integrator, i.e. only one continuous state.
        nu : int
            The plant is expected to only have one vector-valued input, and `nu`
            is the size of this vector.
    """
    # create a base context
    input_port = plant.input_ports[0]
    with input_port.fixed(jnp.zeros(nu)):
        base_context = plant.create_context()

    # base_context.continuous_state is a list if plant is a Diagram
    # and an array if plant is a LeafSystem.
    # Former case with more than one integrator is not supported
    if isinstance(base_context.continuous_state, list):
        if len(base_context.continuous_state) != 1:
            raise NotImplementedError(
                f"Plant has more than one (total {len(base_context.continuous_state)}) "
                f"integrator. Only single vector-valued continuous state is supported."
            )

    _, unravel = ravel_pytree(base_context.continuous_state)

    @jax.jit
    def ode_rhs(x, u, t):
        context = base_context
        with input_port.fixed(u):
            context = context.with_continuous_state(unravel(x))
            context = context.with_time(t)
            _dot_x = plant.eval_time_derivatives(context)
        dot_x, _ = ravel_pytree(_dot_x)
        return dot_x

    return ode_rhs
