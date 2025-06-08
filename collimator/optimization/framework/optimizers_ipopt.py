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

"""
Interior Point Optimizer (IPOPT) for optimization of the objective function
with constraints.
"""

import jax
import jax.numpy as jnp

from collimator.lazy_loader import LazyLoader

from .base import Optimizer, Optimizable

cyipopt = LazyLoader(
    "cyipopt",
    globals(),
    "cyipopt",
    error_message="cyipopt is not installed.",
)


class IPOPT(Optimizer):
    """
    Interior Point Optimizer (IPOPT) for optimization of the objective function
    with constraints.

    Prameters:
        optimizable (Optimizable):
            The optimizable object.
        options (dict):
            Options for the IPOPT solver.
            See https://coin-or.github.io/Ipopt/OPTIONS.html
    """

    def __init__(self, optimizable: Optimizable, options: dict = {"disp": 5}):
        self.optimizable = optimizable
        self.options = options
        self.optimal_params = None

    def optimize(self):
        """Run optimization"""
        params = self.optimizable.params_0_flat
        objective = jax.jit(self.optimizable.objective_flat)
        gradient = jax.jit(jax.grad(objective))
        hessian = jax.jit(jax.hessian(objective))

        constraints = jax.jit(self.optimizable.constraints_flat)
        constraints_jac = jax.jit(jax.jacrev(constraints))
        constraints_hessian = jax.jit(jax.hessian(constraints))

        @jax.jit
        def constraints_hessian_vp(x, v):
            return jnp.sum(
                jnp.multiply(v[:, jnp.newaxis, jnp.newaxis], constraints_hessian(x)),
                axis=0,
            )

        constraints_ipopt = [
            {
                "type": "ineq",
                "fun": constraints,
                "jac": constraints_jac,
                "hess": constraints_hessian_vp,
            }
        ]

        # Handle bounds
        bounds = self.optimizable.bounds_flat

        # Jobs from UI would put (-jnp.inf, jnp.inf) as defualt bounds. The user
        # may also have specified bounds this way. IPOPT scipy interface expects `None`
        # to imply unboundedness.
        if bounds is not None:
            bounds = [
                (
                    None if b[0] == -jnp.inf else b[0],
                    None if b[1] == jnp.inf else b[1],
                )
                for b in bounds
            ]

            # Check if all bounds are None, i.e. no bounds at all, and hence
            # algorithms that do not support bounds can be used.
            flattened_bounds = [element for tup in bounds for element in tup]
            all_none = all(element is None for element in flattened_bounds)
            bounds = None if all_none else bounds

        res = cyipopt.minimize_ipopt(
            objective,
            x0=params,
            jac=gradient,
            hess=hessian,
            constraints=constraints_ipopt,
            bounds=bounds,
            options=self.options,
        )

        params = res.x

        self.optimal_params = self.optimizable.unflatten_params(params)
        if self.optimizable.transformation is not None:
            self.optimal_params = self.optimizable.transformation.inverse_transform(
                self.optimal_params
            )
        return self.optimal_params
