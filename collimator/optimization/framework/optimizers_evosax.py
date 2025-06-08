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
Population based global optimizers from Evosax.
"""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from collimator.lazy_loader import LazyLoader
from collimator.logging import logger
from collimator.optimization.framework.base.metrics import MetricsWriter

from .base import Optimizer, Optimizable

evosax = LazyLoader(
    "evosax", globals(), "evosax", error_message="evosax is not installed."
)

if TYPE_CHECKING:
    import evosax


class Evosax(Optimizer):
    """
    Population based global optimizers from Evosax.

    Parameters:
        optimizable (Optimizable):
            The optimizable object.
        opt_method (str):
            The optimization method to use. See `evosax.Strategies` for
            available methods.
        opt_method_config (dict):
            Configuration for the optimization method.
        pop_size (int):
            The population size.
        num_generations (int):
            The number of generations.
        print_every (int):
            Print progress every `print_every` generations.
        metrics_writer (MetricsWriter|None):
            Optional CSV file to write metrics to.
        seed (int):
            The random seed.
    """

    def __init__(
        self,
        optimizable: Optimizable,
        opt_method="CMA_ES",
        opt_method_config=None,
        pop_size=10,
        num_generations=100,
        print_every=1,
        seed=None,
        metrics_writer: MetricsWriter = None,
    ):
        self.optimizable = optimizable
        self.opt_method = opt_method
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.print_every = print_every
        self.metrics_writer = metrics_writer
        self.optimal_params = None

        self.num_dims = optimizable.params_0_flat.size

        if self.optimizable.has_constraints:
            raise ValueError(
                f"Optimization method evosax:{self.opt_method} "
                "does not support constraints."
            )

        if opt_method not in evosax.Strategies:
            raise ValueError(f"Unknown optimization method: {opt_method}")

        if opt_method_config is None:
            opt_method_config = {}

        self.strategy = evosax.Strategies[opt_method](self.pop_size, self.num_dims)
        self.es_params = self.strategy.default_params.replace(**opt_method_config)

        # Create bounds
        if optimizable.bounds_flat is not None:
            lower_bounds, upper_bounds = zip(*optimizable.bounds_flat)
            lb = jnp.array(lower_bounds)
            ub = jnp.array(upper_bounds)
            self.es_params = self.es_params.replace(clip_min=lb, clip_max=ub)

        # Create initialization bounds
        if optimizable.init_min_max_flat is not None:
            init_min, init_max = zip(*optimizable.init_min_max_flat)
            imin = jnp.array(init_min)
            imax = jnp.array(init_max)
            self.es_params = self.es_params.replace(init_min=imin, init_max=imax)

        else:
            # if bounds are specified, unless they are infinity, we can use
            # them for initialization. If infinity, we initialize in [-0.1,0.1]
            if optimizable.bounds_flat is not None:
                bounds = [
                    (
                        -0.1 if b[0] == -jnp.inf else b[0],
                        0.1 if b[1] == jnp.inf else b[1],
                    )
                    for b in optimizable.bounds_flat
                ]
                lower_bounds, upper_bounds = zip(*bounds)
                lb = jnp.array(lower_bounds)
                ub = jnp.array(upper_bounds)
                self.es_params = self.es_params.replace(init_min=lb, init_max=ub)

            # if strategy defaults are not zero, they are likely set to sensible values,
            # so we use them, otherwise we scale the initial params by a factor of 10
            elif self.es_params.init_min == 0 and self.es_params.init_max == 0:
                factor = 10.0
                imin = jnp.full(self.num_dims, self.optimizable.params_0_flat / factor)
                imax = jnp.full(self.num_dims, self.optimizable.params_0_flat * factor)
                self.es_params = self.es_params.replace(init_min=imin, init_max=imax)

        self.key = jr.PRNGKey(np.random.randint(0, 2**32) if seed is None else seed)

    def optimize(self):
        """Run optimization"""
        fitness_func = jax.jit(self.optimizable.batched_objective_flat)

        state = self.strategy.initialize(self.key, self.es_params)

        # https://github.com/RobertTLange/evosax/issues/45
        state = state.replace(best_fitness=jnp.finfo(jnp.float64).max)

        for gen in range(self.num_generations):
            self.key, subkey = jr.split(self.key)
            x, state = self.strategy.ask(subkey, state, self.es_params)
            fitness = fitness_func(x)
            state = self.strategy.tell(x, fitness, state, self.es_params)

            if self.print_every is not None and (gen + 1) % self.print_every == 0:
                logger.info(
                    "# Gen: %3d|Fitness: %.6f|Params: %s",
                    gen + 1,
                    state.best_fitness,
                    state.best_member,
                )
            if self.metrics_writer is not None:
                self.metrics_writer.write_metrics(best_fitness=state.best_fitness)

        params = state.best_member
        self.optimal_params = self.optimizable.unflatten_params(params)
        if self.optimizable.transformation is not None:
            self.optimal_params = self.optimizable.transformation.inverse_transform(
                self.optimal_params
            )
        return self.optimal_params
