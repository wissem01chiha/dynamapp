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
Base classes for optimizables.

Once created, the optimizables can be passed to an optimizer instance for optimization.
"""

from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.flatten_util import ravel_pytree

from collimator.backend.typing import Array
from collimator.simulation import Simulator, SimulatorOptions, estimate_max_major_steps
from collimator.logging import logger


@dataclass
class DesignParameter:
    """
    For UI purposes, this is used to structure optimization jobs from UI
    """

    param_name: str
    initial: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class Distribution:
    """
    For UI purposes, this is used to structure optimization jobs from UI
    """

    name: str
    # shape is not defined in json: ui_jobs.py creates shapes from json input
    shape: Optional[tuple] = None
    # includes min, max, mean, std_dev
    options: Optional[dict] = field(default_factory=dict)


@dataclass
class StochasticParameter:
    """
    For UI purposes, this is used to structure optimization jobs from UI
    """

    param_name: str
    distribution: Distribution


@dataclass
class DistributionConfig:
    """
    Structure of attributes for specifying distributions for stochastic variables
    """

    names: list[str]
    shapes: list[tuple]
    distributions: list[str]
    distributions_configs: list[dict]


def _flatten_bounds(bounds, params):
    flat_bounds_dict = {}
    for key, value in params.items():
        if isinstance(value, jnp.ndarray):
            flat_bounds_dict[key] = (
                jnp.full(value.size, bounds[key][0]),
                jnp.full(value.size, bounds[key][1]),
            )
        else:
            flat_bounds_dict[key] = bounds[key]

    # Flatten the bounds dictionary
    bounds_flat = []
    for key, (lower, upper) in flat_bounds_dict.items():
        if isinstance(lower, jnp.ndarray):
            bounds_flat.extend(zip(lower.tolist(), upper.tolist()))
        else:
            bounds_flat.append((lower, upper))
    return bounds_flat


class OptimizableBase(ABC):
    """
    Base class for all optimizables.

    It creates a simulator for the user diagram, sets initial parameter values, and
    creates unflattening function for the parameters.

    The abstract methods `optimizable_params` and `objective_from_context` must be
    provided by concrete classes. The `constraints_from_context` method is optional.
    Its default implementation returns None, implying no constraints. When constraints
    are present, derived classes should override this method.

    Parameters:
        diagram: Diagram or LeafSystem
            The system to be optimized.
        base_context: Context
            The base context for the system, which can be used for parameter
            initialization when initial values are not directly provided.
        sim_t_span: tuple
            The time span for simulation.
        params_0: dict of {parameter_name: parameter_value} pairs
            Initial parameter values. If not provided, the `optimizable_params` method
            will be used to extract these from the base context.
        bounds: dict of {parameter_name: (lower_bound, upper_bound)} pairs
            Bounds for optimization. For array parameters a single tuple representing
            the bounds on all the array members is expected.
        transformation: Transform
            transformation to be applied to the parameters before optimization. This
            object should provide two methods `transform` and `inverse_transform`. The
            `transform` method should take a dict of parameters and return a transformed
            dict of parameters. The `inverse_transform` method should take a dict of
            transformed parameters and return the original dict of parameters.
            See the classes in `optimization.framework.base.transformations`.
        init_min_max: dict of {parameter_name: (min, max)} pairs
            Min and max values for initialization. Applies only to population-based
            optimizers. For array parameters a single tuple representing the min/max
            values for all the array members is expected.
        seed: int
            Seed for any necessary randomization.
    """

    def __init__(
        self,
        diagram,
        base_context,
        sim_t_span,
        params_0,
        bounds,
        transformation,
        init_min_max,
        seed,
    ):
        self.diagram = diagram
        self.base_context = base_context

        self.start_time, self.stop_time = sim_t_span

        # Create simulator
        max_major_steps = estimate_max_major_steps(
            diagram,
            sim_t_span,
        )

        options = SimulatorOptions(
            enable_autodiff=True,
            max_major_steps=max_major_steps,
            max_major_step_length=0.01,  # autodiff breaks with large step lengths
            # rtol=1e-08,
            # atol=1e-08,
        )
        self.simulator = Simulator(diagram, options=options)

        # Initialize params
        if params_0 is None:
            self.params_0 = self.optimizable_params(base_context)
        else:
            self.params_0 = params_0

        # Convert to jnp arrays if params_0 values were lists or tuples
        self.params_0 = {key: jnp.array(value) for key, value in self.params_0.items()}

        # Transform parameters
        self.transformation = transformation
        if transformation is not None:
            self.params_0 = transformation.transform(self.params_0)

        self.params_0_flat, self.unflatten_params = ravel_pytree(self.params_0)
        self.num_optvars = self.params_0_flat.size

        # key for any randomization
        self.key = jr.PRNGKey(np.random.randint(0, 2**32) if seed is None else seed)

        # bounds for optimization
        self.bounds = bounds

        # Transform bounds
        if transformation is not None and bounds is not None:
            lbs = {key: value[0] for key, value in bounds.items()}
            ubs = {key: value[1] for key, value in bounds.items()}
            lbs = transformation.transform(lbs)
            ubs = transformation.transform(ubs)
            self.bounds = {key: (lbs[key], ubs[key]) for key in lbs.keys()}

        self.bounds_flat = (
            _flatten_bounds(self.bounds, self.params_0) if bounds is not None else None
        )

        # min max for initilization —-- only for population-based optimizers
        self.init_min_max = init_min_max

        # Transform init_min_max
        if transformation is not None and init_min_max is not None:
            mins = {key: value[0] for key, value in init_min_max.items()}
            maxs = {key: value[1] for key, value in init_min_max.items()}
            mins = transformation.transform(mins)
            maxs = transformation.transform(maxs)
            self.init_min_max = {key: (mins[key], maxs[key]) for key in mins.keys()}

        self.init_min_max_flat = (
            _flatten_bounds(self.init_min_max, self.params_0)
            if init_min_max is not None
            else None
        )

        self.has_constraints = self.constraints_from_context(base_context) is not None

    @abstractmethod
    def optimizable_params(self, context) -> dict:
        """
        Extract optimizable model-specific parameters from the context.
        These should be in the form of a dict of Pytrees.
        """
        pass

    @abstractmethod
    def objective_from_context(self, context):
        """Model-specific objective function, evaluated on final context"""
        pass

    def constraints_from_context(self, context):
        """Model-specific constraints, evaluated on final context.
        If constraints are present then this method should be overridden by the user.
        """
        return None


class Optimizable(OptimizableBase):
    """
    Base class for all optimizables with no stochastic variables.

    For parameters, see `OptimizableBase`.

    The abstract method `prepare_context` should update the context to incorporate the
    optimization parameters.

    This classs creates methods for evaluation of the objective and constraints from the
    concrete implementation of the abstract methods. This class also creates methods for
    batched evaluation of the objective and constraints, which are useful for optimizers
    that can work with batches (eg. Optax), and population-based optimizers.
    """

    def __init__(
        self,
        diagram,
        base_context,
        sim_t_span=(0.0, 1.0),
        params_0=None,
        bounds=None,
        transformation=None,
        init_min_max=None,
        seed=None,
    ):
        super().__init__(
            diagram,
            base_context,
            sim_t_span,
            params_0,
            bounds,
            transformation,
            init_min_max,
            seed,
        )
        self.batched_objective = jax.jit(jax.vmap(self.objective, in_axes=(0,)))
        self.batched_objective_flat = jax.jit(
            jax.vmap(self.objective_flat, in_axes=(0,))
        )

        self.batched_constraints = jax.jit(jax.vmap(self.constraints, in_axes=(0,)))
        self.batched_constraints_flat = jax.jit(
            jax.vmap(self.constraints_flat, in_axes=(0,))
        )

    @abstractmethod
    def prepare_context(self, context, params: dict):
        """
        Model-specific updates to incorporate the sample data and parameters.
        Return the updated context.
        """
        pass

    def run_simulation(self, params: dict):
        """
        Run simulation and return final results context.
        """
        context = self.base_context.with_time(self.start_time)
        context = self.prepare_context(context, params)
        results = self.simulator.advance_to(self.stop_time, context)
        return results.context

    def objective_flat(self, params: Array):
        """Objective function for optimization with flattened parameters input"""
        return self.objective(self.unflatten_params(jnp.atleast_1d(params)))

    def objective(self, params: dict):
        """Objective function for optimization with dict parameters input"""
        if self.transformation is not None:
            params = self.transformation.inverse_transform(params)
        results_context = self.run_simulation(params)
        return self.objective_from_context(results_context)

    def constraints_flat(self, params: Array):
        """Constraints function for optimization with flattened parameters input"""
        return self.constraints(self.unflatten_params(jnp.atleast_1d(params)))

    def constraints(self, params: dict):
        """Constraints function for optimization with dict parameters input"""
        if self.transformation is not None:
            params = self.transformation.inverse_transform(params)
        results_context = self.run_simulation(params)
        return self.constraints_from_context(results_context)


class OptimizableWithStochasticVars(OptimizableBase):
    """
    Base class for all optimizables with stochastic variables. This is designed
    only for Optax optimizers and without constraints. Other optimizers are unlikely to
    work well with stochastic variables.

    This class is similar to `Optimizable` with the key difference that both `params`
    and `vars` (stochastic variables) need to be updated as opposed to `params` alone

    Parameters:
        vars_0: dict
            Initial stochastic variable values. If not provided, the
            `stochastic_vars` method will be used to extract these from the
            base context.
        distribution_config_vars: DistributionConfig
            Configuration for stochastic variables. If not provided, standard normal
            distribution is used.
    """

    def __init__(
        self,
        diagram,
        base_context,
        sim_t_span=(0.0, 1.0),
        params_0=None,
        vars_0=None,
        distribution_config_vars=None,
        bounds=None,
        transformation=None,
        seed=None,
    ):
        super().__init__(
            diagram,
            base_context,
            sim_t_span,
            params_0,
            bounds,
            transformation,
            init_min_max=None,
            seed=seed,
        )

        if vars_0 is None:
            self.vars_0 = self.stochastic_vars(base_context)
        else:
            self.vars_0 = vars_0

        self.vars_0_flat, self.unflatten_vars = ravel_pytree(self.vars_0)
        self.num_stochastic_vars = self.vars_0_flat.size

        self.batched_objective = jax.jit(jax.vmap(self.objective, in_axes=(None, 0)))
        self.batched_objective_flat = jax.jit(
            jax.vmap(self.objective_flat, in_axes=(None, 0))
        )

        if distribution_config_vars is None:
            logger.warning(
                "`distribution_config_vars` is not specified. Using standard normal "
                "as the default distribution"
            )
            self.distribution_config_vars = DistributionConfig(
                names=list(self.vars_0.keys()),
                shapes=[jnp.shape(x) for x in self.vars_0.values()],
                distributions=["normal"] * len(self.vars_0),
                distributions_configs=[{}] * len(self.vars_0),
            )
        else:
            self.distribution_config_vars = distribution_config_vars

    @abstractmethod
    def prepare_context(self, context, params: dict, vars: dict):
        """
        Model-specific updates to incorporate the parameters and stochastic vars.
        Return the updated context.
        """
        pass

    @abstractmethod
    def stochastic_vars(self, context) -> dict:
        """
        Extract stochastic `vars` from the context.
        These should be in the form of a dict of Pytrees.
        """
        pass

    def run_simulation(self, params: dict, vars: dict):
        """Run simulation and return final results context."""
        context = self.base_context.with_time(self.start_time)
        context = self.prepare_context(context, params, vars)
        results = self.simulator.advance_to(self.stop_time, context)
        return results.context

    def objective_flat(self, params: Array, vars: Array):
        """Objective function for optimization with flattened parameters and vars
        input"""
        return self.objective(
            self.unflatten_params(jnp.atleast_1d(params)),
            self.unflatten_vars(jnp.atleast_1d(vars)),
        )

    def objective(self, params: dict, vars: dict):
        """Objective function for optimization with dict parameters and vars input"""
        if self.transformation is not None:
            params = self.transformation.inverse_transform(params)
        results_context = self.run_simulation(params, vars)
        return self.objective_from_context(results_context)

    def sample_random_vars(self, num_samples):
        """Generate random samples of the stochastic variables"""
        names = self.distribution_config_vars.names
        shapes = self.distribution_config_vars.shapes
        distributions = self.distribution_config_vars.distributions
        distributions_configs = self.distribution_config_vars.distributions_configs
        data, flat_data = self._generate_random_data(
            names,
            shapes,
            distributions,
            distributions_configs,
            num_samples,
        )
        return data, flat_data

    def generate_batches(
        self,
        data,
        num_batches,
        batch_size,
    ):
        """
        Given all samples `data`, generate `num_batches` random batches of size
        `batch_size` each
        """
        num_samples = data.shape[0]
        self.key, subkey = jr.split(self.key)
        batch_indices = jax.random.choice(
            subkey, num_samples, (num_batches, batch_size), replace=True
        )
        batches = data[batch_indices]
        return batches

    @staticmethod
    def _distribution(name: str, key, shape, options: dict):
        # remap options from json names to jax.random names
        # FIXME: code users should be able to pass default argnames (i.e. those used
        # by jax.random), for example, "minval" and "maxval" for uniform distribution
        if name == "normal":
            mean = options.get("mean", 0.0)
            std_dev = options.get("std_dev", 1.0)
            return jr.normal(key, shape) * std_dev + mean

        if name == "lognormal":
            mean = options.get("mean", 0.0)
            sigma = options.get("std_dev", 1.0)
            return jr.lognormal(key, sigma=sigma, shape=shape) + mean

        if name == "uniform":
            minval = options.get("min", 0.0)
            maxval = options.get("max", 1.0)
            return jr.uniform(key, shape=shape, minval=minval, maxval=maxval)

        warnings.warn(f"Unknown distribution: {name}.")
        sample_func = getattr(jr, name)
        return sample_func(key, shape, **options)

    def _generate_random_data(
        self,
        names,
        shapes,
        distributions,
        distributions_configs,
        num_samples,
    ):
        data = {}
        self.key, *subkeys = jr.split(self.key, len(names) + 1)
        for key, name, shape, distribution, distribution_config in zip(
            subkeys, names, shapes, distributions, distributions_configs
        ):
            data[name] = self._distribution(
                distribution,
                key,
                (num_samples, *shape),
                distribution_config,
            )

        def _flatten(x):
            x_flat, _ = ravel_pytree(x)
            return x_flat

        return data, _flatten(data)
