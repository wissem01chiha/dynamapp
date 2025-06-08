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
Optimizers using the Optax library.
"""

import inspect
import warnings
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree

from collimator.logging import logdata, logger
from collimator.lazy_loader import LazyLoader, LazyModuleAccessor
from collimator.optimization.framework.base.metrics import MetricsWriter

from .base import Optimizer, Optimizable, OptimizableWithStochasticVars

optax = LazyLoader("optax", globals(), "optax")

if TYPE_CHECKING:
    from jaxtyping import ArrayLike
    import optax
else:
    jaxtyping = LazyLoader("jaxtyping", globals(), "jaxtyping")
    ArrayLike = LazyModuleAccessor(jaxtyping, "ArrayLike")

API_TO_OPTAX_OPTIONS_MAPPING = {
    "adam": {
        "epsilon": "eps",
        "epsilon_root": "eps_root",
        "beta1": "b1",
        "beta2": "b2",
    }
}


def _remap_and_filter_valid_params(func, kwargs):
    func_name = func.__name__
    new_kwargs = {}
    sig = inspect.signature(func)
    for argname, v in kwargs.items():
        new_argname = API_TO_OPTAX_OPTIONS_MAPPING.get(func_name, {}).get(
            argname, argname
        )
        new_kwargs[new_argname] = v

    valid_params = set(sig.parameters.keys())
    extra_keys = set(new_kwargs.keys()) - valid_params
    if extra_keys:
        warnings.warn(
            f"Warning: The following config keys are not used by {func.__name__} and "
            f"will be ignored: {extra_keys}. Supported args: {valid_params}"
        )

    return {k: v for k, v in new_kwargs.items() if k in valid_params}


class OptaxWithStochasticVars(Optimizer):
    """
    Optax optimizer with support for stochastic variables.

    Parameters:
        optimizable (OptimizableWithStochasticVars):
            The optimizable object.
        opt_method (str):
            The optimization method to use.
        learning_rate (float):
            The learning rate.
        opt_method_config (dict):
            Configuration for the optimization method.
        num_epochs (int):
            The number of epochs.
        batch_size (int):
            The batch size.
        num_batches (int):
            The number of batches.
        clip_range (tuple):
            The range to clip the gradients.
        print_every (int):
            Print progress every `print_every` epochs.
        metrics_writer (MetricsWriter|None):
            Optional CSV file to write metrics to.
    """

    def __init__(
        self,
        optimizable: OptimizableWithStochasticVars,
        opt_method: str,
        learning_rate,
        opt_method_config,
        num_epochs=100,
        batch_size=1,
        num_batches=1,
        clip_range=None,
        print_every=None,
        metrics_writer: MetricsWriter = None,
    ):
        self.optimizable = optimizable
        self.opt_method = opt_method
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.clip_range = clip_range
        self.print_every = print_every
        self.metrics_writer = metrics_writer
        self.optimal_params = None
        self.losses = []

        if optimizable.bounds_flat is not None:
            # Jobs from UI would put (-jnp.inf, jnp.inf) as defualt bounds. The user
            # may also have specified bounds this way.
            bounds = [
                (
                    None if b[0] == -jnp.inf else b[0],
                    None if b[1] == jnp.inf else b[1],
                )
                for b in self.optimizable.bounds_flat
            ]

            # Check if all bounds are None, i.e. no bounds at all, and hence Optax
            # algorithms which don't natively support bounds can be used
            flattened_bounds = [element for tup in bounds for element in tup]
            all_none = all(element is None for element in flattened_bounds)

            if not all_none:
                raise ValueError(
                    f"Optimization method {opt_method} does not support bounds."
                )

        opt_func = getattr(optax, opt_method, None)
        if opt_func is None:
            raise ValueError(f"Unknown optax optimizer: {opt_method}")

        # Instantiate the optimizer with validated config
        valid_opts = _remap_and_filter_valid_params(opt_func, opt_method_config)
        self.optimizer = opt_func(learning_rate, **valid_opts)

    def batched_objective_flat(self, params, stochastic_vars_batch_flat):
        """Mean of the objective function over a batch"""
        return jnp.mean(
            self.optimizable.batched_objective_flat(params, stochastic_vars_batch_flat)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, stochastic_vars_batch):
        """Take a single optimization step over one batch"""
        batch_loss, grads = jax.value_and_grad(self.batched_objective_flat)(
            params, stochastic_vars_batch
        )

        grads = jnp.clip(grads, *self.clip_range) if self.clip_range else grads

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, batch_loss

    def optimize(self):
        """Run optimization"""
        params = self.optimizable.params_0_flat
        opt_state = self.optimizer.init(params)

        if self.num_batches * self.batch_size == 1:
            # don't randomize over stochastic variables; use a single
            # batch of size 1 with initial stochastic variables
            data_flat, _ = ravel_pytree(self.optimizable.vars_0)
            stochastic_vars_training_data_flat = data_flat[None, None, :]
        else:
            _, stochastic_vars_training_data_flat = self.optimizable.sample_random_vars(
                self.num_batches * self.batch_size
            )

        @jax.jit
        def _scan_fun(carry, stochastic_vars_batch):
            params, opt_state = carry
            params, opt_state, batch_loss = self.step(
                params, opt_state, stochastic_vars_batch
            )
            return (params, opt_state), batch_loss

        for epoch in range(self.num_epochs):
            if self.num_batches * self.batch_size == 1:
                stochastic_vars_batches = stochastic_vars_training_data_flat
            else:
                stochastic_vars_batches = self.optimizable.generate_batches(
                    stochastic_vars_training_data_flat,
                    self.num_batches,
                    self.batch_size,
                )
            (params, opt_state), batch_losses = lax.scan(
                _scan_fun, (params, opt_state), stochastic_vars_batches
            )

            self.losses.append(jnp.mean(batch_losses))
            if self.print_every and epoch % self.print_every == 0:
                p: dict = self.optimizable.unflatten_params(params)
                if self.optimizable.transformation is not None:
                    p = self.optimizable.transformation.inverse_transform(p)
                p = {k: v.tolist() for k, v in p.items()}
                logger.info(
                    "Epoch %s, average batch loss: %s",
                    epoch,
                    jnp.mean(batch_losses),
                    **logdata(params=p),
                )
            if self.metrics_writer is not None:
                self.metrics_writer.write_metrics(loss=self.losses[-1])

        self.optimal_params = self.optimizable.unflatten_params(params)
        if self.optimizable.transformation is not None:
            self.optimal_params = self.optimizable.transformation.inverse_transform(
                self.optimal_params
            )
        return self.optimal_params

    @property
    def metrics(self):
        return {"loss": self.losses}


class Optax(Optimizer):
    """
    Optax optimizer without support for stochastic variables.

    Paramters:
        optimizable (Optimizable):
            The optimizable object.
        opt_method (str):
            The optimization method to use.
        learning_rate (float):
            The learning rate.
        opt_method_config (dict):
            Configuration for the optimization method.
        num_epochs (int):
            The number of epochs.
        clip_range (tuple):
            The range to clip the gradients.
        print_every (int):
            Print progress every `print_every` epochs.
        metrics_writer (MetricsWriter|None):
            Optional CSV file to write metrics to.
    """

    def __init__(
        self,
        optimizable: Optimizable,
        opt_method,
        learning_rate,
        opt_method_config,
        num_epochs=100,
        clip_range=None,
        print_every=None,
        metrics_writer: MetricsWriter = None,
    ):
        self.optimizable = optimizable
        self.opt_method = opt_method
        self.num_epochs = num_epochs
        self.clip_range = clip_range
        self.print_every = print_every
        self.metrics_writer = metrics_writer
        self.optimal_params = None
        self.losses = []

        if self.optimizable.bounds_flat is not None:
            # Jobs from UI would put (-jnp.inf, jnp.inf) as defualt bounds. The user
            # may also have specified bounds this way.
            bounds = [
                (
                    None if b[0] == -jnp.inf else b[0],
                    None if b[1] == jnp.inf else b[1],
                )
                for b in self.optimizable.bounds_flat
            ]

            # Check if all bounds are None, i.e. no bounds at all, and hence Optax
            # algorithms which don't natively support bounds can be used
            flattened_bounds = [element for tup in bounds for element in tup]
            all_none = all(element is None for element in flattened_bounds)

            if not all_none:
                raise ValueError(
                    f"Optimization method {opt_method} does not support bounds."
                )

        if self.optimizable.has_constraints:
            raise ValueError(
                f"Optimization method optax:{self.opt_method} "
                "does not support constraints."
            )

        opt_func = getattr(optax, opt_method)

        # Instantiate the optimizer with validated config
        valid_opts = _remap_and_filter_valid_params(opt_func, opt_method_config)
        self.optimizer = opt_func(learning_rate, **valid_opts)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state):
        """Take a single optimization step"""
        loss, grads = jax.value_and_grad(self.optimizable.objective_flat)(params)
        grads = jnp.clip(grads, *self.clip_range) if self.clip_range else grads
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def optimize(self) -> dict[str, ArrayLike]:
        """Run optimization"""
        params = self.optimizable.params_0_flat
        opt_state = self.optimizer.init(params)

        for epoch in range(self.num_epochs):
            params, opt_state, loss = self.step(params, opt_state)
            self.losses.append(jnp.mean(loss))
            if self.print_every and epoch % self.print_every == 0:
                p: dict = self.optimizable.unflatten_params(params)
                if self.optimizable.transformation is not None:
                    p = self.optimizable.transformation.inverse_transform(p)
                p = {k: v.tolist() for k, v in p.items()}
                logger.info(
                    "Epoch %s, loss: %s", epoch, jnp.mean(loss), **logdata(params=p)
                )
            if self.metrics_writer:
                self.metrics_writer.write_metrics(loss=self.losses[-1])

        self.optimal_params = self.optimizable.unflatten_params(params)
        if self.optimizable.transformation is not None:
            self.optimal_params = self.optimizable.transformation.inverse_transform(
                self.optimal_params
            )
        return self.optimal_params

    @property
    def metrics(self):
        return {"loss": self.losses}
