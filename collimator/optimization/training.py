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

"""Some basic machine learning utilities

Specifically, for training neural networks and other parameters
via stochastic gradient descent.
"""

import abc
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.flatten_util import ravel_pytree

from collimator.logging import logger
from collimator.simulation import Simulator, estimate_max_major_steps
from collimator.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import optax
else:
    optax = LazyLoader("optax", globals(), "optax")

__all__ = ["Trainer"]


# This is a simple alternative to vmap, since vmap doesn't work with
# our custom adjoints: https://collimator.atlassian.net/browse/WC-53
def batch_scan(forward, *batch_data):
    def _scan_fun(carry, sample_data):
        J, N = carry
        J += forward(*sample_data)
        N += 1
        return (J, N), None

    (J_sum, N), _ = lax.scan(_scan_fun, (0.0, 0), batch_data)

    return J_sum / N


class Trainer:
    """Base class for optimizing model parameters via simulation.

    Should probably get a more descriptive name once we're doing other kinds
    of training...
    """

    def __init__(
        self,
        simulator: Simulator,
        context,
        optimizer="adamw",
        lr=1e-3,
        print_every=10,
        clip_range=(-10.0, 10.0),
        **opt_kwargs,
    ):
        self.simulator = simulator
        self.context = context
        self.opt_state = None

        # See https://optax.readthedocs.io/en/latest/api.html for supported optimizers
        self.optimizer = getattr(optax, optimizer)(lr, **opt_kwargs)

        self.print_every = print_every
        self.clip_range = clip_range

    @abc.abstractmethod
    def optimizable_parameters(self, context):
        """Extract optimizable model-specific parameters from the context.

        These should be in the form of a PyTree (e.g. tuple, dict, array, etc)
        and should be the first arguments to `prepare_context`.
        """
        pass

    @abc.abstractmethod
    def prepare_context(self, context, *data, key=None):
        """Model-specific updates to incorporate the sample data and parameters.

        `data` should be the combination of the output of `optimizable_parameters`
        along with all the per-simulation "training data".  Parameters will
        update once per epoch, and training data will update once per sample.
        """
        pass

    @abc.abstractmethod
    def evaluate_cost(self, context):
        """Model-specific cost function, evaluated on final context"""
        pass

    def make_forward(self, start_time, stop_time):
        """Create a generic forward pass through the simulation, returning loss"""

        # Take all the data and model parameters, run a simulation, return loss.
        def _simulate(key, *data):
            context = self.context.with_time(start_time)
            context = self.prepare_context(context, *data, key=key)
            results = self.simulator.advance_to(stop_time, context)
            return self.evaluate_cost(results.context)

        return _simulate

    def make_loss_fn(self, forward, params):
        """Create a loss function based on a forward pass of the simulation

        `params` here can be any PyTree - it will get flattened to a single array
        """
        # Flatten all optimizable parameters into a single array
        p0, unflatten = ravel_pytree(params)

        # Define the loss as the mean cost function over the data set
        def _loss(p, key, *batch_data):
            # Map the forward pass over all the data points and return the loss
            loss = batch_scan(partial(forward, key, unflatten(p)), *batch_data)
            return loss

        # JIT compile the loss function and return the initial parameter
        # array and unflatten function
        return jax.jit(_loss), p0, unflatten

    def train(
        self,
        training_data,
        sim_start_time,
        sim_stop_time,
        epochs=100,
        key=None,
        params=None,
        opt_state=None,
    ):
        """Run the optimization loop over the training data"""

        if (
            self.simulator.max_major_steps is None
            or self.simulator.max_major_steps <= 0
        ):
            self.simulator.max_major_steps = estimate_max_major_steps(
                self.simulator.system,
                (sim_start_time, sim_stop_time),
                self.simulator.max_major_step_length,
            )

        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 2**32))

        # Create a function to evaluate the forward pass through the simulation
        forward = self.make_forward(sim_start_time, sim_stop_time)

        # Pull out the optimizable parameters from the context
        if params is None:
            params = self.optimizable_parameters(self.context)

        # Initialize the optimizer and create the loss function
        loss, p, unflatten = self.make_loss_fn(forward, params)

        if opt_state is None:
            opt_state = self.optimizer.init(p)

        self.opt_state = opt_state

        @jax.jit
        def opt_step(p, opt_state, key, batch_data):
            key, subkey = jax.random.split(key)
            if batch_data:
                loss_value, grads = jax.value_and_grad(loss)(p, subkey, *batch_data)
            else:
                loss_value, grads = jax.value_and_grad(loss)(p, subkey)

            grads = jnp.clip(grads, *self.clip_range)

            updates, opt_state = self.optimizer.update(grads, opt_state, p)
            p = optax.apply_updates(p, updates)
            return p, opt_state, key, loss_value

        def _scan_fun(carry, batch_data):
            p, opt_state, key, loss_value = opt_step(*carry, batch_data)
            return (p, opt_state, key), loss_value

        # Run the optimization loop
        for epoch in range(epochs):
            (p, self.opt_state, key), batch_loss = jax.lax.scan(
                _scan_fun, (p, self.opt_state, key), training_data
            )

            if epoch % self.print_every == 0:
                logger.info("Epoch %s, loss: %s", epoch, jnp.mean(batch_loss))

        # Return the optimized parameters
        return unflatten(p)
