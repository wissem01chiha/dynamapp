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

# Test automatically finding simple limit cycles via optimization
#
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from scipy.optimize import minimize

import collimator
from collimator.simulation import SimulatorOptions, estimate_max_major_steps
from collimator.models import VanDerPol, RimlessWheel, CompassGait


# from collimator import logging


# logging.set_file_handler("test.log")


def _test_find_limit_cycle(model, xc0, xd0, tf0, max_major_step_length=1.0):
    context = model.create_context()

    # Since the final time is a traced variable, we can't rely on the built-in
    # heuristic to compute the number of major steps, since this requires
    # static final time.
    max_major_steps = estimate_max_major_steps(model, (0.0, tf0), max_major_step_length)
    options = SimulatorOptions(
        enable_autodiff=True,
        max_major_steps=max_major_steps,
        max_major_step_length=max_major_step_length,
        atol=1e-14,
        rtol=1e-12,
        math_backend="jax",
    )

    # Define a function to compute the difference between the final and initial state
    def shoot(model, context, unflatten, vec):
        xc0, tf = unflatten(vec)
        context = context.with_continuous_state(xc0)
        context = context.with_discrete_state(xd0)

        # Run simulation
        results = collimator.simulate(model, context, (0.0, tf), options=options)
        xf = results.context.continuous_state
        return jnp.linalg.norm(xf - xc0) ** 2

    vec, unflatten = ravel_pytree((xc0, tf0))

    func = jax.jit(partial(shoot, model, context, unflatten))
    jac = jax.jit(jax.grad(func))

    # Solve the problem using scipy minimize
    res = minimize(func, vec, jac=jac, method="BFGS")
    x_opt, tf_opt = unflatten(res.x)

    assert res.success

    # Make sure the optimizer didn't just find the origin fixed point
    assert not jnp.allclose(x_opt, jnp.zeros_like(xc0), atol=1e-2)

    # Check that the final state is close to the initial state
    assert jnp.allclose(func(res.x), 0.0, atol=1e-8)


def test_van_der_pol_lco():
    model = VanDerPol()

    xc0 = jnp.array([1.0, 0.0])
    xd0 = None
    tf0 = 7.0

    _test_find_limit_cycle(model, xc0, xd0, tf0)


def test_rimless_wheel_lco():
    model = RimlessWheel()

    xc0 = jnp.array([0.0, 2.0])
    xd0 = 0.0  # Extra variable tracks "toe" position for visualization
    tf0 = 0.5

    _test_find_limit_cycle(model, xc0, xd0, tf0)


def test_compass_gait_lco():
    model = CompassGait()

    xc0 = jnp.array([0.0, -0.2, 0.4, -2.0])
    xd0 = model.DiscreteStateType(
        0.0, False
    )  # Extra variables track position for visualization
    tf0 = 0.7

    _test_find_limit_cycle(model, xc0, xd0, tf0)
