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

import pickle

import matplotlib.pyplot as plt

import numpy as np
import jax
import jax.numpy as jnp

import collimator
from collimator import library
from collimator.optimization import Trainer

from controllers import make_mlp_controller

RTOL = 1e-6
ATOL = 1e-8


def make_cost_system(Q, R, name="cost"):
    builder = collimator.DiagramBuilder()
    lqr_cost = library.QuadraticCost(Q, R, name="lqr_cost")
    running_cost = library.Integrator(initial_state=0.0, name="running_cost")
    builder.add(lqr_cost, running_cost)
    builder.connect(lqr_cost.output_ports[0], running_cost.input_ports[0])
    builder.export_input(lqr_cost.input_ports[0], name="x")
    builder.export_input(lqr_cost.input_ports[1], name="u")
    builder.export_output(running_cost.output_ports[0])
    return builder.build(name=name)


# Full closed-loop system
def make_cl_system(
    plant,
    nn_config,
    reference=None,
    dt=1.0,
    Q=1.0,
    R=1e-2,
    lbu=-np.inf,
    ubu=np.inf,
    sigma=0.0,
    filter_coefficient=100.0,
    delay=0.0,
    name="cl_system",
):
    builder = collimator.DiagramBuilder()

    plant = builder.add(plant)

    # Add a reference input which will also be fed to the NN controller
    if reference is None:
        reference = 0.0

    reference = library.Constant(reference, name="reference")
    builder.add(reference)

    error = builder.add(library.Adder(2, operators="-+", name="error"))
    builder.connect(plant.output_ports[0], error.input_ports[0])
    builder.connect(reference.output_ports[0], error.input_ports[1])

    # Create the NN controller
    controller = builder.add(
        make_mlp_controller(
            nn_config,
            dt=dt,
            sigma=sigma,
            filter_coefficient=filter_coefficient,
            delay=delay,
            name="controller",
        )
    )

    # Combine the reference and plant output for the controller inputs
    builder.connect(error.output_ports[0], controller.input_ports[1])
    saturate = builder.add(
        library.Saturate(lower_limit=lbu, upper_limit=ubu, name="saturate")
    )
    builder.connect(controller.output_ports[0], saturate.input_ports[0])
    builder.connect(saturate.output_ports[0], plant.input_ports[0])

    # Feed the saturated control signal back to the controller
    # This is unused in this case, but is needed for compatibility with
    # other controllers that use, e.g. EKF estimators
    builder.connect(saturate.output_ports[0], controller.input_ports[0])

    # Cost-tracking subsystem
    cost = builder.add(make_cost_system(Q, R, name="cost"))
    builder.connect(controller.output_ports[1], cost.input_ports[0])
    builder.connect(saturate.output_ports[0], cost.input_ports[1])

    return builder.build(name=name)


class QubeTrainer(Trainer):
    def __init__(self, *args, nn_id, ref_id, plant_id, **kwargs):
        self.nn_id = nn_id
        self.ref_id = ref_id
        self.plant_id = plant_id
        super().__init__(*args, **kwargs)

    # Model-specific: what parameters can we optimize?
    def optimizable_parameters(self, context):
        return context[self.nn_id].parameters["mlp_params"]

    # Model-specific updates to add the sample data, parameters, etc.
    # This should be the combination of the output of `optimizable_parameters`
    # along with all the per-simulation "training data".  Parameters will
    # update once per epoch, and training data will update once per sample.
    def prepare_context(self, context, mlp_params, x0, key):
        nn_context = context[self.nn_id].with_parameter("mlp_params", mlp_params)
        plant_context = context[self.plant_id].with_continuous_state(x0)
        noise = self.simulator.system["controller"]["noise"]

        noise_context = context[noise.system_id].with_discrete_state(
            noise.RNGState(key, noise.sample(key, shape=(noise.N, *noise.shape)))
        )
        context = (
            context.with_subcontext(self.nn_id, nn_context)
            .with_subcontext(self.plant_id, plant_context)
            .with_subcontext(noise.system_id, noise_context)
        )
        return context

    # Model-specific cost function
    def evaluate_cost(self, context):
        system = self.simulator.system
        T = context.time
        J = (1 / T) * system["cost"].output_ports[0].eval(context)
        return J


def plot_rollout(results):
    fig, ax = plt.subplots(3, 1, figsize=(7, 3), sharex=True)
    ax[-1].set_xlabel("$t$")

    ax[0].plot(results.time, results.outputs["x"][:, 1])
    ax[0].plot(results.time, results.outputs["r"][:, 1], "k")
    ax[0].set_ylabel("$x$")
    ax[0].grid()

    ax[1].plot(results.time, results.outputs["u"])
    ax[1].set_ylabel("$u$")
    ax[1].grid()

    ax[2].plot(results.time, results.outputs["J"])
    ax[2].set_ylabel("$J$")
    ax[2].grid()
    plt.show()


def run_rollout(nn_config, sys_config, p_opt, plant=None, x0=None, tf=10.0, plot=True):
    if plant is None:
        plant = library.QubeServoModel(full_state_output=False, name="plant")

    # Simulate the system with the trained controller
    system = make_cl_system(plant, nn_config, **sys_config)
    context = system.create_context()

    nn_id = system["controller"]["net"].system_id
    nn_context = context[nn_id].with_parameter("mlp_params", p_opt)
    context = context.with_subcontext(nn_id, nn_context)

    if x0 is not None:
        plant_id = system["plant"].system_id
        plant_context = context[plant_id].with_continuous_state(x0)
        context = context.with_subcontext(plant_id, plant_context)

    options = collimator.SimulatorOptions(
        max_major_steps=1000,
        save_time_series=True,
        max_major_step_length=1.0,
        rtol=1e-6,
        atol=1e-8,
    )
    recorded_signals = {
        "x": system["plant"].output_ports[0],
        "r": system["reference"].output_ports[0],
        "u": system["plant"].input_ports[0],
        "u0": system["controller"]["net"].output_ports[0],
        "J": system["cost"].output_ports[0],
        "e": system["error"].output_ports[0],
        "w": system["controller"]["noisy_ctrl"].output_ports[0],
    }

    results = collimator.simulate(
        system, context, (0.0, tf), options=options, recorded_signals=recorded_signals
    )

    if plot:
        plot_rollout(results)

    return results


# Run a full train-rollout experiment
def run_experiment(
    nn_config,
    sys_config,
    tf=10.0,
    lr=1e-2,
    N=1000,
    epochs=50,
    batch_size=None,
    x0_std=1.0,
    x0=None,
    key=None,
    init_params=None,
    opt_state=None,
):
    plant = library.QubeServoModel(full_state_output=False, name="plant")
    system = make_cl_system(plant, nn_config, **sys_config)
    context = system.create_context()

    # Generate random initial conditions
    if key is None:
        key = jax.random.PRNGKey(seed=nn_config["seed"])

    if x0 is None:
        key, subkey = jax.random.split(key)
        x_up = np.array([0.0, np.pi, 0.0, 0.0])  # Up position as reference
        x0 = x_up + x0_std * jax.random.normal(subkey, (N, 4))

    if batch_size is None:
        batch_size = N

    # Reshape the initial conditions into a batch
    x0 = jnp.reshape(x0, (-1, batch_size, *x0.shape[1:]))

    training_data = (x0,)

    options = collimator.SimulatorOptions(
        max_major_steps=200,
        enable_autodiff=True,
        rtol=1e-6,
        atol=1e-8,
    )
    sim = collimator.Simulator(system, options=options)
    trainer = QubeTrainer(
        sim,
        context,
        nn_id=system["controller"]["net"].system_id,
        ref_id=system["reference"].system_id,
        plant_id=system["plant"].system_id,
        lr=lr,
        optimizer="adam",
    )
    opt_params = trainer.train(
        training_data,
        sim_start_time=0.0,
        sim_stop_time=tf,
        epochs=epochs,
        key=key,
        params=init_params,
        opt_state=opt_state,
    )

    return opt_params, trainer.opt_state


if __name__ == "__main__":
    nn_config = {
        "seed": 0,
        "in_size": 4,
        "out_size": 1,
        "width_size": 16,
        "depth": 4,
        "activation_str": "swish",
    }

    # Weight the states higher than the controls so that the network doesn't
    # learn a suboptimal strategy of just leaving the pendulum in the down
    # position
    Q = np.diag([1.0, 1.0, 1e-2, 1e-2])
    R = 1e-2 * np.eye(1)
    dt = 1 / 500
    sys_config = {
        "dt": dt,
        "Q": Q,
        "R": R,
        "lbu": -10,
        "ubu": 10,
        "reference": np.array([0.0, np.pi]),
        "sigma": 1e-2,
    }

    tf = 0.5  # Time horizon for training
    N = 100  # Number of random initial conditions
    x0 = np.zeros((N, 4)) + np.random.randn(N, 4) * 0.1
    key = jax.random.PRNGKey(0)
    opt_params, opt_state = run_experiment(
        nn_config=nn_config,
        sys_config=sys_config,
        tf=tf,
        lr=1e-3,
        epochs=1000,
        batch_size=10,
        x0=x0,
        key=key,
    )

    with open("models/swingup.pkl", "wb") as f:
        pickle.dump((opt_params, opt_state, nn_config, sys_config), f)
