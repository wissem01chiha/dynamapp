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

# Test case for autotuning PID (optimization use case)
import pytest
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

import collimator
from collimator.optimization import ui_jobs
from collimator.library import (
    FeedthroughBlock,
    Gain,
    Adder,
    Integrator,
    TransferFunction,
    Step,
    PIDDiscrete,
    Saturate,
)
from collimator.simulation import SimulatorOptions

# from collimator import logging
# from collimator.logging import logger


class ParametricGain(collimator.LeafSystem):
    # TODO (before merge): Make the normal Gain behave like this
    def __init__(self, gain, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_input_port()
        self.declare_dynamic_parameter("gain", gain)

        def func(x, gain):
            return gain * x

        def _callback(context):
            inputs, context = self.eval_input(context)
            parameters = context[self.system_id].parameters
            return func(inputs, **parameters), context

        self.declare_output_port(
            _callback, prerequisites_of_calc=[self.input_ports[0].ticket]
        )


def make_cost(Q=1.0, R=0.01):
    builder = collimator.DiagramBuilder()

    error_squared = builder.add(FeedthroughBlock(lambda x: x**2, name="error_squared"))
    control_squared = builder.add(
        FeedthroughBlock(lambda x: x**2, name="control_squared")
    )

    error_weight = builder.add(Gain(Q, name="error_gain"))
    control_weight = builder.add(Gain(R, name="control_gain"))

    cost_signal = builder.add(Adder(2, name="cost_signal"))
    cost_integral = builder.add(Integrator(0.0, name="cost_integral"))

    builder.connect(error_squared.output_ports[0], error_weight.input_ports[0])
    builder.connect(control_squared.output_ports[0], control_weight.input_ports[0])
    builder.connect(error_weight.output_ports[0], cost_signal.input_ports[0])
    builder.connect(control_weight.output_ports[0], cost_signal.input_ports[1])
    builder.connect(cost_signal.output_ports[0], cost_integral.input_ports[0])

    builder.export_input(error_squared.input_ports[0], "e")
    builder.export_input(control_squared.input_ports[0], "u")
    builder.export_output(cost_integral.output_ports[0], "J")

    return builder.build(name="cost")


def make_diagram(R=0.01, kp=1.0, ki=10.0, kd=0.1, dt=0.01, step_time=1.0):
    builder = collimator.DiagramBuilder()
    plant = builder.add(TransferFunction([16.13], [0.00333, 0.201, 1.0], name="plant"))
    reference_signal = builder.add(
        Step(
            start_value=0.0, end_value=1.0, step_time=step_time, name="reference_signal"
        )
    )
    error_signal = builder.add(Adder(2, operators="+-", name="error_signal"))
    controller = builder.add(PIDDiscrete(kp=kp, ki=ki, kd=kd, dt=dt, name="controller"))
    control_limits = builder.add(
        Saturate(lower_limit=0.0, upper_limit=1.0, name="control_limits")
    )

    cost = builder.add(make_cost(R=R))

    builder.connect(controller.output_ports[0], control_limits.input_ports[0])
    builder.connect(control_limits.output_ports[0], plant.input_ports[0])
    builder.connect(reference_signal.output_ports[0], error_signal.input_ports[0])
    builder.connect(plant.output_ports[0], error_signal.input_ports[1])
    builder.connect(error_signal.output_ports[0], controller.input_ports[0])

    builder.connect(error_signal.output_ports[0], cost.input_ports[0])
    builder.connect(controller.output_ports[0], cost.input_ports[1])

    return builder.build()


@pytest.mark.slow
def test_pid_autotune():
    collimator.set_backend("jax")  # Need for autodiff
    diagram = make_diagram()
    t_span = (0.0, 2.0)

    options = SimulatorOptions(
        enable_autodiff=True,
        rtol=1e-8,
        atol=1e-10,
    )

    controller = diagram["controller"]
    cost_integral = diagram["cost"]["cost_integral"]

    @jax.jit
    def forward(k, context):
        controller_context = context[controller.system_id].with_parameters(
            {"kp": k[0], "ki": k[1], "kd": k[2]}
        )
        context = context.with_subcontext(controller.system_id, controller_context)
        results = collimator.simulate(diagram, context, t_span, options=options)
        return results.context[cost_integral.system_id].continuous_state

    k0 = jnp.array([1.0, 0.0, 0.0])
    context = diagram.create_context()

    J0 = forward(k0, context)

    res = minimize(partial(forward, context=context), k0, method="BFGS")
    Jf = res.fun

    print(J0)
    print(res)

    # assert res.success
    assert Jf < J0


def test_pid_autotune_new_api():
    expected_opt_kp = 1.0
    expected_opt_ki = 2.65
    expected_opt_kd = 0.01
    atol = [0.2, 0.5, 0.2]

    collimator.set_backend("jax")  # Need for autodiff
    sim_t_span = (0.0, 2.0)
    job_type = "pid"

    pid_gains_init = [1.0, 0.0, 0.0]

    diagram = make_diagram()
    pid_blocks = [diagram["controller"]]
    objective_port = diagram["cost"]["cost_integral"].output_ports[0]

    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="sequential_least_squares",
        options={"maxiter": 100},
        design_parameters=None,
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        pid_blocks=pid_blocks,
        pid_gains_init=pid_gains_init,
    )

    print(f"{opt_param=}")
    for k, v in opt_param.items():
        assert v >= 0.0
        if k.startswith("kp"):
            assert jnp.isclose(v, expected_opt_kp, atol=atol[0])
        elif k.startswith("ki"):
            assert jnp.isclose(v, expected_opt_ki, atol=atol[1])
        elif k.startswith("kd"):
            assert jnp.isclose(v, expected_opt_kd, atol=atol[2])


if __name__ == "__main__":
    # test_pid_autotune()
    test_pid_autotune_new_api()
