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
Standalone tests for optimization framework. These are separated from any UI or json
processing.
"""

import platform
import pytest
import numpy as np
from collimator import DiagramBuilder, Parameter
from collimator.library import (
    Adder,
    Constant,
    Gain,
    Integrator,
    Power,
    PID,
    PIDDiscrete,
)
from collimator.optimization import ui_jobs

from collimator.optimization.framework.base.optimizable import (
    Distribution,
    StochasticParameter,
)


def _make_diagram(discrete_pid=False, m=1.0, c=1.0, v0=0.1, x0=1.0):
    model_parameters = {
        "c": Parameter(np.array(1.0)),
        "k": Parameter(np.array(1.0)),
    }

    builder = DiagramBuilder()

    if discrete_pid:
        dt = 0.01
        pid = builder.add(PIDDiscrete(dt, 1.0, 1.0, 1.0, name="pid"))
    else:
        pid = builder.add(PID(1.0, 1.0, 1.0, 100, name="pid"))

    k_x = builder.add(Gain(model_parameters["k"], name="k_x"))
    c_v = builder.add(Gain(model_parameters["c"], name="c_v"))
    adder = builder.add(Adder(3, operators="+--", name="adder"))
    one_over_m = builder.add(Gain(1.0 / m, name="one_over_m"))
    v = builder.add(Integrator(v0, name="v"))
    x = builder.add(Integrator(x0, name="x"))

    builder.connect(pid.output_ports[0], adder.input_ports[0])
    builder.connect(k_x.output_ports[0], adder.input_ports[1])
    builder.connect(c_v.output_ports[0], adder.input_ports[2])
    builder.connect(adder.output_ports[0], one_over_m.input_ports[0])
    builder.connect(one_over_m.output_ports[0], v.input_ports[0])
    builder.connect(v.output_ports[0], x.input_ports[0])

    builder.connect(v.output_ports[0], c_v.input_ports[0])
    builder.connect(x.output_ports[0], k_x.input_ports[0])

    ref_x = builder.add(Constant(0.0, name="ref_x"))
    ref_v = builder.add(Constant(0.0, name="ref_v"))

    err_v = builder.add(Adder(2, operators="+-", name="err_v"))
    err_x = builder.add(Adder(2, operators="+-", name="err_x"))
    sq_err_v = builder.add(Power(2.0, name="sq_err_v"))
    sq_err_x = builder.add(Power(2.0, name="sq_err_x"))
    cost_v = builder.add(Integrator(0.0, name="cost_v"))
    cost_x = builder.add(Integrator(0.0, name="cost_x"))
    objective = builder.add(Adder(2, operators="++", name="objective"))

    builder.connect(ref_v.output_ports[0], err_v.input_ports[0])
    builder.connect(v.output_ports[0], err_v.input_ports[1])
    builder.connect(ref_x.output_ports[0], err_x.input_ports[0])
    builder.connect(x.output_ports[0], err_x.input_ports[1])

    builder.connect(err_v.output_ports[0], sq_err_v.input_ports[0])
    builder.connect(err_x.output_ports[0], sq_err_x.input_ports[0])
    builder.connect(sq_err_v.output_ports[0], cost_v.input_ports[0])
    builder.connect(sq_err_x.output_ports[0], cost_x.input_ports[0])
    builder.connect(cost_v.output_ports[0], objective.input_ports[0])
    builder.connect(cost_x.output_ports[0], objective.input_ports[1])

    builder.connect(err_x.output_ports[0], pid.input_ports[0])

    # constraints
    constraint = builder.add(Adder(2, operators="+-", name="constraint"))
    min_objective_value = builder.add(Constant(1.67, name="min_objective_value"))
    builder.connect(objective.output_ports[0], constraint.input_ports[0])
    builder.connect(min_objective_value.output_ports[0], constraint.input_ports[1])

    diagram = builder.build(parameters=model_parameters)

    return (
        diagram,
        diagram["objective"].output_ports[0],
        pid,
    )


@pytest.mark.parametrize("discrete_pid", [False, True])
@pytest.mark.slow
def test_optimization_pid(discrete_pid):
    sim_t_span = (0.0, 2.0)
    job_type = "pid"

    expected_opt_kp = 0.0
    expected_opt_ki = 0.0
    expected_opt_kd = 1.0
    atol = 0.2

    # Scipy
    diagram, objective_port, pid = _make_diagram(discrete_pid=discrete_pid)
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        # algorithm="l_bfgs_b",
        # algorithm="sequential_least_squares",
        algorithm="nelder_mead",
        options={"maxiter": 50},
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        pid_blocks=[pid],
        print_every=10,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["pid.Kp"], expected_opt_kp, atol=atol)
    np.testing.assert_allclose(opt_param["pid.Ki"], expected_opt_ki, atol=atol)
    np.testing.assert_allclose(opt_param["pid.Kd"], expected_opt_kd, atol=atol)

    # Verify proper metadata is attached to the returned value
    assert opt_param["pid.Kp"].system == pid

    # Evosax PSO
    diagram, objective_port, pid = _make_diagram(discrete_pid=discrete_pid)
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="particle_swarm_optimization",
        # algorithm="covariance_matrix_adaptation_evolution_strategy",
        # algorithm="simulated_annealing",
        options={"pop_size": 200, "num_generations": 100},
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        pid_blocks=[pid],
        print_every=20,
    )
    print(f"{opt_param=}")

    # TODO: Check why PSO intermittently fails. Perhaps something due to
    # initial population with pid_gains_max being 1e03, and garbage sim output for
    # high gains
    check_PSO_results = False
    if check_PSO_results:
        np.testing.assert_allclose(opt_param["pid.Kp"], expected_opt_kp, atol=atol)
        np.testing.assert_allclose(opt_param["pid.Ki"], expected_opt_ki, atol=atol)
        np.testing.assert_allclose(opt_param["pid.Kd"], expected_opt_kd, atol=atol)

    # NLopt direct
    if platform.machine() != "arm64":
        diagram, objective_port, pid = _make_diagram(discrete_pid=discrete_pid)
        opt_param, _ = ui_jobs.jobs_router(
            job_type,
            diagram,
            # algorithm="dividing_rectangles",
            algorithm="method_of_moving_asymptotes",
            options={},
            sim_t_span=sim_t_span,
            objective_port=objective_port,
            pid_blocks=[pid],
            print_every=10,
        )
        print(f"{opt_param=}")
        np.testing.assert_allclose(opt_param["pid.Kp"], expected_opt_kp, atol=atol)
        np.testing.assert_allclose(opt_param["pid.Ki"], expected_opt_ki, atol=atol)
        np.testing.assert_allclose(opt_param["pid.Kd"], expected_opt_kd, atol=atol)

    # Optax
    diagram, objective_port, pid = _make_diagram(discrete_pid=discrete_pid)
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        # algorithm="adam",
        algorithm="adam",
        options={"learning_rate": 0.01, "num_epochs": 500},
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        pid_blocks=[pid],
        print_every=100,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["pid.Kp"], expected_opt_kp, atol=atol)
    np.testing.assert_allclose(opt_param["pid.Ki"], expected_opt_ki, atol=atol)
    np.testing.assert_allclose(opt_param["pid.Kd"], expected_opt_kd, atol=atol)


@pytest.mark.parametrize("discrete_pid", [False, True])
@pytest.mark.slow
def test_optimization_pid_stochastic(discrete_pid):
    sim_t_span = (0.0, 2.0)
    job_type = "pid"

    expected_opt_kp = 0.0
    expected_opt_ki = 0.0
    expected_opt_kd = 1.0
    atol = 0.2

    stochastic_var = StochasticParameter(
        param_name="k",
        distribution=Distribution(name="uniform", options={"min": 0.5, "max": 1.5}),
    )

    # Optax
    diagram, objective_port, pid = _make_diagram(discrete_pid=discrete_pid)
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="rmsprop",
        options={"learning_rate": 0.01, "num_epochs": 100},
        stochastic_parameters=[stochastic_var],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        pid_blocks=[pid],
        print_every=10,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["pid.Kp"], expected_opt_kp, atol=atol)
    np.testing.assert_allclose(opt_param["pid.Ki"], expected_opt_ki, atol=atol)
    np.testing.assert_allclose(opt_param["pid.Kd"], expected_opt_kd, atol=atol)


if __name__ == "__main__":
    test_optimization_pid(discrete_pid=False)
    test_optimization_pid_stochastic(discrete_pid=False)
    test_optimization_pid(discrete_pid=True)
    test_optimization_pid_stochastic(discrete_pid=True)
