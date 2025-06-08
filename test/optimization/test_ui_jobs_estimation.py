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
Standalone tests for optimization framework. No json processing is performed.
Incoming jobs from the UI are artificially generated.
"""

from pathlib import Path
import platform

import numpy as np
import pytest

from collimator import DiagramBuilder, Parameter
from collimator.library import Adder, Constant, Gain, Integrator
from collimator.optimization import ui_jobs
from collimator.optimization.framework.base.optimizable import DesignParameter


def _here():
    return Path(__file__).parent


def _make_subdiagram(m=1.0, v0=0.1, x0=1.0, c=1.0, k=1.0, min_c_plus_k=3.0):
    model_parameters = {
        "c": Parameter(np.array(c)),
        "k": Parameter(np.array(k)),
    }

    builder = DiagramBuilder()

    # Plant dynamics
    k_x = builder.add(Gain(model_parameters["k"], name="k_x"))
    c_v = builder.add(Gain(model_parameters["c"], name="c_v"))

    adder = builder.add(Adder(3, operators="+--", name="adder"))

    one_over_m = builder.add(Gain(1.0 / m, name="one_over_m"))
    v = builder.add(Integrator(v0, name="v"))
    x = builder.add(Integrator(x0, name="x"))

    builder.connect(k_x.output_ports[0], adder.input_ports[1])
    builder.connect(c_v.output_ports[0], adder.input_ports[2])
    builder.connect(adder.output_ports[0], one_over_m.input_ports[0])
    builder.connect(one_over_m.output_ports[0], v.input_ports[0])
    builder.connect(v.output_ports[0], x.input_ports[0])

    builder.connect(v.output_ports[0], c_v.input_ports[0])
    builder.connect(x.output_ports[0], k_x.input_ports[0])

    # constraint
    c = builder.add(Constant(model_parameters["c"], name="c"))
    k = builder.add(Constant(model_parameters["c"], name="k"))
    min_c_plus_k = builder.add(Constant(min_c_plus_k, name="min_c_plus_k"))
    g = builder.add(Adder(3, operators="++-", name="g"))

    builder.connect(c.output_ports[0], g.input_ports[0])
    builder.connect(k.output_ports[0], g.input_ports[1])
    builder.connect(min_c_plus_k.output_ports[0], g.input_ports[2])

    input_ports = [adder.input_ports[0]]
    output_ports = [x.output_ports[0], v.output_ports[0]]
    constraint_ports = [g.output_ports[0]]

    input_port_indices = []
    for port in input_ports:
        port_index = builder.export_input(port)
        input_port_indices.append(port_index)

    output_port_indices = []
    for port in output_ports:
        port_index = builder.export_output(port)
        output_port_indices.append(port_index)

    consraint_port_indices = []
    for port in constraint_ports:
        port_index = builder.export_output(port)
        consraint_port_indices.append(port_index)

    diagram = builder.build(name="subdiagram", parameters=model_parameters)

    # Get port names
    input_port_names = [diagram.input_ports[idx].name for idx in input_port_indices]
    output_port_names = [diagram.output_ports[idx].name for idx in output_port_indices]
    constraint_port_names = [
        diagram.output_ports[idx].name for idx in consraint_port_indices
    ]

    input_columns = ["F_t.out_0"]  # in data file
    output_columns = ["x.out_0", "v.out_0"]  # in data file

    # mapping between input_port_names and input_columns
    input_port_names_to_column_names = dict(zip(input_port_names, input_columns))

    # mapping between output_port_names and output_columns
    output_port_names_to_column_names = dict(zip(output_port_names, output_columns))

    return (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        constraint_port_names,
    )


@pytest.mark.slow
def test_optimization_unbounded():
    data_file = _here() / "SpringMass-estimation-0p11_1p11.csv"
    time_column = "time"
    sim_t_span = (0.0, 2.0)

    job_type = "estimation"
    true_c = 0.11
    true_k = 1.11

    design_parameters = [
        DesignParameter(param_name="c", initial=0.5, min=-np.inf, max=np.inf),
        DesignParameter(param_name="k", initial=0.5, min=-np.inf, max=np.inf),
    ]

    # Scipy with L-BFGS-B
    (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        _,
    ) = _make_subdiagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="l_bfgs_b",
        design_parameters=design_parameters,
        options={"maxiter": 20},
        sim_t_span=sim_t_span,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=[],
        print_every=10,
    )
    print(f"{opt_param=}")
    assert np.isclose(opt_param["c"], true_c, atol=0.1)
    assert np.isclose(opt_param["k"], true_k, atol=0.1)

    # Scipy Nelder-Mead
    (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        _,
    ) = _make_subdiagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="nelder_mead",
        design_parameters=design_parameters,
        options={"maxiter": 20},
        sim_t_span=sim_t_span,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=[],
        print_every=10,
    )
    print(f"{opt_param=}")
    assert np.isclose(opt_param["c"], true_c, atol=0.1)
    assert np.isclose(opt_param["k"], true_k, atol=0.1)

    # Optax sgd
    (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        _,
    ) = _make_subdiagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="stochastic_gradient_descent",
        design_parameters=design_parameters,
        options={"learning_rate": 0.01, "num_epochs": 2000},
        sim_t_span=sim_t_span,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=[],
        print_every=200,
    )
    assert np.isclose(opt_param["c"], true_c, atol=0.1)
    assert np.isclose(opt_param["k"], true_k, atol=0.1)

    # Evosax PSO
    (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        _,
    ) = _make_subdiagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="particle_swarm_optimization",
        design_parameters=design_parameters,
        options={"pop_size": 50, "num_generations": 100},
        sim_t_span=sim_t_span,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=[],
        print_every=10,
    )
    print(f"{opt_param=}")
    assert np.isclose(opt_param["c"], true_c, atol=0.1)
    assert np.isclose(opt_param["k"], true_k, atol=0.1)


@pytest.mark.slow
def test_optimization_bounded():
    data_file = _here() / "SpringMass-estimation-0p11_1p11.csv"
    time_column = "time"
    sim_t_span = (0.0, 2.0)

    job_type = "estimation"
    expected_c = 0.15
    expected_k = 0.8

    design_parameters = [
        DesignParameter(param_name="c", initial=0.5, min=0.15, max=np.inf),
        DesignParameter(param_name="k", initial=0.5, min=0.0, max=0.8),
    ]

    # Scipy with L-BFGS-B
    (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        _,
    ) = _make_subdiagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="l_bfgs_b",
        options={"maxiter": 20},
        design_parameters=design_parameters,
        sim_t_span=sim_t_span,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=[],
        print_every=10,
    )
    print(f"{opt_param=}")
    assert np.isclose(opt_param["c"], expected_c, atol=0.1)
    assert np.isclose(opt_param["k"], expected_k, atol=0.1)

    # Evosax with PSO
    (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        _,
    ) = _make_subdiagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="particle_swarm_optimization",
        options={"pop_size": 50, "num_generations": 100},
        design_parameters=design_parameters,
        sim_t_span=sim_t_span,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=[],
        print_every=10,
    )
    print(f"{opt_param=}")
    assert np.isclose(opt_param["c"], expected_c, atol=0.1)
    assert np.isclose(opt_param["k"], expected_k, atol=0.1)


@pytest.mark.slow
def test_optimization_constrained_scipy():
    data_file = _here() / "SpringMass-estimation-0p11_1p11.csv"
    time_column = "time"
    sim_t_span = (0.0, 2.0)

    job_type = "estimation"
    expected_c = 0.65
    expected_k = 1.43

    min_c_plus_k = 1.3

    design_parameters = [
        DesignParameter(param_name="c", initial=0.8, min=0.0, max=2.0),
        DesignParameter(param_name="k", initial=0.8, min=0.0, max=2.0),
    ]

    # Scipy SLSQP
    (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        constraint_port_names,
    ) = _make_subdiagram(min_c_plus_k=min_c_plus_k)
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="sequential_least_squares",
        options={"maxiter": 100},
        design_parameters=design_parameters,
        sim_t_span=sim_t_span,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=constraint_port_names,
        print_every=10,
    )
    print(f"{opt_param=}")
    assert np.isclose(opt_param["c"], expected_c, atol=0.1)
    assert np.isclose(opt_param["k"], expected_k, atol=0.1)


@pytest.mark.slow
@pytest.mark.skipif(
    platform.machine() == "arm64", reason="nlopt not available for arm64"
)
def test_optimization_constrained_nlopt():
    data_file = _here() / "SpringMass-estimation-0p11_1p11.csv"
    time_column = "time"
    sim_t_span = (0.0, 2.0)

    job_type = "estimation"
    expected_c = 0.65
    expected_k = 1.43

    min_c_plus_k = 1.3

    design_parameters = [
        DesignParameter(param_name="c", initial=0.8, min=0.0, max=2.0),
        DesignParameter(param_name="k", initial=0.8, min=0.0, max=2.0),
    ]

    # NLopt Direct
    (
        diagram,
        input_port_names_to_column_names,
        output_port_names_to_column_names,
        constraint_port_names,
    ) = _make_subdiagram(min_c_plus_k=min_c_plus_k)
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="dividing_rectangles",
        options={},
        design_parameters=design_parameters,
        sim_t_span=sim_t_span,
        data_file=data_file,
        time_column=time_column,
        input_port_names_to_column_names=input_port_names_to_column_names,
        output_port_names_to_column_names=output_port_names_to_column_names,
        constraint_port_names=constraint_port_names,
        print_every=10,
    )
    print(f"{opt_param=}")
    assert np.isclose(opt_param["c"], expected_c, atol=0.1)
    assert np.isclose(opt_param["k"], expected_k, atol=0.1)
    assert opt_param["c"] + opt_param["k"] >= min_c_plus_k


if __name__ == "__main__":
    test_optimization_unbounded()
    test_optimization_bounded()
    test_optimization_constrained_scipy()
    test_optimization_constrained_nlopt()
