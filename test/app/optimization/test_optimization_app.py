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

import dataclasses
import platform
import shutil

import numpy as np
import pytest

from collimator.cli.run_optimization import run_optimization
import collimator.testing as testing


# Note about atol:
# Adam stochastic is configured in a way that it will not reach the optimal
# solution, in order to make the test somewhat faster.
# We care mostly about validating that json ingestion works, not so much about the
# actual optimization results. Set a seed to make the test deterministic, and a
# VERY high tolerance because the result is quite bad.


@dataclasses.dataclass
class OptTestScenario:
    expected: dict
    atol: float = 0.1
    dir: str = "."
    request: str = "request.json"
    model: str = "model.json"
    datafiles: list[str] = dataclasses.field(default_factory=list)


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "scenario",
    [
        OptTestScenario(
            dir="design_optimization",
            request="request_adam.json",
            expected={"c": 1.6524792908656067},
        ),
        OptTestScenario(
            dir="design_optimization",
            request="request_adam_stochastic.json",
            expected={"c": 1.6524792908656067},
            atol=2.0,
        ),
        OptTestScenario(
            dir="design_optimization",
            request="request_pso.json",
            expected={"c": 1.6524792908656067},
        ),
        OptTestScenario(
            dir="pid_tuning",
            request="request.json",
            expected={
                "PID_0.Kp": 2.9439373500991484,
                "PID_0.Ki": 3.9832554431122977,
                "PID_0.Kd": 0.866736028389201,
            },
        ),
        OptTestScenario(
            dir="param_estimation",
            request="request.json",
            expected={
                "Kp": 0.041262037085867107,
                "Ki": 1.0,
                "Kd": 0.041262036148298556,
            },
            datafiles=[
                "recorded-results.csv",
                "submodel-fdc5b41f-73ba-4e80-8522-67a883e0feea-latest.json",
            ],
        ),
    ],
)
@pytest.mark.skipif(platform.system() == "Emscripten", reason="No support for WASM")
def test_optimization_api(request, scenario: OptTestScenario):
    np.random.seed(0)

    print("running optimization test scenario:", scenario)

    test_paths = testing.get_paths(request)
    testdir = test_paths["testdir"]
    workdir = test_paths["workdir"]

    all_files = scenario.datafiles + [scenario.request, scenario.model]
    print("all_files:", all_files)

    for datafile in all_files:
        shutil.copyfile(testdir / scenario.dir / datafile, workdir / datafile)

    with testing.set_cwd(workdir):
        optimal_params, _ = run_optimization(
            request=scenario.request,
            model=scenario.model,
        )
        print("optimal_params:", optimal_params)
        assert optimal_params is not None

        for param_name, expected_value in scenario.expected.items():
            print("expected_value for", param_name, ":", expected_value)
            p = optimal_params.get(param_name)
            print("optimal_value:", type(p), p)
            assert np.isclose(p, expected_value, atol=scenario.atol)
