#!/bin/env pytest
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

import pytest
import collimator.testing as test
import subprocess as sp
import collimator
import pathlib
import matplotlib.pyplot as plt
from collimator.backend import numpy_api as cnp

pytestmark = pytest.mark.app


def test_load(request):
    test.run(pytest_request=request)


def test_cli():
    cmd = [
        "collimator_cli",
        "run",
        "--modeldir",
        "test/models/app/test_load/",
        "--model",
        "model.json",
    ]
    p = sp.Popen(cmd)
    _ = p.wait()


def test_simset():
    testdir = pathlib.Path(__file__).parent
    load_model = collimator.load_model(testdir, model="simset.json")

    assert load_model.simulator_options.max_minor_step_size == 33.0
    assert load_model.simulator_options.min_minor_step_size == 22.0
    assert load_model.simulator_options.rtol == 99.0
    assert load_model.simulator_options.atol == 55.0
    assert load_model.results_options.max_results_interval == 100000


# Is this test still necessary? or can these things be tested from wildcat directly?
def test_feedthru_bool():
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="feedthru_bool.json")

    non_feedthrough_blk_names = []
    feedthrough_blk_names = [
        "DerivativeDiscrete_1",
        "FilterDiscrete_1",
        "PID_Discrete_1",
        "PythonScript_0",  # Discrete mode (still has feedthrough)
        "PythonScript_1",  # Agnostic mode
    ]

    for node in model.diagram.nodes:
        print(f"Node: {node.name} ft={node.get_feedthrough()}")
        if node.name in non_feedthrough_blk_names:
            assert not node.get_feedthrough()
        if node.name in feedthrough_blk_names:
            assert node.get_feedthrough()


@pytest.mark.skip(reason="development test")
def test_impact_detection_results(show_plot=False):
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="double_ball.json")
    r = model.simulate(stop_time=1.0)

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(r["time"], r["BouncingBall1.Position"], label="pos1")
        ax.plot(r["time"], r["BouncingBall2.Position"], label="pos2")
        ax.plot(r["time"], r["BouncingBall1.ImpactDetection.out_0"], label="impact1")
        ax.plot(r["time"], r["BouncingBall2.ImpactDetection.out_0"], label="impact2")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # test_simset()
    cnp.set_backend("numpy")
    test_impact_detection_results(show_plot=True)
