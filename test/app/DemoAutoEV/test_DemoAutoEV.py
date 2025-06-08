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
import collimator
from collimator.models import CompactEV, DummyBlock
from collimator.simulation import SimulatorOptions
from matplotlib import pyplot as plt

# import timeit
from collimator.library import (
    DataSource,
)
import pathlib


pytestmark = pytest.mark.app


@pytest.mark.skip(reason="~10 minute JIT compile time")
def test_DemoCompactEv(request):
    # just running test with failing is the test
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "ECE.csv")
    test.run(test_paths=test_paths, stop_time=10)


@pytest.mark.skip(reason="~10 minute JIT compile time")
def test_DemoCompactEv_no_pytest(stop_time=10):
    # manually build testpaths so we dont need pytest request arg passed in.
    # FIXME why not use the pytest_request arg?
    test_paths = test.get_paths(
        None,
        testdir_=pathlib.Path(__file__).parent,
        test_name_="test_DemoCompactEv_no_pytest",
    )
    test_paths["workdir"] = test_paths["testdir"]
    print(test_paths)
    # test.copy_to_workdir(test_paths, "ECE.csv")
    test.run(test_paths=test_paths, stop_time=stop_time)


def build_nested(idx, n=2, nest_this=None):
    builder = collimator.DiagramBuilder()
    for i in range(n):
        builder.add(DummyBlock(name=f"dummy{str(i)}"))
    if nest_this is not None:
        builder.add(nest_this)
    diagram = builder.build(name=f"dummy_diagram{str(idx)}")
    return diagram


@pytest.mark.skip(reason="development test")
def test_CompactEvLeaf():
    dt = 0.1
    builder = collimator.DiagramBuilder()
    ds = builder.add(
        DataSource(
            file_name="test/app/DemoCompactEv/ECE.csv",
            interpolation="linear",
            time_samples_as_column=True,
            name="ds",
        )
    )
    cev = builder.add(CompactEV(dt=dt))

    # dummy_count = 0
    # for i in range(dummy_count):
    #     builder.add(DummyBlock(name=f"dummy{str(i)}"))

    # dummy_diagram_count = 10
    # nested_dummy_count = 5
    # for i in range(dummy_diagram_count):
    #     nest_this = build_nested(i + 10000, n=nested_dummy_count)
    #     block = build_nested(i, n=nested_dummy_count, nest_this=nest_this)
    #     builder.add(block)

    builder.connect(ds.output_ports[0], cev.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()

    print(diagram.tree)

    end_time = 200.0

    options = SimulatorOptions(
        max_major_steps=int(1.1 * int(end_time // dt)),
        # max_major_step_length=0.1,
    )

    recorded_signals = {
        "dc": ds.output_ports[0],
        "state": cev.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, end_time),
        options=options,
        recorded_signals=recorded_signals,
    )

    return results


def plot_compact_ev(results):
    time = results.outputs["time"]

    # print(time)
    # print(results)
    # print(results.outputs["state"][:, 1])

    # self.velocity_state_idx = 0
    # self.position_state_idx = 1
    # self.accel_loop_breaker_state_idx = 2
    # self.driver_pid_iterm_state_idx = 3
    # self.coolant_temp_state_idx = 4
    # self.battery_soc_state_idx = 5
    # self.current_loop_breaker_state_idx = 6
    # self.driver_pid_iterm2_state_idx = 7

    print(f"end time={time[-1]}")

    fig, ax = plt.subplots(3, 1, figsize=(8, 4))
    ax[0].plot(time, results.outputs["state"][:, 0], label="vel")
    # ax[0].plot(time, results.outputs["state"][:, 1], label="pos")
    ax[0].plot(time, results.outputs["state"][:, 2], label="accel loop break state")
    # ax[0].plot(time, results.outputs["state"][:, 3], label="3")
    ax[0].legend()

    ax[1].plot(time, results.outputs["state"][:, 4], label="coolant temp")
    ax[1].plot(time, results.outputs["dc"], label="dc")
    ax[1].legend()

    ax[2].plot(time, results.outputs["state"][:, 5], label="soc")
    ax[2].set_ylim([-0.01, 1.01])
    ax[2].legend()
    plt.plot()
    plt.show()


@pytest.mark.skip(reason="development test")
def test_TestHarnessAutoXms(request):
    # thos version is the full test harness, but the controller is implemented
    # using a StateMachine.
    test.run(pytest_request=request, model_json="TestHarness_AutoXms.json")


@pytest.mark.skip(reason="development test")
def test_ControllerAutoXms(request):
    # just the controller, but using the old core block implementation
    # this test is interesting because at creation time, it was causing
    # recursion limit error in wildcat.
    test.run(pytest_request=request, model_json="ControllerAutoXms.json")


# @pytest.mark.skip(reason="development test")
# def test_ControllerAutoXms_flat(request):
#     # just the controller, but using the old core block implementation
#     # this test is interesting because at creation time, it was causing
#     # recursion limit error in wildcat.
#     test.run(pytest_request=request, model_json="ControllerAutoXms_flat.json")

#     assert False


if __name__ == "__main__":
    # number = 3
    # results = timeit.timeit(test_CompactEvLeaf, number=number)
    # print(f"timit time={results/number}")

    # results = test_CompactEvLeaf()
    # plot_compact_ev(results)
    test_DemoCompactEv_no_pytest(stop_time=200)
