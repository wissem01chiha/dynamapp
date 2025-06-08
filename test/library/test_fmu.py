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

import os
import sys
import pandas as pd

import collimator
from collimator.library import (
    Constant,
    ModelicaFMU,
)
from matplotlib import pyplot as plt

import pytest

from collimator import logging
import numpy as np


pytestmark = pytest.mark.slow


logging.set_file_handler("test.log")


@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
def test_FMU_Thermal(show_plot=False):
    """
    this test was for development reasons. later improved with ref from fmusim
    load <filename>.fmu
    create wildcat ModelicaFMU block object
    connect to some inputs
    attempt simulation
    """
    # .fmu file
    abspath = os.path.abspath(__file__)
    library_dir, _ = os.path.split(abspath)
    test_dir, _ = os.path.split(library_dir)
    fmu_file = os.path.join(test_dir, "app", "ModelicaFMU", "thermal_1.fmu")
    fmu_ref_csv = os.path.join(test_dir, "app", "ModelicaFMU", "fmusim.csv")
    print(f"fmu_file={fmu_file}")

    # system builder
    builder = collimator.DiagramBuilder()

    # blocks
    ctrl = builder.add(Constant(8e4, name="ctrl"))
    fmu_block = builder.add(ModelicaFMU(fmu_file, 1.0, name="fmu_block"))

    # connections
    builder.connect(ctrl.output_ports[0], fmu_block.input_ports[0])

    # prep model and run simulation
    diagram = builder.build()
    context = diagram.create_context()

    recorded_signals = {
        "ctrl": ctrl.output_ports[0],
        "fmu_block_o": fmu_block.output_ports[0],
    }
    r = collimator.simulate(
        diagram, context, (0.0, 100.0), recorded_signals=recorded_signals
    )

    # Generate reference fmusim.csv with:
    # fmusim --stop-time 100 --output-interval 1 --output-file fmusim.csv \
    #   --input-file input.csv thermal_1.fmu
    # @am. I think fmusim reference is misleading us here. if we make our FMU block
    # pass this test exactly, then FMU block clock output values do not match simulation
    # clock values. As such, I have 'fixed' the FMU block implementation, and 'tweaked'
    # the expected results here.
    expected_temp = pd.read_csv(fmu_ref_csv, index_col="time")["y"]

    if show_plot:
        fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
        ax1.plot(r.time, r.outputs["ctrl"], label="ctrl")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(r.time[:-1], r.outputs["fmu_block_o"][1:], label="fmu_block_o")
        ax2.plot(r.time[:-1], expected_temp[:-1], label="expected_temp")
        ax2.grid(True)
        ax2.legend()

        plt.show()

    fmu_block_o = r.outputs["fmu_block_o"]  # [:, 0]
    if not np.allclose(fmu_block_o[1:], expected_temp[:-1]):
        print("expected_temp:", expected_temp)
        print("fmu_block_o:", fmu_block_o)
        assert False, "numerical results don't match"


start_times = [0.0, 5.0]


@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
@pytest.mark.parametrize("start_time", start_times)
def test_FMU_StartTime(start_time, show_plot=False):
    """
    this test verifies that setting 'start_time' has the expected outcome.
    """
    # .fmu file
    abspath = os.path.abspath(__file__)
    library_dir, _ = os.path.split(abspath)
    test_dir, _ = os.path.split(library_dir)
    print(f"test_dir={test_dir}")
    fmu_file = os.path.join(test_dir, "app", "ModelicaFMU", "fmu_clock.fmu")

    print(f"fmu_file={fmu_file}")

    # system builder
    builder = collimator.DiagramBuilder()

    # blocks
    input_val = 2.0
    two = builder.add(Constant(input_val, name="two"))
    fmu_block = builder.add(
        ModelicaFMU(fmu_file, 1.0, name="fmu_block", start_time=start_time)
    )

    # connections
    builder.connect(two.output_ports[0], fmu_block.input_ports[0])

    # prep model and run simulation
    diagram = builder.build()
    context = diagram.create_context()

    recorded_signals = {
        "fmu_add": fmu_block.output_ports[0],
        "fmu_clock": fmu_block.output_ports[1],
        "in_mult_ramp": fmu_block.output_ports[2],
        "fmu_ramp": fmu_block.output_ports[3],
    }
    r = collimator.simulate(
        diagram, context, (start_time, 10.0), recorded_signals=recorded_signals
    )

    if show_plot:
        fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 6))
        ax1.plot(r.time, r.outputs["fmu_clock"], label="fmu_clock", marker="o")
        ax1.plot(r.time, r.outputs["fmu_add"], label="fmu_add", marker="<")
        ax1.plot(r.time, r.outputs["fmu_ramp"], label="fmu_ramp", marker="x")
        ax1.plot(r.time, r.outputs["in_mult_ramp"], label="in_mult_ramp")
        ax1.grid(True)
        ax1.legend()

        plt.show()

    # so behavior is here is not ideal, but its not far from it.
    # at least we can say we've done everything we can with fmupy
    # and OpenModelica FMU export to get start time to work.
    if start_time == 0.0:
        clk_ramp_sol = np.arange(2, 11, 1)
        print(f"clk_ramp_sol = {clk_ramp_sol}")
        assert np.allclose(r.outputs["fmu_clock"][2:], clk_ramp_sol)
        assert np.allclose(r.outputs["fmu_ramp"][2:], clk_ramp_sol)
        assert np.allclose(r.outputs["in_mult_ramp"][2:], clk_ramp_sol * input_val)
    elif start_time == 5.0:
        clk_ramp_sol = np.arange(6, 11, 1)
        print(f"clk_ramp_sol = {clk_ramp_sol}")
        assert np.allclose(r.outputs["fmu_clock"][1:], clk_ramp_sol)
        assert np.allclose(r.outputs["fmu_ramp"][1:], clk_ramp_sol)
        assert np.allclose(r.outputs["in_mult_ramp"][1:], clk_ramp_sol * input_val)


if __name__ == "__main__":
    # test_FMU_Thermal(show_plot=True)
    test_FMU_StartTime(show_plot=True, start_time=0.0)
    test_FMU_StartTime(show_plot=True, start_time=5.0)
