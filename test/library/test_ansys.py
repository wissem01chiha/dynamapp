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

"""Test for Ansys integration
Contains tests for:
- PyTwin
"""

import os
import sys
import shutil

import pytest

import collimator
from collimator.library import PyTwin


@pytest.fixture
def twin_file():
    # Download the twin file
    from pytwin import download_file  # MacOS may have problems with this package

    this_file_path = os.path.abspath(__file__)
    twin_model_dir = os.path.join(os.path.dirname(this_file_path), "twin_files")
    twin_file = download_file(
        "ElectricRange_23R1_other.twin",
        "twin_files",
        force_download=False,
        destination=twin_model_dir,
    )

    yield twin_file

    # Teardown: Delete the directory after the test
    shutil.rmtree(twin_model_dir, ignore_errors=True)


@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
def test_pytwin(twin_file, plot=False):
    """Basic test to check if the Collimator model runs"""

    dt = 10.0
    tf = 100.0

    parameters = {
        "ElectricRange_powerLoad": 2000.0,
        "ElectricRange_vehicleMass": 2000.0,
    }

    diagram = PyTwin(twin_file, dt, parameters=parameters, name="pytwin")

    context = diagram.create_context()

    recorded_signals = {
        "pack_SoC": diagram.output_ports[3],
        "position": diagram.output_ports[4],
        "speed_m": diagram.output_ports[5],
        "speed_ref": diagram.output_ports[6],
    }

    sol = collimator.simulate(
        diagram, context, (0.0, tf), recorded_signals=recorded_signals
    )

    if plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[18, 7])
        ax1.plot(sol.time, sol.outputs["speed_m"], label="measured speed")
        ax1.plot(sol.time, sol.outputs["speed_ref"], label="reference speed")

        ax2.plot(
            sol.outputs["pack_SoC"], sol.outputs["position"], label="position vs SoC"
        )

        ax1.set_xlim([0, 32 * 60])
        ax2.set_xlim([0.1, 0.9])

        ax1.legend()
        ax2.legend()

        fig.tight_layout()
        plt.show()
