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
import sys
import collimator.testing as test

pytestmark = pytest.mark.app

"""
the intend here is just to ensure that wildcat can load/run an FMU
when this op is specified in the model.json
"""


@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
def test_ModelicaFMU(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "thermal_1.fmu")
    test.run(test_paths=test_paths, stop_time=0.1, check_only=True)


@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
def test_fmu_clock(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "fmu_clock.fmu")
    test.run(test_paths=test_paths, model_json="fmu_clock.json", check_only=True)
