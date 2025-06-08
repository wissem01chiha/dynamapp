#!env pytest
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
import time

pytestmark = pytest.mark.app


@pytest.mark.timeout(30)
def test_DemoWindTurbine(request):
    test_paths = test.get_paths(request)
    tic = time.perf_counter()
    test.copy_to_workdir(test_paths, "full_load_windfield.csv")
    test.run(test_paths=test_paths, stop_time=1, check_only=True)
    toc = time.perf_counter()
    print(f"exe_time={toc - tic:0.4f}")
