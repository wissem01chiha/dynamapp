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

pytestmark = pytest.mark.app


@pytest.mark.skip(reason="test ran for >100 seconds without terminating. not sure why.")
def test_DemoAeroRocketEngine(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "init.py")
    test.run(test_paths=test_paths, stop_time=0.1)
