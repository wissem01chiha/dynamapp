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
"""
blocks in test_rotations.py are tested in:
test/app/CoordinateRotation/
test/app/CoordinateRotationConversion/
test/app/RigidBody/

FMU tested in:
test/app/ModelicaFMU/

SINDy tested in:
test/app/Sindy/

Predictor tested in:
test/app/test_predictor/

StateMachine tested in:
test/app/StateMachine/
"""


@pytest.mark.parametrize(
    "model_json",
    [
        "test_continuous.json",
        "test_custom.json",
        "test_discontinuities.json",
        "test_discrete.json",
        "test_logic.json",
        "test_lookup_tables.json",
        "test_math.json",
        "test_signal_routing.json",
        "test_source.json",
        "test_sink.json",
    ],
)
def test_json(request, model_json):
    test.run(pytest_request=request, check_only=True, model_json=model_json)
