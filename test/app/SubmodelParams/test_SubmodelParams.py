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
from collimator.framework.error import BlockParameterError, BlockInitializationError

pytestmark = pytest.mark.app


def test_0080_sm_params(request):
    # not failing to load model and evaluate params is considered test success
    res = test.run(pytest_request=request, stop_time=0.1, model_json="model_0080.json")

    # define pass conditions based on the param value that the various gain should have
    assert res["sm_params_as_numbers.Outport_0"][0] == 2.0
    assert res["sm_params_pass_from_model.Outport_0"][0] == 2.0 * 4.0
    assert res["Group_0.Outport_0"][0] == 2.0 * 6.0
    assert res["Group_0.sm_params_pass_from_model.Outport_0"][0] == 2.0 * 4.0
    assert res["sm_nester.Outport_0"][0] == 2.0
    assert res["sm_nester.sm_params_from_nester.Outport_0"][0] == 2.0 * 8.0


def test_0080a_sm_params_simworkergo(request):
    res = test.run(pytest_request=request, stop_time=0.1, model_json="model_0080a.json")

    # define pass conditions based on the param value that the various gain should have
    assert res["sm_params_as_numbers.Outport_0"][0] == 3.0
    assert res["sm_params_pass_from_model.Outport_0"][0] == 3.0 * 4.0
    assert res["Group_0.Outport_0"][0] == 3.0 * 6.0
    assert res["Group_0.sm_params_pass_from_model.Outport_0"][0] == 3.0 * 4.0
    assert res["sm_nester.Outport_0"][0] == 3.0
    assert res["sm_nester.sm_params_from_nester.Outport_0"][0] == 3.0 * 8.0


def test_0081_sm_v2_clean_namespace_fail_load(request):
    with pytest.raises(BlockParameterError) as _:
        test.run(pytest_request=request, stop_time=0.1, model_json="model_0081.json")


def test_OrphanParam(request):
    # this test has a submodel with an orphan param, which is best explained by example:
    #   create submodel with params A,B, C
    #   instantiate it once, called first_instance, this creats an instance reference in
    #       the model.json in the data base, and has entries for the user modified values of A,B,C
    #   edit the source submodel to remove param A
    #   the model.json entry for first_instance will not be updated to have the A param removed,
    #       so when you run a simulation, wildcat preprocessing identifies these orphan params,
    #       and raises a warning.
    #   add another instance called second_instance, another entry in made in model.json, but this
    #       only has user values for params B and C.
    #
    # in that example, param A of first_instance is an orphan param. orphan params only occur
    # as a result of not cleansing the database of A at all instances when A is removed from
    # the source submodel.
    #
    # the desired behavior is to ignore orphan params when they are unused in the instance submodel.
    # the desired behavior is to raise error when orphan params are used by the instance submodel.

    with pytest.raises(BlockInitializationError) as _:
        test.run(
            pytest_request=request, stop_time=0.1, model_json="model_orphanParams.json"
        )
