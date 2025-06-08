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
import json
import os
import collimator
from collimator.framework.error import ShapeMismatchError

pytestmark = pytest.mark.app


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def test_0039_signal_types(request):
    test_paths = test.get_paths(request)
    collimator.load_model(
        test_paths["testdir"],
        model="model_signal_types.json",
        logsdir=test_paths["logsdir"],
    )
    # model.check() runs in __init__

    sol_path = os.path.join(test_paths["testdir"], "signal_types_expected.json")
    res_path = os.path.join(test_paths["logsdir"], "signal_types.json")

    sol_data = load_json(sol_path)
    res_data = load_json(res_path)

    assert sol_data == res_data


def test_time_mode(request):
    test_paths = test.get_paths(request)
    collimator.load_model(
        test_paths["testdir"],
        model="model_time_mode.json",
        logsdir=test_paths["logsdir"],
    )

    # get the time_mode assignment from the log json
    tma_path = os.path.join(test_paths["logsdir"], "signal_types.json")
    tma = load_json(tma_path)
    # print(tma)

    tma_sol = {
        "dd_max": "Constant",
        "dd_cmp": "Discrete",
        "dd_Inv": "Constant",
        "dd_int": "Discrete",
        "dd_one": "Constant",
        "dd_if": "Discrete",
        "dfc_clk": "Continuous",
        "dfc_gain": "Continuous",
        "dfc_add": "Continuous",
        "dfc_pow": "Continuous",
        "cc_cos": "Continuous",
        "cc_int": "Continuous",
        "cc_cmp": "Continuous",
        "d3d_cos": "Continuous",
        "d3d_int": "Discrete",
        "d3d_cmp": "Discrete",  # "Continuous", cant control source block time mode
        "h5c_cos": "Continuous",
        "h5d_int": "Discrete",
        "h5c_int": "Continuous",
        "h5c_cmp": "Continuous",
        "hd_one": "Constant",
        "hd_add": "Discrete",
        "hd_ud": "Discrete",
        "hc_sin": "Continuous",
        "hc_mult": "Continuous",
        "hc_int": "Continuous",
        "hc_gain": "Continuous",
        "h2c_mult2": "Continuous",
        "h2c_int1": "Continuous",
        "h2c_exp": "Continuous",
        "h2c_mult1": "Continuous",
        "h2c_int2": "Continuous",
        "h2c_recp": "Continuous",
        "h2c_gain1": "Continuous",
        "h2c_lut": "Continuous",
        "h2c_trig": "Continuous",
        "h2c_sqrt": "Continuous",
        "h2c_add1": "Continuous",
        "h2c_log": "Continuous",
        "h2c_minmax": "Continuous",
        "h2d_zoh": "Discrete",
        "h2d_gain2": "Continuous",  # "Discrete", ?
        "h2d_abs": "Continuous",  # "Discrete", ?
        "h2c_add2": "Continuous",
        "h2c_const": "Constant",
        "h2c_cpm": "Continuous",
        "h2c_gain3": "Continuous",
        "h2c_mult3": "Continuous",
        "h2d_if": "Continuous",
        "h2d_int3": "Discrete",
        "d2d_one": "Constant",
        "d2d_add": "Discrete",
        "d2d_ud": "Discrete",
        "d2d_gain": "Discrete",
        "d2d_recp": "Discrete",
        "d2d_log": "Discrete",
        "d2d_sqrt": "Discrete",
        "d2d_add1": "Discrete",
        "d2d_gain1": "Discrete",
        "d2d_log1": "Discrete",
        "d2d_sqrt1": "Discrete",
        "h3c_int": "Continuous",
        "h3c_add": "Continuous",
        "h3d_ud": "Discrete",
        "h4c_int": "Continuous",
        "h4d_zoh": "Discrete",
        "h4d_add": "Continuous",  # "Discrete", dependency tracking of states is not intercepted and stopped at ZOH block.
        "h4d_ud": "Discrete",
        "h6c_step": "Continuous",
        "h6c_g_step": "Continuous",
        "h6c_abs_step": "Continuous",
        "h6c_int": "Continuous",
        "h6c_g_int": "Continuous",
        "h6c_abs_int": "Continuous",
        "h6d_ud": "Discrete",
        "h6d_g_ud": "Discrete",
        "h6d_abs_ud": "Discrete",
        "h7c_g_step": "Continuous",
        "h7c_abs_step": "Continuous",
        "h7c_int": "Continuous",
        "h7c_g_int": "Continuous",
        "h7c_abs_int": "Continuous",
        "h7d_ud": "Discrete",
        "h7d_g_ud": "Discrete",
        "h7d_abs_ud": "Discrete",
        "h8d_g_step": "Discrete",
        "h8d_abs_step": "Discrete",
        "h8c_int": "Continuous",
        "h8c_g_int": "Continuous",
        "h8c_abs_int": "Continuous",
        "h8d_ud": "Discrete",
        "h8d_g_ud": "Discrete",
        "h8d_abs_ud": "Discrete",
        "d4d_step": "Continuous",
        "d4d_g_step": "Continuous",
        "d4d_abs_step": "Continuous",
        "d4d_ud1": "Discrete",
        "d4d_g_ud1": "Discrete",
        "d4d_abs_ud1": "Discrete",
        "d4d_ud": "Discrete",
        "d4d_g_ud": "Discrete",
        "d4d_abs_ud": "Discrete",
        "h9c_cos": "Continuous",
        "h9c_g_cos": "Continuous",
        "h9c_abs_cos": "Continuous",
        "h9c_int": "Continuous",
        "h9c_g_int": "Continuous",
        "h9d_ud": "Discrete",
        "h9d_g_ud": "Discrete",
        "h9d_abs_ud": "Discrete",
        "d5c_cos": "Continuous",
        "d5c_g_cos": "Continuous",
        "d5c_abs_cos": "Continuous",
        "d5d_ud1": "Discrete",
        "d5d_g_ud1": "Discrete",
        "d5d_abs_ud1": "Discrete",
        "d5d_ud": "Discrete",
        "d5d_g_ud": "Discrete",
        "d5d_abs_ud": "Discrete",
        "h11c_cos": "Continuous",
        "h11c_cmp": "Discrete",  # "Continuous", cant control source block time mode
        "h11c_gain": "Continuous",
        "h11d_int": "Discrete",
        "h11d_one": "Constant",
        "h11c_if": "Discrete",  # "Continuous", cant control source block time mode
        "d6d_one": "Constant",
        "d6d_int": "Discrete",
        "d6d_cmp": "Discrete",
    }
    all_match = True
    for node in tma["nodes"]:
        blk_name = node["namepath"][0]
        tm = node["time_mode"]
        tm_sol = tma_sol.get(blk_name, None)
        # print(f'blk_nam_sol={blk_nam_sol}, tm_sol={tm_sol}, tm={tm}.')
        if tm_sol is not None:
            if tm_sol != tm:
                print(f"blk_nam_sol={blk_name}, tm_sol={tm_sol}, tm={tm}.")
                all_match = False
            # else:
            #     print(f'blk_nam_sol={blk_nam_sol} is OK')
    assert all_match
    # assert False


def test_sm_time_mode(request):
    test_paths = test.get_paths(request)
    collimator.load_model(
        test_paths["testdir"], model="sm_model.json", logsdir=test_paths["logsdir"]
    )

    # get the time_mode assignment from the log json
    tma_path = os.path.join(test_paths["logsdir"], "signal_types.json")
    tma = load_json(tma_path)
    # print(tma)

    tma_sol = {
        "<root>": "Hybrid",
        "sm_const": "Constant",
        "sm_d": "Discrete",
        "sm_cont": "Continuous",
        "sm_h": "Hybrid",
        "sm_h_nested.sm_nested_d": "Discrete",
        "sm_h_nested.sm_nested_c": "Continuous",
        "sm_h_nested": "Hybrid",
        # check for ports time_mode also.
        "sm_d_ports.in0_d": "Discrete",
        "sm_d_ports.in1_d": "Discrete",
        "sm_d_ports.out1_sm_port_d_c": "Continuous",
        "sm_d_ports.out1_sm_port_d_d": "Discrete",
        "sm_c_ports.in0_c": "Continuous",
        "sm_c_ports.in1_c": "Continuous",
        "sm_c_ports.out1_sm_port_c_c": "Continuous",
        "sm_c_ports.out0_sm_port_c_c": "Continuous",
        "grp_const": "Constant",
        "grp_d": "Discrete",
        "grp_cont": "Continuous",
        "grp_h": "Hybrid",
        "rep_d": "Discrete",
        "rep_cont": "Continuous",
        "rep_h": "Hybrid",
        "cd_d": "Discrete",
        "cd_cont": "Continuous",
        "cd_h": "Hybrid",
    }
    all_match = True
    for node in tma["nodes"]:
        namepath = node["namepath"]
        blk_name = ".".join(namepath)
        tm = node["time_mode"]
        tm_sol = tma_sol.get(blk_name, None)
        if tm_sol is not None:
            if tm_sol != tm:
                print(f"blk_nam_sol={blk_name}, tm_sol={tm_sol}, tm={tm}.")
                all_match = False
            # else:
            #     print(f"blk_nam_sol={blk_name} is OK")
    assert all_match, "Some blocks time_mode are not as expected."


def test_time_mode_user_def(request):
    test_paths = test.get_paths(request)
    collimator.load_model(
        test_paths["testdir"],
        model="model_time_mode_user_def.json",
        logsdir=test_paths["logsdir"],
    )

    # get the time_mode assignment from the log json
    tma_path = os.path.join(test_paths["logsdir"], "signal_types.json")
    tma = load_json(tma_path)
    # print(tma)

    tma_sol = {
        "psb_d": "Discrete",
        "psb_c": "Continuous",
        "smach_d": "Discrete",
        "smach_c": "Continuous",
    }
    all_match = True
    for node in tma["nodes"]:
        blk_name = node["namepath"][0]
        tm = node["time_mode"]
        tm_sol = tma_sol.get(blk_name, None)
        # print(f'blk_nam_sol={blk_nam_sol}, tm_sol={tm_sol}, tm={tm}.')
        if tm_sol is not None:
            if tm_sol != tm:
                print(f"blk_nam_sol={blk_name}, tm_sol={tm_sol}, tm={tm}.")
                all_match = False
            # else:
            #     print(f'blk_nam_sol={blk_nam_sol} is OK')
    assert all_match
    # assert False


def test_signal_type_error(request):
    test_paths = test.get_paths(request)
    with pytest.raises(ShapeMismatchError) as exc:
        collimator.load_model(
            test_paths["testdir"],
            model="signal_type_error.json",
            logsdir=test_paths["logsdir"],
        )

    # Even though the previous step raised an Error, we still
    # expect the process to have generated signal_types.json,
    # and that this json contains correct signa type data.
    sol_path = os.path.join(
        test_paths["testdir"], "signal_types_with_error_expected.json"
    )
    res_path = os.path.join(test_paths["logsdir"], "signal_types.json")

    sol_data = load_json(sol_path)
    res_data = load_json(res_path)

    assert sol_data == res_data

    # FIXME. for some reason, the 'exc' variable doesn't appear to be the same
    # error object that was raised in model_interface:AppInterface.
    # Not sure how to fix this, it would be better if it were then we could have
    # a pass condition here requiring the error object to be specific.
    print("pytest exc print")
    print(exc)
