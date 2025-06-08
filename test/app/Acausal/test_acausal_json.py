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
import collimator
import pathlib
import matplotlib.pyplot as plt
import collimator.testing as test
import numpy as np
import os
import json

pytestmark = pytest.mark.app


def _discrete_lpf(time, input, R=1.0, C=1.0):
    # naively use a discrete low pass filter as the 'expected solution' for models that are
    # electrical domain low pass filters.
    # https://en.wikipedia.org/wiki/Low-pass_filter#Difference_equation_through_discrete_time_sampling
    output = np.zeros_like(time)
    RC = R * C

    for idx in range(len(time)):
        if idx > 0:
            T = time[idx] - time[idx - 1]
            B = np.exp(-T / RC)
            output[idx] = output[idx - 1] * B + (1 - B) * input[idx]

    return output


def _compute_tols(signal1, signal2):
    absolute_differences = np.abs(signal1 - signal2)
    relative_differences = absolute_differences / np.abs(signal1)

    return np.amax(absolute_differences), np.amax(relative_differences)


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def test_AcausalSignalTypes(request):
    test_paths = test.get_paths(request)
    collimator.load_model(
        test_paths["testdir"],
        logsdir=test_paths["logsdir"],
        model="acausal_signal_types.json",
    )
    result_signal_types_path = os.path.join(test_paths["logsdir"], "signal_types.json")
    expect_signal_types_path = os.path.join(test_paths["testdir"], "signal_types.json")
    result_signal_types = load_json(result_signal_types_path)
    expected_signal_types = load_json(expect_signal_types_path)

    # print("expect_signal_types_path", expect_signal_types_path)
    # print("result_signal_types_path", result_signal_types_path)

    # for nde, ndr in zip(expected_signal_types["nodes"], result_signal_types["nodes"]):
    #     if nde != ndr:
    #         print(f"\n{nde=}")
    #         print(f"{ndr=}")
    #         print("not same")
    # assert result_signal_types == expected_signal_types
    node_diffs = []
    for nde in expected_signal_types["nodes"]:
        print(f"\n{nde=}")
        for ndr in result_signal_types["nodes"]:
            print(f"{ndr=}")
            if nde["namepath"] == ndr["namepath"]:
                if nde != ndr:
                    print("not same")
                    node_diffs.append((nde, ndr))
                break

    assert not node_diffs


def test_acausal_system_params():
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="acausal_system_params.json")
    # r = model.simulate(stop_time=10.0)

    # also enforce that the acausal_system context have all the component params, with the correct values.
    found_elec_acausal_system = False
    found_rot_acausal_system = False
    found_trans_acausal_system = False
    found_therm_acausal_system = False
    for idx, subctx in model.context.subcontexts.items():
        print(f"\n\n{subctx.owning_system.name=}")
        if subctx.owning_system.name == "root.electrical.acausal_system":
            found_elec_acausal_system = True
            print(f"\tparams {subctx.parameters.keys()}")
            assert subctx.parameters["Resistor_0_R"] == 1.0
            assert subctx.parameters["Capacitor_0_C"] == 1.0
            assert subctx.parameters["VoltageSource_0_v"] == 1.0
            # would be nice to check that capacitor initial_voltage is also correct
        elif subctx.owning_system.name == "root.rotational.acausal_system":
            found_rot_acausal_system = True
            print(f"\tparams {subctx.parameters.keys()}")
            # assert subctx.parameters["ConstantTorque_0_T"] == 20.0 # FIXME: doesn't appear in acausal_system params
            assert subctx.parameters["Spring_0_K"] == 1.0
            assert subctx.parameters["Damper_0_D"] == 1.0
            assert subctx.parameters["Inertia_0_I"] == 1.0
            # would be nice to check that spring, damper, inertia initial values are also correct
        elif subctx.owning_system.name == "root.translational.acausal_system":
            found_trans_acausal_system = True
            print(f"\tparams {subctx.parameters.keys()}")
            # assert subctx.parameters["ConstantForce_0_T"] == 40.0 # FIXME: doesn't appear in acausal_system params
            assert subctx.parameters["Spring_0_K"] == 1.0
            assert subctx.parameters["Damper_0_D"] == 1.0
            assert subctx.parameters["Mass_0_M"] == 1.0
            # would be nice to check that spring, damper, mass initial values are also correct
        elif subctx.owning_system.name == "root.thermal.acausal_system":
            found_therm_acausal_system = True
            print(f"\tparams {subctx.parameters.keys()}")
            assert subctx.parameters["ThermalResistor_0_R"] == 1.0
            assert subctx.parameters["HeatCapacitor_0_C"] == 1.0
            # would be nice to check that spring, damper, mass initial values are also correct

    assert found_elec_acausal_system
    assert found_rot_acausal_system
    assert found_trans_acausal_system
    assert found_therm_acausal_system


def test_lowpass_filter(show_plot=False):
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="lowpass_filter.json")
    r = model.simulate(stop_time=10.0)

    # print(r)
    time = r["time"]
    sw = r["SineWave_0.out_0"]
    sensV = r["VoltageSensor_0.v"]

    sensV_sol = _discrete_lpf(time, sw)

    rel_err = (sensV_sol - sensV) / np.abs(sensV)
    err_cmp_idx = np.argmin(np.abs(time - 2.0))
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.plot(time, sw, label="sw")
        ax1.plot(time, sensV, label="sensV")
        ax1.plot(time, sensV_sol, label="sensV_sol")
        ax1.legend()

        ax2.plot(time[err_cmp_idx:], rel_err[err_cmp_idx:], label="rel_err")
        ax2.legend()
        plt.show()

    assert np.allclose(
        sensV[err_cmp_idx:], sensV_sol[err_cmp_idx:], atol=0.0, rtol=0.05
    )


def test_mimo(show_plot=False):
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="mimo.json")
    r = model.simulate(stop_time=10.0)

    # print(r)
    time = r["time"]
    sw = r["SineWave_0.out_0"]

    # to verify that multiple outputs works, it's enough to
    # just make sure these are in the results.
    sensVc = r["VoltageSensor_0.v"]
    sensI = r["CurrentSensor_0.i"]

    if show_plot:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 3))
        ax1.plot(time, sw, label="sw")
        ax1.plot(time, sensVc, label="sensVc")
        ax1.plot(time, sensI, label="sensI")
        ax1.legend()
        plt.show()

    # this is a hacky way to verify that both the inputs worked.
    assert np.max(sensVc) > 1.5  # makes sure the const input worked


def test_ideal_motor(show_plot=False):
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="ideal_motor.json")
    r = model.simulate(stop_time=10.0)

    print(r.keys())
    time = r["time"]
    curr = r["CurrentSensor_1.i"]
    # FIXME: add these sensors in to this test once uniport sensors work
    # spd = r["SpeedSensor_0.w_rel"]
    # temp = r["TemperatureSensor_0.T_rel"]

    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 3))
        ax1.plot(time, curr, label="sensI")
        ax1.legend()

        # ax2.plot(time, spd, label="spd")
        # ax2.legend()

        # ax3.plot(time, temp, label="temp")
        # ax3.legend()
        plt.show()

    # FIXME: add some pass conditions


def test_uniport_sensors(show_plot=False):
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="uniport_sensors.json")
    r = model.simulate(stop_time=10.0)

    print(r.keys())
    time = r["time"]
    spd1_2port = r["SpeedSensor_1.w_rel"]
    spd0_1port = r["SpeedSensor_0.w_rel"]
    spd3_1port = r["SpeedSensor_3.w_rel"]

    # same as spd1_2port, but connected the othe way around so the
    # sign of its output is inverted.
    spd2_2port = r["SpeedSensor_2.w_rel"]

    if show_plot:
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 3))
        ax1.plot(time, spd1_2port, label="spd1_2port")
        ax1.plot(time, spd2_2port, label="spd2_2port")
        ax1.plot(time, spd0_1port, label="spd0_1port")
        ax1.plot(time, spd3_1port, label="spd3_1port")
        ax1.legend()
        plt.show()

    assert np.allclose(spd0_1port, spd1_2port)
    assert np.allclose(spd0_1port, spd3_1port)

    # see note above for justification of '-' sign
    assert np.allclose(spd0_1port, -spd2_2port)


def test_acausal_system_in_submodel(show_plot=False):
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="acausal_system_in_submodel.json")
    r = model.simulate(stop_time=1.0)

    # print(r.keys())
    time = r["time"]
    g_AcausalSm = r["g_AcausalSm.out_0"]
    g_Group_0 = r["g_Group_0.out_0"]
    g_AcausalSm_with_more_0 = r["g_AcausalSm_with_more_0.out_0"]
    g_AcausalSm_with_more_1 = r["g_AcausalSm_with_more_1.out_0"]

    sw = r["SineWave_0.out_0"]

    g_AcausalSm_sol = _discrete_lpf(time, sw)
    g_Group_0_sol = _discrete_lpf(time, sw + 1)
    g_AcausalSm_with_more_0_sol = g_AcausalSm_sol
    g_AcausalSm_with_more_1_sol = _discrete_lpf(time, sw + 2)

    if show_plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 3))
        ax1.plot(time, g_AcausalSm, label="g_AcausalSm", marker="o")
        ax1.plot(time, g_AcausalSm_sol, label="g_AcausalSm_sol")
        ax1.grid()
        ax1.legend()

        ax2.plot(time, g_Group_0, label="g_Group_0", marker="o")
        ax2.plot(time, g_Group_0_sol, label="g_Group_0_sol")
        ax2.grid()
        ax2.legend()

        ax3.plot(
            time, g_AcausalSm_with_more_0, label="g_AcausalSm_with_more_0", marker="o"
        )
        ax3.plot(time, g_AcausalSm_with_more_0_sol, label="g_AcausalSm_with_more_0_sol")
        ax3.grid()
        ax3.legend()

        ax4.plot(
            time, g_AcausalSm_with_more_1, label="g_AcausalSm_with_more_1", marker="o"
        )
        ax4.plot(time, g_AcausalSm_with_more_1_sol, label="g_AcausalSm_with_more_1_sol")
        ax4.grid()
        ax4.legend()
        plt.show()

    # a0, r0 = _compute_tols(g_AcausalSm, g_AcausalSm_sol)
    # print(f"{a0=} {r0=}")
    # a0, r0 = _compute_tols(g_Group_0, g_Group_0_sol)
    # print(f"{a0=} {r0=}")
    # a0, r0 = _compute_tols(g_AcausalSm_with_more_0, g_AcausalSm_with_more_0_sol)
    # print(f"{a0=} {r0=}")
    # a0, r0 = _compute_tols(g_AcausalSm_with_more_1, g_AcausalSm_with_more_1_sol)
    # print(f"{a0=} {r0=}")

    atol = 0.02
    rtol = 0.0
    assert np.allclose(g_AcausalSm, g_AcausalSm_sol, atol=atol, rtol=rtol)
    assert np.allclose(g_Group_0, g_Group_0_sol, atol=atol, rtol=rtol)
    assert np.allclose(
        g_AcausalSm_with_more_0,
        g_AcausalSm_with_more_0_sol,
        atol=atol,
        rtol=rtol,
    )
    assert np.allclose(
        g_AcausalSm_with_more_1,
        g_AcausalSm_with_more_1_sol,
        atol=atol,
        rtol=rtol,
    )


def test_acausal_model_params(show_plot=False):
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="model_params.json")
    r = model.simulate(stop_time=10.0)

    time = r["time"]
    # print(r)
    top_level = r["SpeedSensor_0.x_rel"]
    sm_default = r["sm_default.SpeedSensor_1.x_rel"]
    sm_model_param = r["sm_model_param.SpeedSensor_1.x_rel"]

    top_level_sol = np.cos(time)
    sm_default_sol = np.cos(time / np.sqrt(2))
    sm_model_param_sol = np.cos(time * np.sqrt(2))

    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 3))
        ax1.plot(time, top_level, label="top_level")
        ax1.plot(time, top_level_sol, label="top_level_sol")
        # ax1.plot(time, top_level_sol - top_level, label="err")
        ax1.legend()
        ax1.grid()

        ax2.plot(time, sm_default, label="sm_default")
        ax2.plot(time, sm_default_sol, label="sm_default_sol")
        # ax2.plot(time, sm_default_sol - sm_default, label="err")
        ax2.legend()
        ax2.grid()

        ax3.plot(time, sm_model_param, label="sm_model_param")
        ax3.plot(time, sm_model_param_sol, label="sm_model_param_sol")
        # ax3.plot(time, sm_model_param_sol - sm_model_param, label="err")
        ax3.legend()
        ax3.grid()
        plt.show()

    assert np.allclose(top_level, top_level_sol, rtol=0.0, atol=1e-2)
    assert np.allclose(sm_default, sm_default_sol, rtol=0.0, atol=1e-2)
    assert np.allclose(sm_model_param, sm_model_param_sol, rtol=0.0, atol=2e-2)


def test_acausal_battery(show_plot=False):
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="battery.json")
    r = model.simulate(stop_time=1.0)
    time = r["time"]
    soc = r["Battery_0.soc"]
    i = r["CurrentSensor_0.i"]

    if show_plot:
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 3))
        ax1.plot(time, soc, label="soc")
        ax1.plot(time, i, label="i")
        ax1.legend()
        plt.show()


def test_wc415():
    # just the fact that they dont raise any errors is a pass :)
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="wc415_ok.json")
    model.simulate(stop_time=10.0)

    model = collimator.load_model(testdir, model="wc415_error_in_in.json")
    model.simulate(stop_time=1.0)


def test_bldc():
    # just the fact that they dont raise any errors is a pass :)
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="bldc.json")
    model.simulate(stop_time=1.0)


def test_friction():
    # just the fact that they dont raise any errors is a pass :)
    testdir = pathlib.Path(__file__).parent
    model = collimator.load_model(testdir, model="friction.json")
    model.simulate(stop_time=1.0)


if __name__ == "__main__":
    # test_acausal_system_params()
    # test_lowpass_filter(show_plot=True)
    # test_mimo(show_plot=True)
    # test_ideal_motor(show_plot=True)
    # test_uniport_sensors(show_plot=True)
    # test_acausal_system_in_submodel(show_plot=True)
    # test_acausal_model_params(show_plot=True)
    # test_acausal_battery(show_plot=True)
    # test_wc415()
    # test_bldc()
    test_friction()
