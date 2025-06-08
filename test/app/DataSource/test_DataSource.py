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

import numpy as np
import pytest
import collimator.testing as test
from ...gen_datasource_csv import gen_files

pytestmark = pytest.mark.app

"""
desired tests
1.0] no header, no time col, choose data column by index=1, EoF=hold, interpolation=zoh
1.1] no header, no time col, choose data column by index=1, EoF=hold, interpolation=linear
1.2] no header, no time col, choose data column by index=1, EoF=zero, interpolation=linear
1.3] no header, no time col, choose data column by index=1, EoF=zero, interpolation=zoh

2.0] no header, has time data in col=0, choose data column by index=2, EoF=hold, interpolation=zoh
2.1] no header, has time data in col=0, choose data column by index=2, EoF=hold, interpolation=linear
2.2] no header, has time data in col=0, choose data column by index=2, EoF=zero, interpolation=linear
2.3] no header, has time data in col=0, choose data column by index=2, EoF=zero, interpolation=zoh

3.0] has header, no time col, choose data column by index=1, EoF=hold, interpolation=linear
3.1] has header, no time col, choose data column=1 by name=c1, EoF=hold, interpolation=linear

4.0] has header, has time data in col=0, choose data column by index=2, EoF=hold, interpolation=linear
4.1] has header, has time data in col=0, choose data column=1 by name=c1, EoF=hold, interpolation=linear

5.0] no header, has time data in col=1, choose data column by index=3, EoF=hold, interpolation=linear
5.1] has header, has time data in col=1, choose data column=3 by name=c2, EoF=hold, interpolation=linear

6.0] no header, has time data in col=0, data in col=1 that is same as time,  EoF=hold, interpolation=zoh
6.1] no header, has time data in col=0, data in col=1 that is same as time,  EoF=hold, interpolation=linear
6.2] no header, has time data in col=0, data in col=1 that is same as time,  EoF=zero, interpolation=linear
6.3] no header, has time data in col=0, data in col=1 that is same as time,  EoF=zero, interpolation=zoh

7.0] has header, has time data in col=0, data in col=2,     EoF=hold, interpolation=zoh
7.1] has header, has time data in col=0, data in col='c1',  EoF=hold, interpolation=zoh
7.2] has header, has time data in col=0, data in col=1:4,,  EoF=hold, interpolation=zoh

other tests that should fail
1] data clolumn index out of range
2] empty cells in the header
"""


def check_1p1(res, times, sw):
    # Check results for test 1.1
    time = np.array(res["time"])
    ds_t1_sol = np.zeros(np.shape(time))
    t1_st = 1.0
    for idx in range(len(time)):
        t = time[idx]
        row_index = int(np.floor(t / t1_st))
        if row_index >= len(sw):
            ds_t1_sol[idx] = sw[-1]
        else:
            ds_t1_sol[idx] = np.interp(t, times, sw)
    test.calc_err_and_test_pass_conditions(time, ds_t1_sol, res, "t1_1.out_0")


def check_6(res, times, times_t6):
    # Check results for test 6.0
    time = np.array(res["time"])
    t6_0_sol = np.zeros(np.shape(time))
    for idx in range(len(time)):
        t = time[idx]
        row_index = np.searchsorted(times, t, side="right") - 1
        if row_index >= len(times):
            t6_0_sol[idx] = times[-1]
        elif row_index < 0:
            t6_0_sol[idx] = times[0]
        else:
            t6_0_sol[idx] = times[row_index]
    test.calc_err_and_test_pass_conditions(time, t6_0_sol, res, "t6_0.out_0")

    # Check results for test 6.1
    t6_1_sol = np.zeros(np.shape(time))
    for idx in range(len(time)):
        t = time[idx]
        row_index = np.searchsorted(times, t, side="right") - 1
        if t >= times[-1]:
            t6_1_sol[idx] = times[-1]
        elif row_index < 0:
            t6_1_sol[idx] = times[0]
        else:
            t6_1_sol[idx] = np.interp(t, times, times)
    test.calc_err_and_test_pass_conditions(time, t6_1_sol, res, "t6_1.out_0")

    # Check results for test 6.2
    t6_2_sol = np.zeros(np.shape(time))
    for idx in range(len(time)):
        t = time[idx]
        row_index = np.searchsorted(times, t, side="right") - 1
        if t >= times_t6[-1]:
            if t <= times_t6[-1] + 1e-16:
                t6_2_sol[idx] = times_t6[-1]
            else:
                t6_2_sol[idx] = 0.0
        elif row_index < 0:
            t6_2_sol[idx] = times[0]
        else:
            t6_2_sol[idx] = np.interp(t, times, times)
    test.calc_err_and_test_pass_conditions(time, t6_2_sol, res, "t6_2.out_0")

    # Check results for test 6.3
    t6_3_sol = np.zeros(np.shape(time))
    for idx in range(len(time)):
        t = time[idx]
        row_index = np.searchsorted(times, t, side="right") - 1
        if t >= times_t6[-1]:
            if t <= times_t6[-1] + 1e-16:
                t6_3_sol[idx] = times_t6[-1]
            else:
                t6_3_sol[idx] = 0.0
        elif row_index < 0:
            t6_3_sol[idx] = times[0]
        else:
            t6_3_sol[idx] = times[row_index]
    test.calc_err_and_test_pass_conditions(time, t6_3_sol, res, "t6_3.out_0")


def check_7(res, times, sw, cw):
    # Check results for test 7.0, 7.1
    time = np.array(res["time"])
    t7_0_sol = np.zeros(np.shape(time))
    for idx in range(len(time)):
        t = time[idx]
        row_index = np.searchsorted(times, t, side="right") - 1
        if row_index >= len(times):
            t7_0_sol[idx] = sw[-1]
        elif row_index < 0:
            t7_0_sol[idx] = sw[0]
        else:
            t7_0_sol[idx] = sw[row_index]

    t7_1_sol = t7_0_sol
    test.calc_err_and_test_pass_conditions(time, t7_0_sol, res, "t7_0.out_0")
    test.calc_err_and_test_pass_conditions(time, t7_1_sol, res, "t7_1.out_0")

    # Check results for test 7.2
    t7_2_0_sol = np.ones(np.shape(time))
    test.calc_err_and_test_pass_conditions(time, t7_2_0_sol, res, "t7_2_.out_0")
    t7_2_1_sol = t7_0_sol
    test.calc_err_and_test_pass_conditions(time, t7_2_1_sol, res, "t7_2_.out_1")
    t7_2_2_sol = np.zeros(np.shape(time))
    for idx in range(len(time)):
        t = time[idx]
        row_index = np.searchsorted(times, t, side="right") - 1
        if row_index >= len(times):
            t7_2_2_sol[idx] = cw[-1]
        elif row_index < 0:
            t7_2_2_sol[idx] = cw[0]
        else:
            t7_2_2_sol[idx] = cw[row_index]
    test.calc_err_and_test_pass_conditions(time, t7_2_2_sol, res, "t7_2_.out_2")


def check_9(res):
    time = np.array(res["time"])
    t9_0_sol = np.ones(np.shape(time))
    test.calc_err_and_test_pass_conditions(time, t9_0_sol, res, "t9_0.out_0")


def test_DataSource(request):
    test_paths = test.get_paths(request)
    workdir = test_paths["workdir"]
    # Prepare the test data
    stop_time = 10

    # create data for test cases
    times, sw, times_t6, cw, _ = gen_files(workdir, stop_time)

    # copy non-terminated file since pandas cant generate this for us. must be created manually in text editor.
    test.copy_to_workdir(test_paths, "mydata_time_head_xy.csv")

    res = test.run(test_paths=test_paths, stop_time=stop_time + 2)

    # these 'sections' highlight where pass conditions are applied.
    # ==========================================test 1 ===================================
    check_1p1(res, times, sw)
    # ==========================================test 2 ===================================
    # ==========================================test 3 ===================================
    # ==========================================test 4 ===================================
    # ==========================================test 5 ===================================
    # ==========================================test 6 ===================================
    check_6(res, times, times_t6)
    # ==========================================test 7 ===================================
    check_7(res, times, sw, cw)
    # ==========================================test 8 ===================================
    # just the fact that the sim did not fail is a pass condition
    # ==========================================test 9 ===================================
    check_9(res)
