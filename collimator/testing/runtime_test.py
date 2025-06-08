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

from contextlib import contextmanager
import os
import pathlib
import shutil
import numpy as np
import glob

import collimator


@contextmanager
def set_cwd(path):
    """Sets the cwd within the context

    Args:
        path (str): The path to the cwd

    Yields:
        None
    """

    origin = os.path.abspath(path)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def get_paths(pytest_request, testdir_=None, test_name_=None):
    # get the relative path to the test file, e.g. "app/Abs"
    if pytest_request is not None:
        testdir = pathlib.Path(pytest_request.path).parent
    elif testdir_ is not None:
        testdir = pathlib.Path(testdir_)
    else:
        raise ValueError("pytest_request and testdir_ cannot both be None")

    # @am. this part is embarassing
    wildcat_root = pathlib.Path(__file__).parent.parent.parent
    test_root = pathlib.Path(os.path.join(wildcat_root, "test"))
    relpath = testdir.relative_to(test_root)

    # sanitize names like 'some_test[param]'
    if pytest_request is not None:
        test_name = pytest_request.node.name
    elif test_name_ is not None:
        test_name = test_name_
    else:
        raise ValueError("pytest_request and test_name_ cannot both be None")

    name = test_name.replace("[", "_").replace("]", "")
    workdir = test_root / "workdir" / relpath / name

    # create the directories
    logsdir = os.path.join(workdir, "logs")
    npydir = os.path.join(workdir, "npy")
    if workdir.exists():
        shutil.rmtree(workdir)
    os.makedirs(workdir, mode=0o777, exist_ok=True)
    os.makedirs(logsdir, mode=0o777, exist_ok=True)
    os.makedirs(npydir, mode=0o777, exist_ok=True)

    return {
        "testdir": testdir,
        "workdir": workdir,
        "npydir": npydir,
        "logsdir": logsdir,
    }


def copy_to_workdir(test_paths, *globs, dirs_exist_ok=False):
    """Copy files from testdir to workdir

    e.g. `test.copy_to_workdir(test_paths, "*.csv", "data.npz")`
    """
    testdir = test_paths["testdir"]
    workdir = test_paths["workdir"]
    for g in globs:
        g = os.path.join(testdir, g)
        for src in glob.glob(g):
            dst = pathlib.Path(src).relative_to(testdir)
            dst = os.path.join(workdir, dst)
            # @am. this was in the original. not sure if still needed
            # if os.path.isdir(src):
            #     shutil.copytree(src, dst, dirs_exist_ok=dirs_exist_ok)
            # else:
            shutil.copyfile(src, dst)


def run(
    pytest_request=None,
    test_paths=None,
    model_json=None,
    stop_time=None,
    check_only=False,
):
    print("IN runtime_test run()")

    if test_paths is None:
        if pytest_request is None:
            raise ValueError("test_paths and pytest_request cannot both be None")
        test_paths = get_paths(pytest_request)
    else:
        # this is used when the test script create the test paths before calling
        # run(), so that the test script could place some files in the workdir
        pass

    testdir = test_paths["testdir"]
    workdir = test_paths["workdir"]
    npydir = test_paths["npydir"]
    logsdir = test_paths["logsdir"]

    if model_json is None:
        model_json = "model.json"

    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        model = collimator.load_model(
            modeldir=testdir, model=model_json, logsdir=logsdir, npydir=npydir
        )

        if check_only:
            # model.check() # Note: somehow, the load_model() call performs check()
            return

        results = model.simulate(stop_time=stop_time)
    finally:
        os.chdir(cwd)

    return results


def calc_err_and_test_pass_conditions(
    time,
    sol,
    res_cont,
    sim_sol_sig_name,
    name=None,
    rel_err_den_clip=1e-2,
    err_max_limit=1e-6,  # block_test_utils.py used 1e-3
    rel_err_max_limit=1e-3,  # block_test_utils.py used 1e-6
    err_type="max",
):
    """
    Assert that `res_cont[sim_sol_sig_name]` is equal to `sol` within some
    tolerance.

    Instead of this, use `numpy.testing.assert_allclose(res[column], expected)`.
    """
    sim_sol = np.squeeze(np.array(res_cont[sim_sol_sig_name]))
    if sol.dtype == "bool":
        err = sim_sol == sol
        assert np.all(err)
        return
    else:
        err = np.subtract(sim_sol, sol)
        print("\n")
        print(sim_sol_sig_name)
        print("sol=", sol)
        print("sim_sol=", sim_sol)
        print("err=", err)
        rel_err_den_sign = np.sign(sol)  # get the sign of the denominator
        # make the denominator sign has no zeros
        rel_err_den_sign = np.where(rel_err_den_sign == 0, 1, rel_err_den_sign)
        rel_err_den = np.multiply(
            np.clip(np.abs(sol), rel_err_den_clip, np.Infinity),
            rel_err_den_sign,
        )
        # print("rel_err_den=", rel_err_den)
        np.nan_to_num(rel_err_den, copy=False, nan=rel_err_den_clip)
        rel_err = np.divide(err, rel_err_den)
        # print("rel_err=", rel_err)
        # check pass conditions
        if err_type == "max":
            err_max = np.max(np.abs(err))
            rel_err_max = np.max(np.abs(rel_err))
        elif err_type == "mean":
            err_max = np.mean(np.abs(err))
            rel_err_max = np.mean(np.abs(rel_err))
        else:
            raise ValueError("unsupported input arg for err_type")

        if name is None:
            sig_name = sim_sol_sig_name
        else:
            sig_name = name
        if err_max > err_max_limit:
            print(sig_name)
            print(f"abs err NOT ok. err_max={err_max}, err_max_limit={err_max_limit}")
        if rel_err_max > rel_err_max_limit:
            print(sig_name)
            print(
                f"rel err not OK. rel_err_max={rel_err_max}, rel_err_max_limit={rel_err_max_limit}"
            )

        assert err_max <= err_max_limit
        assert rel_err_max <= rel_err_max_limit

        return


# For all i in len lhs, compare lhs[i] with rhs[i] where rhs[i]=f(inps[i])
# We could dispense with the loop here, construct the rhs as one large array
# and compare the arrays with np.testing.assert_equal. However that would
# necessitate realizing the entire array in memory first. This is impractical
# since each signal is potentially a very large memory-mapped stream.
# Is there a better name for this function?
def map_assert_equal(lhs, f, inps):
    for i in range(len(lhs.signal_data)):
        try:
            np.testing.assert_equal(
                lhs.signal_data[i], f([x.signal_data[i] for x in inps])
            )
        except AssertionError as ae:
            raise AssertionError(f"\nindex={i}" + ae.args[0])


# rtol and atol defaults are from numpy.testing.assert_allclose
def map_assert_allclose(lhs, f, inps, rtol=1e-07, atol=0):
    for i in range(len(lhs.signal_data)):
        try:
            np.testing.assert_allclose(
                lhs.signal_data[i],
                f([x.signal_data[i] for x in inps]),
                rtol,
                atol,
            )
        except AssertionError as ae:
            raise AssertionError(f"\nindex={i}" + ae.args[0])
