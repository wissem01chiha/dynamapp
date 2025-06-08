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

"""
Tests for the Sindy block
"""

import pytest

import os

from types import SimpleNamespace

import numpy as np
import pysindy as ps

import collimator
from collimator.library import Sindy, Constant
from collimator.library.utils import read_csv, extract_columns


@pytest.fixture
def lorenz_csv():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_dir, "assets", "lorenz_sindy.csv")
    df = read_csv(file_path, header_as_first_row=True)
    x_train = extract_columns(df, "1:4")
    u_train = extract_columns(df, "4:6")
    t_train = extract_columns(df, -1)
    x_train = np.array(x_train)
    u_train = np.array(u_train)
    t_train = np.array(t_train)
    dt = 0.002
    return file_path, x_train, u_train, t_train, dt


@pytest.fixture
def synthetic_pendulum_csv():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_dir, "assets", "synthetic_pendulum.csv")
    df = read_csv(file_path, header_as_first_row=True)
    x_train = extract_columns(df, [6, -1])
    u_train = extract_columns(df, 1)
    t_train = extract_columns(df, 0)
    x_train = np.array(x_train)
    u_train = np.array(u_train)
    t_train = np.array(t_train)
    return file_path, x_train, u_train, t_train


def train_sindy(
    x_train,
    u_train=None,
    x_dot_train=None,
    time=None,
    discrete_time=False,
    poly_order=None,
    fourier_n_frequencies=None,
    threshold=0.1,
    alpha=0.05,
    max_iter=20,
    normalize_columns=False,
):
    """Train the SINDy model"""
    library = []

    if poly_order is not None:
        library.append(ps.PolynomialLibrary(degree=poly_order))

    if fourier_n_frequencies is not None:
        library.append(ps.FourierLibrary(n_frequencies=fourier_n_frequencies))

    optimizer = ps.STLSQ(
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
    )

    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=ps.feature_library.GeneralizedLibrary(library),
        discrete_time=discrete_time,
    )

    model.fit(x_train, u=u_train, x_dot=x_dot_train, t=time, quiet=True)

    return model


#################### Continious time  ####################


@pytest.mark.parametrize(
    "polyorder, fourier_n_frequencies, threshold, alpha, max_iter, normalize_columns, with_control, discrete_time",
    [
        # Continuous time with control
        (2, None, 0.1, 0.05, 20, False, True, False),  # basic
        (3, 2, 0.05, 0.01, 50, True, True, False),  #  Fourier features
        (5, None, 0.2, 0.1, 10, False, True, False),  # higher-order poly
        # Continuous time without control
        (2, None, 0.1, 0.05, 20, False, False, False),  # basic
        (3, 2, 0.05, 0.01, 50, True, False, False),  # Fourier features
        (5, None, 0.2, 0.1, 10, False, False, False),  # higher-order poly
        # Discrete time with control
        (2, None, 0.1, 0.05, 20, False, True, True),  # basic
        (3, 2, 0.05, 0.01, 50, True, True, True),  #  Fourier features
        (5, None, 0.2, 0.1, 10, False, True, True),  # higher-order poly
        # Discrete time without control
        (2, None, 0.1, 0.05, 20, False, False, True),  # basic
        (3, 2, 0.05, 0.01, 50, True, False, True),  #  Fourier features
        (5, None, 0.2, 0.1, 10, False, False, True),  # higher-order poly
    ],
)
def test_sindy_continuous_with_control_constant_dt(
    lorenz_csv,
    polyorder,
    fourier_n_frequencies,
    threshold,
    alpha,
    max_iter,
    normalize_columns,
    with_control,
    discrete_time,
):
    abs_file_path, x_train, u_train, t_train, dt = lorenz_csv

    if not with_control:
        u_train = None
        control_input_columns = None
    else:
        control_input_columns = [4, 5]

    model = train_sindy(
        x_train,
        u_train=u_train,
        time=dt,
        discrete_time=discrete_time,
        poly_order=polyorder,
        fourier_n_frequencies=fourier_n_frequencies,
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
    )

    sindy_block = Sindy(
        abs_file_path,
        header_as_first_row=True,
        state_columns=[1, 2, 3],
        control_input_columns=control_input_columns,
        dt=dt,
        discrete_time=discrete_time,
        poly_order=polyorder,
        fourier_n_frequencies=fourier_n_frequencies,
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
    )

    state = SimpleNamespace()
    for test_index in [0, 11, 55, -10, -1]:
        x_test = x_train[test_index, :]
        if with_control:
            u_test = u_train[test_index, :]

            x_and_u = np.concatenate([x_test, u_test])
            pysindy_output = model.predict(x_test.reshape((1, 3)), u=u_test)
            wildcat_output_direct = np.matmul(
                sindy_block.coefficients,
                np.atleast_1d(sindy_block.features_func(x_and_u)),
            )
            if discrete_time:
                state.discrete_state = x_test
                wildcat_output = sindy_block._discrete_update(0.0, state, u_test)
            else:
                state.continuous_state = x_test
                wildcat_output = sindy_block._ode(0.0, state, u_test)

        else:
            pysindy_output = model.predict(x_test.reshape((1, 3)))
            wildcat_output_direct = np.matmul(
                sindy_block.coefficients,
                np.atleast_1d(sindy_block.features_func(x_test)),
            )

            if discrete_time:
                state.discrete_state = x_test
                wildcat_output = sindy_block._discrete_update(0.0, state, ())
            else:
                state.continuous_state = x_test
                wildcat_output = sindy_block._ode(0.0, state, ())

        assert np.allclose(pysindy_output, wildcat_output_direct)
        assert np.allclose(wildcat_output_direct, wildcat_output)


@pytest.mark.parametrize(
    "polyorder, fourier_n_frequencies, threshold, alpha, max_iter, normalize_columns, with_control, discrete_time",
    [
        # Continuous time with control
        (2, None, 0.1, 0.05, 20, False, True, False),  # basic
        (3, 2, 0.05, 0.01, 50, True, True, False),  #  Fourier features
        (5, None, 0.2, 0.1, 10, False, True, False),  # higher-order poly
        # Continuous time without control
        (2, None, 0.1, 0.05, 20, False, False, False),  # basic
        (3, 2, 0.05, 0.01, 50, True, False, False),  # Fourier features
        (5, None, 0.2, 0.1, 10, False, False, False),  # higher-order poly
        # Discrete time with control
        (2, None, 0.1, 0.05, 20, False, True, True),  # basic
        (3, 2, 0.05, 0.01, 50, True, True, True),  #  Fourier features
        (5, None, 0.2, 0.1, 10, False, True, True),  # higher-order poly
        # Discrete time without control
        (2, None, 0.1, 0.05, 20, False, False, True),  # basic
        (3, 2, 0.05, 0.01, 50, True, False, True),  #  Fourier features
        (5, None, 0.2, 0.1, 10, False, False, True),  # higher-order poly
    ],
)
def test_sindy_continuous_with_control_tarray(
    lorenz_csv,
    polyorder,
    fourier_n_frequencies,
    threshold,
    alpha,
    max_iter,
    normalize_columns,
    with_control,
    discrete_time,
):
    abs_file_path, x_train, u_train, t_train, dt = lorenz_csv

    if not with_control:
        u_train = None
        control_input_columns = None
    else:
        control_input_columns = ["u_0", "u_1"]

    model = train_sindy(
        x_train,
        u_train=u_train,
        time=t_train,
        discrete_time=discrete_time,
        poly_order=polyorder,
        fourier_n_frequencies=fourier_n_frequencies,
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
    )

    sindy_block = Sindy(
        abs_file_path,
        header_as_first_row=True,
        state_columns=["x_0", "x_1", "x_2"],
        control_input_columns=control_input_columns,
        time_column="t",
        discrete_time=discrete_time,
        poly_order=polyorder,
        fourier_n_frequencies=fourier_n_frequencies,
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
    )

    state = SimpleNamespace()
    for test_index in [0, 11, 55, -10, -1]:
        x_test = x_train[test_index, :]

        if with_control:
            u_test = u_train[test_index, :]
            x_and_u = np.concatenate([x_test, u_test])
            pysindy_output = model.predict(x_test.reshape((1, 3)), u=u_test)
            wildcat_output_direct = np.matmul(
                sindy_block.coefficients,
                np.atleast_1d(sindy_block.features_func(x_and_u)),
            )
            if discrete_time:
                state.discrete_state = x_test
                wildcat_output = sindy_block._discrete_update(0.0, state, u_test)
            else:
                state.continuous_state = x_test
                wildcat_output = sindy_block._ode(0.0, state, u_test)

        else:
            pysindy_output = model.predict(x_test.reshape((1, 3)))
            wildcat_output_direct = np.matmul(
                sindy_block.coefficients,
                np.atleast_1d(sindy_block.features_func(x_test)),
            )
            if discrete_time:
                state.discrete_state = x_test
                wildcat_output = sindy_block._discrete_update(0.0, state, ())
            else:
                state.continuous_state = x_test
                wildcat_output = sindy_block._ode(0.0, state, ())

        assert np.allclose(pysindy_output, wildcat_output_direct)
        assert np.allclose(wildcat_output_direct, wildcat_output)


####################  Serialization  ####################


def test_sindy_serialization(lorenz_csv):
    abs_file_path, x_train, u_train, t_train, dt = lorenz_csv
    model = train_sindy(x_train, u_train=u_train, time=t_train, poly_order=2)

    sindy_block_for_serialization = Sindy(
        abs_file_path,
        header_as_first_row=True,
        state_columns=["x_0", "x_1", "x_2"],
        control_input_columns=["u_0", "u_1"],
        time_column="t",
        fourier_n_frequencies=None,
        threshold=0.1,
    )

    serialized_file_name = "sindy_lorenz.json"
    abs_serialized_file_path = os.path.abspath(serialized_file_name)

    sindy_block_for_serialization.serialize(abs_serialized_file_path)

    # Load the serialized model
    sindy_block = Sindy(
        None,
        pretrained=True,
        pretrained_file_path=abs_serialized_file_path,
    )

    os.remove(abs_serialized_file_path)

    state = SimpleNamespace()
    for test_index in [0, 11, 55, -10, -1]:
        x_test = x_train[test_index, :].reshape((1, 3))
        u_test = u_train[test_index, :].reshape((1, 2))

        x_and_u = np.hstack([x_test[0], u_test[0]])

        pysindy_output = model.predict(x_test, u=u_test)
        wildcat_output_direct = np.matmul(
            sindy_block.coefficients, np.atleast_1d(sindy_block.features_func(x_and_u))
        )
        state.continuous_state = x_test[0]
        wildcat_output_ode = sindy_block._ode(0.0, state, u_test[0])

        assert np.allclose(pysindy_output, wildcat_output_direct)
        assert np.allclose(wildcat_output_direct, wildcat_output_ode)


####################  Simulation  ####################


@pytest.mark.parametrize(
    "with_control, discrete_time, initial_state",
    [
        (True, True, None),
        (True, False, None),
        (False, True, None),
        (False, False, None),
        (True, True, np.array([0.1, 0.2])),
        (True, False, np.array([0.1, 0.2])),
        (False, True, np.array([0.1, 0.2])),
        (False, False, np.array([0.1, 0.2])),
    ],
)
def test_sindy_simulation(
    synthetic_pendulum_csv,
    with_control,
    discrete_time,
    initial_state,
):
    abs_file_path, x_train, u_train, t_train = synthetic_pendulum_csv

    polyorder = 2
    fourier_n_frequencies = 2
    threshold = 0.1
    alpha = 0.05
    max_iter = 20
    normalize_columns = False
    discrete_time_update_interval = 0.02

    if with_control:
        control_input_columns = 1
    else:
        u_train = None
        control_input_columns = None

    model = train_sindy(
        x_train,
        u_train=u_train,
        time=t_train,
        discrete_time=discrete_time,
        poly_order=polyorder,
        fourier_n_frequencies=fourier_n_frequencies,
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
    )

    sindy_block = Sindy(
        abs_file_path,
        header_as_first_row=True,
        state_columns=["x.out_0", "v.out_0"],
        control_input_columns=control_input_columns,
        time_column="time",
        discrete_time=discrete_time,
        poly_order=polyorder,
        fourier_n_frequencies=fourier_n_frequencies,
        threshold=threshold,
        discrete_time_update_interval=discrete_time_update_interval,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
        initial_state=initial_state,
    )

    compare_pysindy_and_wildcat = False

    if compare_pysindy_and_wildcat:
        # Compare Sindy and Collimator block outputs
        state = SimpleNamespace()
        for test_index in [0, 11, 55, -10, -1]:
            x_test = x_train[test_index, :]

            if with_control:
                u_test = np.array([u_train[test_index]])
                x_and_u = np.concatenate([x_test, u_test])
                pysindy_output = model.predict(x_test.reshape((1, 2)), u=u_test)
                wildcat_output_direct = np.matmul(
                    sindy_block.coefficients,
                    np.atleast_1d(sindy_block.features_func(x_and_u)),
                )
                if discrete_time:
                    state.discrete_state = x_test
                    wildcat_output = sindy_block._discrete_update(0.0, state, u_test)
                else:
                    state.continuous_state = x_test
                    wildcat_output = sindy_block._ode(0.0, state, u_test)

            else:
                pysindy_output = model.predict(x_test.reshape((1, 2)))
                wildcat_output_direct = np.matmul(
                    sindy_block.coefficients,
                    np.atleast_1d(sindy_block.features_func(x_test)),
                )
                if discrete_time:
                    state.discrete_state = x_test
                    wildcat_output = sindy_block._discrete_update(0.0, state, ())
                else:
                    state.continuous_state = x_test
                    wildcat_output = sindy_block._ode(0.0, state, ())

            assert np.allclose(pysindy_output, wildcat_output_direct)
            assert np.allclose(wildcat_output_direct, wildcat_output)

    # Check that simulation runs without errors

    builder = collimator.DiagramBuilder()
    builder.add(sindy_block)

    if with_control:
        control = builder.add(Constant(0.0, name="control"))
        builder.connect(control.output_ports[0], sindy_block.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    collimator.simulate(diagram, context, (0.0, 0.2))
