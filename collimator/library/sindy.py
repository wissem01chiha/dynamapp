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
Wrappers for PySINDy
"""

import warnings
import numpy as np
import json
from typing import TYPE_CHECKING

from ..backend import numpy_api as cnp
from ..logging import logdata, logger

from collimator.framework import LeafSystem, parameters
from collimator.lazy_loader import LazyLoader

from .utils import read_csv, extract_columns

ps = LazyLoader("ps", globals(), "pysindy")
if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")


__all__ = [
    "Sindy",
]


def _reduce(base_feature_names, feature_names, coefficients):
    """
    Reduce the sparse SINDy coefficients
    by removing columns with all zeros
    """
    nx, _ = coefficients.shape
    nxu = len(base_feature_names)

    # Identify zero columns
    zero_columns = cnp.all(coefficients == 0, axis=0)

    # Filter out zero columns from coefficients
    coefficients_filtered = coefficients[:, ~zero_columns]

    # Filter out corresponding feature names
    feature_names_filtered = [
        name for name, is_zero in zip(feature_names, zero_columns) if not is_zero
    ]

    # Handle the case where all coefficients are zero
    if len(feature_names_filtered) == 0:
        feature_names_filtered = base_feature_names
        coefficients_filtered = cnp.zeros((nx, nxu))

    return feature_names_filtered, coefficients_filtered


def train(
    x_train,
    u_train=None,
    x_dot_train=None,
    time=None,
    discrete_time=False,
    differentiation_method=None,
    poly_order=None,
    fourier_n_frequencies=None,
    custom_basis_functions=None,
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

    if custom_basis_functions is not None:
        library.append(ps.CustomLibrary(library_functions=custom_basis_functions))

    optimizer = ps.STLSQ(
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
    )

    if differentiation_method is None:
        differentiation_method = "centered difference"

    ## TODO: Add support for other differentiation methods. Accept strings as input
    ## and convert them to appropriate differentiation objects.
    differentiation_objects_map = {
        "centered difference": ps.differentiation.FiniteDifference(order=2, axis=0),
    }

    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=ps.feature_library.GeneralizedLibrary(library),
        differentiation_method=differentiation_objects_map[differentiation_method],
        discrete_time=discrete_time,
    )

    # Convert cnp arrays to numpy arrays explicitly for Sindy fit
    x_train = np.array(x_train)

    if u_train is not None:
        u_train = np.array(u_train)

    if x_dot_train is not None:
        x_dot_train = np.array(x_dot_train)

    time = np.array(time) if isinstance(time, cnp.ndarray) else time

    model.fit(x_train, u=u_train, x_dot=x_dot_train, t=time, quiet=True)

    base_feature_names = model.feature_names

    feature_names, coefficients = _reduce(
        base_feature_names, model.get_feature_names(), model.coefficients()
    )

    equations = model.equations()

    return equations, base_feature_names, feature_names, cnp.array(coefficients)


def train_from_csv(
    file_name,
    header_as_first_row=False,
    state_columns=1,
    control_input_columns=None,
    dt=None,
    time_column=None,
    state_derivatives_columns=None,
    discrete_time=False,
    differentiation_method="centered difference",
    # optimizer parameters
    threshold=0.1,
    alpha=0.05,
    max_iter=20,
    normalize_columns=False,
    # Library parameters
    poly_order=2,
    fourier_n_frequencies=None,
    custom_basis_functions=None,
):
    df = read_csv(file_name, header_as_first_row=header_as_first_row)

    if time_column is not None:
        time_column_name = (
            time_column if isinstance(time_column, str) else df.columns[time_column]
        )
        df = df.drop_duplicates(subset=[time_column_name], keep="first")
        time = extract_columns(df, time_column)
    elif dt:
        time = dt
    else:
        time = None

    has_control_input = control_input_columns is not None

    u_train = extract_columns(df, control_input_columns) if has_control_input else None

    x_dot_train = (
        extract_columns(df, state_derivatives_columns)
        if state_derivatives_columns
        else None
    )

    x_train = extract_columns(df, state_columns)

    (
        equations,
        base_feature_names,
        feature_names,
        coefficients,
    ) = train(
        x_train,
        u_train,
        x_dot_train,
        time,
        discrete_time=discrete_time,
        differentiation_method=differentiation_method,
        poly_order=poly_order,
        fourier_n_frequencies=fourier_n_frequencies,
        custom_basis_functions=custom_basis_functions,
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=normalize_columns,
    )

    return equations, base_feature_names, feature_names, coefficients, has_control_input


def _validate_leafsystem_inputs(
    pretrained, pretrained_file_path, dt, time_column, poly_order, fourier_n_frequencies
):
    if (pretrained is True) and (pretrained_file_path is None):
        raise ValueError(
            "Please provide `pretrained_file_path` as boolean "
            "`pretrained` is set to True"
        )

    if (pretrained is False) and (pretrained_file_path is not None):
        raise ValueError(
            "Boolean `pretrained` is False but `pretrained_file_path` " "is provided."
        )

    if pretrained is False:
        if (dt is None) and (time_column is None):
            warnings.warn(
                "Neither fixed dt nor column for time data are provided. "
                "Default dt=1.0 will be used for SINDy training"
            )

    if (poly_order is None) and (fourier_n_frequencies is None):
        raise ValueError(
            "Please use atleast one of the Polynomial or Fourier libraries "
            "by setting `poly_order` and/or `fourier_n_frequencies` "
        )


def _validate_ui_pretrained_data(coefficients, feature_names, base_feature_names, name):
    ui_pretrained_data = [coefficients, feature_names, base_feature_names]
    all_ui_pretrained_data_is_none = all(v is None for v in ui_pretrained_data)
    all_ui_pretrained_data_is_not_none = all(v is not None for v in ui_pretrained_data)

    if not (all_ui_pretrained_data_is_none or all_ui_pretrained_data_is_not_none):
        raise ValueError(
            f"Only some pretrained data from UI was provided for SINDY block: {name}. "
            f"{ui_pretrained_data}"
        )
    return all_ui_pretrained_data_is_not_none


class Sindy(LeafSystem):
    """
    This class implements System Identification (SINDy) algorithm with or without
    control inputs for contiuous-time and discrete-time systems.

    The learned continuous-time dynamical system model will be of the form:

    ```
    dx/dt = f(x, u)
    ```
    where `x` is the state vector and `u` is the optional control input vector. The
    block will output the full state vector `x` of the system.

    The learned discrete-time dynamical system model will be of the form:

    ```
    x_{k+1} = f(x_k, u_k)
    ```

    where `x_k` is the state vector at time step `k` and `u_k` is the optional control
    vector. The block will update the output to `x_k` at an interval provided by the
    parameter `discrete_time_update_interval`.

    Input ports:
        (0) u: control vector for the system. This port is only available if the Sindy
            model is trained with control inputs, i.e. `control_input_columns` is not
            `None` during training.

    Output ports:
        (0) x: full state output of the system.

    Parameters:
        file_name (str):
            Path to the CSV file containing training data.

        header_as_first_row (bool):
            If True, the first row of the CSV file is treated as the header.

        state_columns (int | str | list[int] | list[str]):
            For training, either one of the following for CSV columns representing
            state variables `x`:
                - a string or integer (for a single column)
                - a list of strings or integers (for multiple columns)
                - a string representing a slice of columns, e.g. '0:3'


        control_input_columns (int | str | list[int] | list[str]):
            For training, either one of the following for CSV columns representing
            control inputs `u`:
                - a string or integer (for a single column)
                - a list of strings or integers (for multiple columns)
                - a string representing a slice of columns, e.g. '0:3'
            If None, then the SINDy model will be trained without control inputs.

        dt (float):
            Fixed value of dt if rows of the CSV file represent equidistant time steps.

        time_column (str, int):
            Column name (str) for column index (int) for time data `t`.
            If `time_column` is provided, then fixed `dt` above will be ignored.
            If neither `dt` nor `time_column` is provided, then the SINDy model will
            use a fixed detault time step of `dt=1`.

        state_derivatives_columns (int | str | list[int] | list[str]):
            For training, either one of the following for csv columns representing
            state derivatives `x_dot`:
                - a string or integer (for a single column)
                - a list of strings or integers (for multiple columns)
                - a string representing a slice of columns, e.g. '0:3'
            This field is optional. If provided, the SINDy model will estimate directly
            use these state derivatives for training. If not provided, the SINDy model
            will approximate the state derivatives `dot_x = dx/dt` from `x` by using
            the specified `differentiation_method`.

        discrete_time (bool):
            If True, the SINDy model will be trained for discrete-time systems. In
            this case, the dynamical system is treated as a map. Rather than
            predicting derivatives, the right hand side functions step the system
            forward by one time step. If False, dynamical system is assumed to be a
            flow (right-hand side functions predict continuous time derivatives).
            See documentation for `pysindy`.

        differentiation_method (str):
            Method to use for differentiating the state data `x` to obtain state
            derivatives `dot_x = dx/dt`. Available options are:
                'centered difference' (default)

        threshold (float):
            Threshold for the Sequentially thresholded least squares (STLSQ) algorithm
            used for training SINDy model.

        alpha (float):
            Regularization strength for the STLSQ algorithm.

        max_iter (int):
            Maximum number of iterations for the STLSQ algorithm.

        normalize_columns (bool):
            If True, normalize the columns of the data matrix before regression.

        poly_order (int):
            Degree of polynomial features. Set to `None` to omit this library.

        fourier_n_frequencies (int):
            Number of Fourier frequencies. Set to `None` to omit this library.

        custom_basis_functions (list of functions):
            A list of custom basis functions to use for training the SINDy model.
            For example to include `f(x) = 1/x` and `g(x) = exp(-x)`,
            provide [lambda x: 1.0/(x.0 + 1e-06), lamda x: jnp.exp(-x)]

            Currently only supported for pycollimator interface. Calls from UI and
            pretrained model loading does not support custom basis functions.

        pretrained (bool):
            If True, use a pretrained model specified by the `pretrained_file_path`
            argument.

        pretrained_file_path (str, optional): Path to the pretrained model file.

        initial_state (ndarray):
                Initial state of the system for propagating the continuous-time
                or discrete-time system forward duiring simulation.

        discrete_time_update_interval (float):
            Interval at which the discrete-time model should be updated. Default
            is 1.0.

        equations (list of strings):
            (For internal UI use only) The identified system equations.

        base_feature_names (list of strings):
            (For internal UI use only) Features x_i and u_i.

        feature_names (list of strings):
            (For internal UI use only) Composed features with basis libraries.

        coefficients (ndarray):
            (For internal UI use only) Coefficients of the identified model.

        has_control_input (bool):
            (For internal UI use only) If True, the model was trained with control.
            For standard training from CSV file, this is inferred from the
            parameter `control_input_columns`.
    """

    @parameters(
        static=[
            "file_name",
            "header_as_first_row",
            "state_columns",
            "control_input_columns",
            "discrete_time",
            "dt",
            "time_column",
            "state_derivatives_columns",
            "differentiation_method",
            "threshold",
            "alpha",
            "max_iter",
            "normalize_columns",
            "poly_order",
            "fourier_n_frequencies",
            "pretrained",
            "equations",
            "discrete_time_update_interval",
            "pretrained_file_path",
        ],
        dynamic=["initial_state"],
    )
    def __init__(
        self,
        file_name=None,
        header_as_first_row=False,
        state_columns=1,
        control_input_columns=None,
        dt=None,
        time_column=None,
        state_derivatives_columns=None,
        discrete_time=False,
        differentiation_method="centered difference",
        # optimizer parameters
        threshold=0.1,
        alpha=0.05,
        max_iter=20,
        normalize_columns=False,
        # Library parameters
        poly_order=2,
        fourier_n_frequencies=None,
        custom_basis_functions=None,
        pretrained=False,
        pretrained_file_path=None,
        # for parameters obtained from UI training
        equations=None,
        base_feature_names=None,
        feature_names=None,
        coefficients=None,
        has_control_input=True,
        # Simulation parameters
        initial_state=None,
        discrete_time_update_interval=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._initcnt = 0

        _validate_leafsystem_inputs(
            pretrained,
            pretrained_file_path,
            dt,
            time_column,
            poly_order,
            fourier_n_frequencies,
        )

        ui_is_providing_pretrained_data = _validate_ui_pretrained_data(
            coefficients, feature_names, base_feature_names, self.name
        )

        if ui_is_providing_pretrained_data:
            self.equations = equations
            self.base_feature_names = base_feature_names.tolist()
            self.feature_names = feature_names.tolist()
            self.coefficients = cnp.array(coefficients)
            self.has_control_input = has_control_input
            self.custom_basis_functions = None

        elif pretrained:
            with open(pretrained_file_path, "r") as f:
                deserialized_model = json.load(f)

            self.equations = deserialized_model["equations"]
            self.base_feature_names = deserialized_model["base_feature_names"]
            self.feature_names = deserialized_model["feature_names"]
            self.coefficients = cnp.array(deserialized_model["coefficients"])
            self.has_control_input = deserialized_model["has_control_input"]
            self.custom_basis_functions = None

        else:
            (
                self.equations,
                self.base_feature_names,
                self.feature_names,
                self.coefficients,
                self.has_control_input,
            ) = train_from_csv(
                file_name,
                header_as_first_row=header_as_first_row,
                state_columns=state_columns,
                control_input_columns=control_input_columns,
                dt=dt,
                time_column=time_column,
                state_derivatives_columns=state_derivatives_columns,
                discrete_time=discrete_time,
                differentiation_method=differentiation_method,
                threshold=threshold,
                alpha=alpha,
                max_iter=max_iter,
                normalize_columns=normalize_columns,
                poly_order=poly_order,
                custom_basis_functions=custom_basis_functions,
                fourier_n_frequencies=fourier_n_frequencies,
            )
            self.custom_basis_functions = custom_basis_functions

        self.nx, _ = self.coefficients.shape

        if cnp.all(self.coefficients == 0):
            warnings.warn(
                "No features were selected for the SINDy model. "
                "Please check the training data and the feature selection "
                "parameters."
            )

        if initial_state is not None:
            if len(initial_state) != self.nx:
                raise ValueError(
                    f"Provided initial state has {len(initial_state)} elements. "
                    f"Expected {self.nx} elements."
                )
        else:
            initial_state = cnp.zeros(self.nx)

        if self.has_control_input:
            self.declare_input_port()  # one vector valued input port for u

        if discrete_time:
            self.declare_discrete_state(
                shape=(self.nx,),
                default_value=initial_state,
                as_array=True,
            )
            self.declare_periodic_update(
                (
                    self._discrete_update
                    if self.has_control_input
                    else lambda time, state, **params: self._discrete_update(
                        time, state, (), **params
                    )
                ),
                period=discrete_time_update_interval,
                offset=0.0,
            )
            self.declare_output_port(
                self._full_discrete_state_output,
                period=discrete_time_update_interval,
                offset=0.0,
                default_value=initial_state,
                requires_inputs=False,
            )

        else:
            self.declare_continuous_state(
                ode=(
                    self._ode
                    if self.has_control_input
                    else lambda time, state, **params: self._ode(
                        time, state, (), **params
                    )
                ),
                shape=(self.nx,),
                default_value=cnp.array(initial_state),
            )
            self.declare_continuous_state_output()  # output of the state in ODE

        # SymPy parsing to compute $f(x,u)$
        # For continuous-time systems $\dot{x} = f(x,u)$
        # For discrete-time systems $x_{k+1} = f(x_k, u_k)$
        sympy_base_features = sp.symbols(self.base_feature_names)

        # Convert feature names to sympy expressions
        sympy_feature_expressions = []
        for name in self.feature_names:
            # Replace spaces with multiplication
            name = name.replace(" ", "*")
            expr = sp.sympify(name)
            sympy_feature_expressions.append(expr)

        x_and_u_vec = sp.Matrix(sympy_base_features)
        custom_functions_dict = (
            {f"f{idx}": func for idx, func in enumerate(self.custom_basis_functions)}
            if self.custom_basis_functions
            else None
        )
        self.features_func = sp.lambdify(
            (x_and_u_vec,),
            sympy_feature_expressions,
            modules=[custom_functions_dict, "jax"] if custom_functions_dict else "jax",
        )  # feature functions

        self.declare_static_parameters(
            base_feature_names=self.base_feature_names,
            feature_names=self.feature_names,
            coefficients=self.coefficients.tolist(),
            has_control_input=self.has_control_input,
        )

    def initialize(self, **kwargs):
        if self._initcnt == 1:
            logger.warning(
                "SINDy block does not support re-initialization yet, this means "
                "that this block's parameter values will not change during ensemble "
                "simulations or optimizations. Use SINDy fit method instead.",
                **logdata(block=self),
            )
        self._initcnt += 1

    def _ode(self, _time, state, inputs, **_params):
        """
        The ODE system RHS. The RHS is given by `coefficients @ features`
        """
        x = state.continuous_state
        u = inputs
        x_and_u = cnp.hstack([x, u])
        features_evaluated = self.features_func(x_and_u)
        x_dot = cnp.matmul(self.coefficients, cnp.atleast_1d(features_evaluated))
        return x_dot

    def _discrete_update(self, _time, state, inputs, **_params):
        """
        Update map is given by `coefficients @ features`
        """
        x = state.discrete_state
        u = inputs
        x_and_u = cnp.hstack([x, u])
        features_evaluated = self.features_func(x_and_u)
        x_plus = cnp.matmul(self.coefficients, cnp.atleast_1d(features_evaluated))
        return x_plus

    def _full_discrete_state_output(self, _time, state, *_inputs, **_params):
        return state.discrete_state

    def serialize(self, filename):
        """
        Save the relevant class attributes post training
        so that model state can be restored
        """
        sindy_data = {
            "equations": self.equations,
            "base_feature_names": self.base_feature_names,
            "feature_names": self.feature_names,
            "coefficients": self.coefficients.tolist(),  # Can't serialize numpy arrays
            "has_control_input": self.has_control_input,
        }
        with open(filename, "w") as f:
            json.dump(sindy_data, f)

    @staticmethod
    def serialize_trained_pysindy_model(model, filename):
        """
        Serialize a PySindy model trained outside of Collimator.
        The saved file can be used as a pretrained model in Collimator.
        """

        feature_names, coefficients = _reduce(
            model.feature_names, model.get_feature_names(), model.coefficients()
        )

        has_control_input = model.model.n_input_features_ > 0

        sindy_data = {
            "equations": model.equations,
            "base_feature_names": model.feature_names,
            "feature_names": feature_names,
            "coefficients": coefficients.tolist(),  # Can't serialize numpy arrays
            "has_control_input": has_control_input,
        }

        with open(filename, "w") as f:
            json.dump(sindy_data, f)
