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

import argparse
import logging
import pickle

from collimator.library import sindy

logger = logging.getLogger(__name__)
Coefficients = list[float]
FeatureNames = list[str]
BaseFeatureNames = list[str]
Equations = list[str]
HasControlInput = bool
FitResult = tuple[
    Coefficients, FeatureNames, BaseFeatureNames, Equations, HasControlInput
]


def fit_sindy(
    file_name: str,
    state_columns: list[str] | list[int] | int | str,
    header_as_first_row: bool = False,
    control_input_columns: list[str] | list[int] | int | str | None = None,
    dt: float | None = None,
    time_column: str | int | None = None,
    state_derivatives_columns: list[str] | list[int] | int | str | None = None,
    discrete_time: bool = False,
    differentiation_method: str = "centered difference",
    threshold: float = 0.1,
    alpha: float = 0.05,
    max_iter: int = 20,
    normalize_columns: bool = False,
    poly_order: int | None = 2,
    fourier_n_frequencies: int | None = None,
) -> FitResult:
    equations, base_feature_names, feature_names, coefficients, has_control_input = (
        sindy.train_from_csv(
            file_name,
            state_columns=state_columns,
            header_as_first_row=header_as_first_row,
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
            fourier_n_frequencies=fourier_n_frequencies,
        )
    )

    logger.info(
        "Equations: %s, Base feature names: %s, Feature names: %s, Coefficients: %s, Has control input: %s",
        equations,
        base_feature_names,
        feature_names,
        coefficients,
        has_control_input,
    )

    return (
        coefficients.tolist(),
        feature_names,
        base_feature_names,
        equations,
        has_control_input,
    )


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Working directory.",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        required=False,
        help="Pickled kwargs filepath.",
    )

    return parser.parse_args()


def main():
    args = _get_args()

    pickled_kwargs = {}
    if args.kwargs:
        with open(args.kwargs, "rb") as f:
            pickled_kwargs = pickle.load(f)

    results = fit_sindy(**pickled_kwargs)

    with open(f"{args.work_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
