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

import jax.numpy as jnp

from collimator.optimization.framework.base import transformations


def assert_dicts_equal(dict1, dict2):
    assert dict1.keys() == dict2.keys(), "Dictionaries do not have the same keys"
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        assert jnp.allclose(val1, val2), f"Values for key {key} are not equal"


def test_transformation():
    params = {
        "x": 2.0,
        "y": jnp.array([5.0, 10.0]),
        "z": jnp.array([[10.0, 20.0], [30.0, 40.0]]),
    }

    params_min = {"x": 0.0, "y": -20.0, "z": 10.0}
    params_max = {"x": 4.0, "y": +20.0, "z": 50.0}

    normalize = transformations.NormalizeTransform(params_min, params_max)

    transformed_params = normalize.transform(params)

    expected_transfomed_params = {
        "x": 0.5,
        "y": jnp.array([0.625, 0.75]),
        "z": jnp.array([[0.0, 0.25], [0.5, 0.75]]),
    }

    assert_dicts_equal(transformed_params, expected_transfomed_params)
    assert_dicts_equal(normalize.inverse_transform(transformed_params), params)


def test_composite_transformation():
    params = {
        "x": 2.0,
        "y": jnp.array([5.0, 10.0]),
        "z": jnp.array([[10.0, 20.0], [30.0, 40.0]]),
    }

    params_min = {"x": 0.0, "y": -20.0, "z": 0.0}
    params_max = {"x": 4.0, "y": +20.0, "z": 50.0}

    normalize = transformations.NormalizeTransform(params_min, params_max)
    logit = transformations.LogitTransform()

    composite = transformations.CompositeTransform([normalize, logit])

    transformed_params = composite.transform(params)
    recovered_params = composite.inverse_transform(transformed_params)
    print(f"{params=}")
    print(f"{transformed_params=}")
    print(f"{recovered_params=}")

    assert_dicts_equal(recovered_params, params)


if __name__ == "__main__":
    test_transformation()
    test_composite_transformation()
