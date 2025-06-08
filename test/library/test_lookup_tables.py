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

"""Test blocks using interpolation

Contains tests for:
- LookupTable1d
- LookupTable2d
"""

import pytest
from scipy.interpolate import LinearNDInterpolator
import jax.numpy as jnp
import collimator
from collimator import library
from collimator.backend import numpy_api as cnp


class TestLookupTable1d:
    @pytest.fixture
    def input_array(self):
        return jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    @pytest.fixture
    def output_array(self):
        return jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])

    def evaluate_lookuptable_block_output(
        self, input, input_array, output_array, interpolation_type
    ):
        builder = collimator.DiagramBuilder()
        lookup_block = builder.add(
            library.LookupTable1d(input_array, output_array, interpolation_type)
        )
        input_block = builder.add(library.Constant(input))
        builder.connect(input_block.output_ports[0], lookup_block.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        return lookup_block.output_ports[0].eval(ctx)

    @pytest.mark.parametrize("input,expected", [(-0.5, 0.0), (2.5, 6.5), (4.5, 16.0)])
    @pytest.mark.parametrize("backend", ["jax", "numpy"])
    def test_linear_interpolation(
        self,
        input,
        expected,
        input_array,
        output_array,
        backend,
    ):
        if backend == "numpy":
            cnp.set_backend("numpy")
        else:
            cnp.set_backend("jax")

        assert jnp.allclose(
            self.evaluate_lookuptable_block_output(
                input, input_array, output_array, "linear"
            ),
            expected,
        )
        cnp.set_backend("jax")

    @pytest.mark.parametrize(
        "input,expected",
        [(-0.5, 0.0), (2.3, 4.0), (2.5, 4.0), (2.51, 9.0), (4.5, 16.0)],
    )
    def test_nearest_interpolation(self, input, expected, input_array, output_array):
        assert jnp.allclose(
            self.evaluate_lookuptable_block_output(
                input, input_array, output_array, "nearest"
            ),
            expected,
        )

    @pytest.mark.parametrize(
        "input,expected",
        [(-0.5, 0.0), (2.3, 4.0), (2.5, 4.0), (2.8, 4.0), (3.1, 9.0), (4.5, 16.0)],
    )
    def test_flat_interpolation(self, input, expected, input_array, output_array):
        assert jnp.allclose(
            self.evaluate_lookuptable_block_output(
                input, input_array, output_array, "flat"
            ),
            expected,
        )

    def test_invalid_interpolation(self, input_array, output_array):
        with pytest.raises(ValueError):
            library.LookupTable1d(input_array, output_array, "invalid")

    def test_invalid_input_array(self, output_array):
        with pytest.raises(ValueError):
            library.LookupTable1d(jnp.array([[0, 1], [2, 3]]), output_array, "linear")

    def test_invalid_output_array(self, input_array):
        with pytest.raises(ValueError):
            library.LookupTable1d(input_array, jnp.array([[0, 1], [2, 3]]), "linear")


class TestLookupTable2d:
    @pytest.fixture
    def input_x_array(self):
        return jnp.array([0, 1, 2, 3, 4], dtype=jnp.float64)

    @pytest.fixture
    def input_y_array(self):
        return jnp.array([0, 1, 2, 3], dtype=jnp.float64)

    @pytest.fixture
    def output_table_array(self, input_x_array, input_y_array):
        x, y = jnp.meshgrid(input_x_array, input_y_array, indexing="ij")
        return x**2 + y**2

    @pytest.fixture
    def scipy_interpolant(self, input_x_array, input_y_array, output_table_array):
        x, y = jnp.meshgrid(input_x_array, input_y_array, indexing="ij")
        xy = jnp.vstack([x.flatten(), y.flatten()]).T
        interp = LinearNDInterpolator(xy, output_table_array.flatten())
        return interp

    def evaluate_lookuptable_block_output(
        self, input_x, input_y, input_x_array, input_y_array, output_table_array
    ):
        builder = collimator.DiagramBuilder()
        lookup_block = builder.add(
            library.LookupTable2d(
                input_x_array, input_y_array, output_table_array, "linear"
            )
        )
        input_x_block = builder.add(library.Constant(input_x))
        input_y_block = builder.add(library.Constant(input_y))
        builder.connect(input_x_block.output_ports[0], lookup_block.input_ports[0])
        builder.connect(input_y_block.output_ports[0], lookup_block.input_ports[1])
        diagram = builder.build()
        ctx = diagram.create_context()
        return lookup_block.output_ports[0].eval(ctx)

    @pytest.mark.parametrize(
        "input_x,input_y",
        [
            (-0.5, 0.0),
            (0.0, -1.0),
            (0.5, 0.7),
            (1.0, 2.5),
            (3.3, 2.2),
            (3.9, 3.1),
            (4.5, 3.5),
        ],
    )
    @pytest.mark.parametrize("backend", ["jax", "numpy"])
    def test_linear_interpolation(
        self,
        input_x,
        input_y,
        input_x_array,
        input_y_array,
        output_table_array,
        scipy_interpolant,
        backend,
    ):
        if backend == "numpy":
            cnp.set_backend("numpy")
        else:
            cnp.set_backend("jax")

        output = self.evaluate_lookuptable_block_output(
            input_x, input_y, input_x_array, input_y_array, output_table_array
        )

        clipped_input_x = jnp.clip(
            input_x, jnp.min(input_x_array), jnp.max(input_x_array)
        )
        clipped_input_y = jnp.clip(
            input_y, jnp.min(input_y_array), jnp.max(input_y_array)
        )
        expected = scipy_interpolant(clipped_input_x, clipped_input_y)

        assert jnp.allclose(output, expected)
        cnp.set_backend("jax")

    def test_invalid_interpolation(
        self, input_x_array, input_y_array, output_table_array
    ):
        with pytest.raises(NotImplementedError):
            library.LookupTable2d(
                input_x_array, input_y_array, output_table_array, "invalid"
            )

    def test_invalid_input_x_array(self, input_y_array, output_table_array):
        with pytest.raises(ValueError):
            library.LookupTable2d(
                jnp.array([[0, 1], [2, 3]]), input_y_array, output_table_array, "linear"
            )

    def test_invalid_input_y_array(self, input_x_array, output_table_array):
        with pytest.raises(ValueError):
            library.LookupTable2d(
                input_x_array, jnp.array([[0, 1], [2, 3]]), output_table_array, "linear"
            )

    def test_invalid_output_table_array(self, input_x_array, input_y_array):
        with pytest.raises(ValueError):
            library.LookupTable2d(
                input_x_array, input_y_array, jnp.array([0, 1, 2, 3, 4]), "linear"
            )

    def test_input_output_shape_mismatch(self, input_x_array, input_y_array):
        with pytest.raises(ValueError):
            library.LookupTable2d(
                input_x_array, input_y_array, jnp.ones((4, 5)), "linear"
            )


if __name__ == "__main__":
    cnp.set_backend("numpy")
    tlut2 = TestLookupTable2d()
    tlut2.evaluate_lookuptable_block_output(
        0.5,
        1,
        jnp.array([0, 1], dtype=jnp.float64),
        jnp.array([0, 1], dtype=jnp.float64),
        jnp.array([[0, 1], [2, 4]], dtype=jnp.float64),
    )
