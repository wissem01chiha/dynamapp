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

"""Test primitive math blocks.

Contains tests for:
- Abs
- Adder
- CrossProduct
- DotProduct
- Exponent
- Gain
- Logarithm
- MatrixConcatenation
- MatrixInversion
- MatrixTransposition
- MatrixMultiplication
- Offset
- Power
- Product
- ProductOfElements
- Reciprocal
- ScalarBroadcast
- Slice
- SumOfElements

"""

import pytest
import numpy as np
import jax.numpy as jnp

import collimator
from collimator import library
from collimator.framework.error import BlockParameterError
from collimator.framework.error import StaticError
from collimator.logging import logger
from collimator.backend import numpy_api as cnp


pytestmark = pytest.mark.minimal

float_dtypes = [
    jnp.float64,
    jnp.float32,
    jnp.float16,
]

int_dtypes = [
    jnp.int64,
    jnp.int32,
    jnp.int16,
]


class TestAbs:
    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_sin_input(self, dtype):
        t = np.linspace(0.0, 10.0, 100, dtype=dtype)

        block = library.Abs(name="Abs_0")
        block.input_ports[0].fix_value(0)
        ctx = block.create_context()

        block.input_ports[0].fix_value(np.sin(t))
        x = block.output_ports[0].eval(ctx)
        assert np.allclose(x, np.abs(jnp.sin(t)))
        assert isinstance(x, cnp.ndarray)
        assert x.shape == t.shape
        assert x.dtype == dtype

    def test_sine_simulation(self):
        builder = collimator.DiagramBuilder()
        sine = builder.add(library.Sine(name="sine"))
        abs = builder.add(library.Abs(name="abs"))
        builder.connect(sine.output_ports[0], abs.input_ports[0])

        system = builder.build()
        context = system.create_context()

        recorded_signals = {"sine": sine.output_ports[0], "abs": abs.output_ports[0]}
        options = collimator.SimulatorOptions(max_major_step_length=0.1)
        results = collimator.simulate(
            system,
            context,
            (0.0, 10.0),
            recorded_signals=recorded_signals,
            options=options,
        )

        assert jnp.allclose(results.outputs["abs"], jnp.abs(results.outputs["sine"]))

    def test_diagram_simulation(self):
        "Part of the FeedthruBlocks test case"
        builder = collimator.DiagramBuilder()
        const = builder.add(library.Constant(value=1.0, name="const"))
        integrator = builder.add(
            library.Integrator(initial_state=0.0, name="integrator")
        )
        gain = builder.add(library.Gain(gain=-1.0, name="gain"))
        abs = builder.add(library.Abs(name="abs"))

        builder.connect(const.output_ports[0], integrator.input_ports[0])
        builder.connect(integrator.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], abs.input_ports[0])

        system = builder.build()
        context = system.create_context()

        recorded_signals = {"abs": abs.output_ports[0]}
        options = collimator.SimulatorOptions(max_major_step_length=0.1)
        results = collimator.simulate(
            system,
            context,
            (0.0, 10.0),
            recorded_signals=recorded_signals,
            options=options,
        )

        assert jnp.allclose(results.outputs["abs"], results.time)

    def test_conditional_zc_event(self):
        # Check that a zero-crossing event is only declared when the input
        # is used as the RHS of an ODE.

        builder = collimator.DiagramBuilder()
        clock = builder.add(library.Clock(name="clock"))
        abs_1 = builder.add(library.Abs(name="abs_1"))
        abs_2 = builder.add(library.Abs(name="abs_2"))
        int_1 = builder.add(library.Integrator(initial_state=0.0, name="int_1"))
        builder.connect(clock.output_ports[0], abs_1.input_ports[0])
        builder.connect(clock.output_ports[0], abs_2.input_ports[0])
        builder.connect(abs_2.output_ports[0], int_1.input_ports[0])

        system = builder.build()

        # Creating the context does the static analysis, which is where the
        # zero-crossing events are conditionally declared
        system.create_context()

        assert len(abs_1.zero_crossing_events) == 0
        assert len(abs_2.zero_crossing_events) == 1


class TestAdder:
    scalars = [[1], [2], [3]]
    vectors2 = [[1, 2]] * 3
    vectors3 = [[1, 2, 3]] * 3
    mat3x3 = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]] * 3
    mat2x4 = [[[2, 3], [2, 3], [2, 3], [2, 3]]] * 3

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    @pytest.mark.parametrize("inputs", [scalars, vectors2, vectors3, mat3x3, mat2x4])
    def test_different_inputs(self, dtype, inputs):
        Adder_0 = library.Adder(3, name="Adder_0")
        assert len(Adder_0.input_ports) == 3
        assert len(Adder_0.output_ports) == 1

        Constant_0 = library.Constant(jnp.array(inputs[0], dtype=dtype), name="const1")
        Constant_1 = library.Constant(jnp.array(inputs[1], dtype=dtype), name="const2")
        Constant_2 = library.Constant(jnp.array(inputs[2], dtype=dtype), name="const3")

        builder = collimator.DiagramBuilder()
        builder.add(Adder_0, Constant_0, Constant_1, Constant_2)

        builder.connect(Constant_0.output_ports[0], Adder_0.input_ports[0])
        builder.connect(Constant_1.output_ports[0], Adder_0.input_ports[1])
        builder.connect(Constant_2.output_ports[0], Adder_0.input_ports[2])

        diagram = builder.build()
        ctx = diagram.create_context()
        y = Adder_0.output_ports[0].eval(ctx)

        # compute expected solution
        in0 = np.array(inputs[0], dtype=dtype)
        in1 = np.array(inputs[1], dtype=dtype)
        in2 = np.array(inputs[2], dtype=dtype)
        y_sol = in0 + in1 + in2

        assert jnp.allclose(y, y_sol)
        assert y.dtype == dtype

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_operators(self, dtype):
        Adder_0 = library.Adder(3, name="Adder_0", operators="+-+")
        assert len(Adder_0.input_ports) == 3
        assert len(Adder_0.output_ports) == 1

        Constant_0 = library.Constant(jnp.array([1], dtype=dtype), name="const1")
        Constant_1 = library.Constant(jnp.array([2], dtype=dtype), name="const2")
        Constant_2 = library.Constant(jnp.array([3], dtype=dtype), name="const3")

        builder = collimator.DiagramBuilder()
        builder.add(Adder_0, Constant_0, Constant_1, Constant_2)

        builder.connect(Constant_0.output_ports[0], Adder_0.input_ports[0])
        builder.connect(Constant_1.output_ports[0], Adder_0.input_ports[1])
        builder.connect(Constant_2.output_ports[0], Adder_0.input_ports[2])

        diagram = builder.build()

        ctx = diagram.create_context()

        diagram.pprint()

        y = Adder_0.output_ports[0].eval(ctx)

        assert jnp.allclose(y, 2)
        assert y.dtype == dtype

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.Adder(2, operators="*/", name="Adder")
        # Success! The test failed as expected.
        print(e)
        assert (
            "Adder block Adder has invalid operators */. Can only contain '+' and '-'"
            in str(e)
        )


class TestCrossProduct:
    def test_cross_product(self):
        arr1 = np.array([1.0, 1.0, 1.0])
        arr2 = np.array([2.0, 2.0, 2.0])
        vec1 = library.Constant(arr1)
        vec2 = library.Constant(arr2)
        cp = library.CrossProduct()

        builder = collimator.DiagramBuilder()
        builder.add(vec1, vec2, cp)
        builder.connect(vec1.output_ports[0], cp.input_ports[0])
        builder.connect(vec2.output_ports[0], cp.input_ports[1])
        diagram = builder.build()
        ctx = diagram.create_context()

        y = cp.output_ports[0].eval(ctx)

        assert np.allclose(y, np.cross(arr1, arr2))
        assert y.dtype == jnp.float64


class TestDotProduct:
    def test_dot_product(self):
        vec1 = library.Constant(np.array([1.0, 1.0, 1.0]))
        vec2 = library.Constant(np.array([2.0, 2.0, 2.0]))
        dp = library.DotProduct()

        builder = collimator.DiagramBuilder()
        builder.add(vec1, vec2, dp)
        builder.connect(vec1.output_ports[0], dp.input_ports[0])
        builder.connect(vec2.output_ports[0], dp.input_ports[1])
        diagram = builder.build()
        ctx = diagram.create_context()

        y = dp.output_ports[0].eval(ctx)

        assert y == 6.0
        assert y.dtype == jnp.float64


class TestExponent:
    def test_floats(self):
        four = library.Constant(4.0)
        exp2 = library.Exponent(base="2")
        exp = library.Exponent(base="exp")

        builder = collimator.DiagramBuilder()
        builder.add(four, exp2, exp)
        builder.connect(four.output_ports[0], exp2.input_ports[0])
        builder.connect(four.output_ports[0], exp.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        y2 = exp2.output_ports[0].eval(ctx)
        y = exp.output_ports[0].eval(ctx)

        assert jnp.allclose(y2, jnp.exp2(4.0))
        assert y2.dtype == jnp.float64
        assert jnp.allclose(y, jnp.exp(4.0))

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.Exponent(base=2, name="Exponent")
        # Success! The test failed as expected.
        print(e)
        assert (
            "Exponent block Exponent has invalid selection 2 for 'base'. Valid selections: exp, 2"
            in str(e)
        )


class TestGain:
    def _make_diagram(self, Constant_0, Gain_0):
        builder = collimator.DiagramBuilder()

        builder.add(Constant_0, Gain_0)
        builder.connect(Constant_0.output_ports[0], Gain_0.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        gain_ctx = ctx[Gain_0.system_id]
        return diagram, ctx, gain_ctx

    def test_floats(self, k=0.5):
        Constant_0 = library.Constant(3.0, name="Constant_0")
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)
        diagram.pprint(logger.debug)

        y = Gain_0.output_ports[0].eval(ctx)  # 1.5
        assert y == 1.5
        assert y.dtype == jnp.float64

    def test_ints(self, k=2):
        Constant_0 = library.Constant(3, name="Constant_0")
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)

        y = Gain_0.output_ports[0].eval(ctx)
        assert y == 6
        assert y.dtype == jnp.int64

    def test_scalar_type_promotion(self, k=2):
        Constant_0 = library.Constant(3.0, name="Constant_0")
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)

        y = Gain_0.output_ports[0].eval(ctx)
        assert y == 6.0
        assert y.dtype == jnp.float64

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_input_array_1d(self, dtype, k=2):
        Constant_0 = library.Constant(
            np.array([1, 2, 3], dtype=dtype), name="Constant_0"
        )
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)

        y = Gain_0.output_ports[0].eval(ctx)
        assert np.allclose(y, np.array([2, 4, 6]))
        assert isinstance(y, cnp.ndarray)
        assert y.shape == (3,)
        assert y.dtype == dtype

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_input_array_2d(self, dtype, k=2):
        C = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)

        Constant_0 = library.Constant(C, name="Constant_0")
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)

        y = Gain_0.output_ports[0].eval(ctx)
        assert np.allclose(y, C * k)
        assert isinstance(y, cnp.ndarray)
        assert y.shape == C.shape
        assert y.dtype == dtype

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_gain_array_1d(self, dtype):
        k = np.array([1, 2, 3], dtype=dtype)

        Constant_0 = library.Constant(3, name="Constant_0")
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)

        y = Gain_0.output_ports[0].eval(ctx)
        assert np.allclose(y, np.array([3, 6, 9]))
        assert isinstance(y, cnp.ndarray)
        assert y.shape == (3,)
        assert y.dtype == dtype

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_gain_array_2d(self, dtype):
        k = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)

        Constant_0 = library.Constant(3, name="Constant_0")
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)

        y = Gain_0.output_ports[0].eval(ctx)
        assert np.allclose(y, np.array([[3, 6, 9], [12, 15, 18]]))
        assert isinstance(y, cnp.ndarray)
        assert y.shape == k.shape
        assert y.dtype == dtype

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_both_arrays(self, dtype):
        # Also tests broadcasting
        C = np.array([1, 2, 3], dtype=dtype)
        k = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)

        Constant_0 = library.Constant(C, name="Constant_0")
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)

        y = Gain_0.output_ports[0].eval(ctx)
        assert np.allclose(y, np.array([[1, 4, 9], [4, 10, 18]]))
        assert isinstance(y, cnp.ndarray)
        assert y.shape == k.shape
        assert y.dtype == dtype

    def test_vector_type_promotion(self):
        # Also tests broadcasting
        C = np.array([1, 2, 3], dtype=jnp.int32)
        k = np.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float64)

        Constant_0 = library.Constant(C, name="Constant_0")
        Gain_0 = library.Gain(k, name="Gain_0")

        diagram, ctx, gain_ctx = self._make_diagram(Constant_0, Gain_0)

        y = Gain_0.output_ports[0].eval(ctx)
        assert np.allclose(y, np.array([[1, 4, 9], [4, 10, 18]]))
        assert isinstance(y, cnp.ndarray)
        assert y.shape == k.shape
        assert y.dtype == np.float64


class TestLogarithm:
    def test_floats(self):
        four = library.Constant(4.0)
        log = library.Logarithm(base="natural")
        log2 = library.Logarithm(base="2")
        log10 = library.Logarithm(base="10")

        builder = collimator.DiagramBuilder()
        builder.add(four, log, log2, log10)
        builder.connect(four.output_ports[0], log.input_ports[0])
        builder.connect(four.output_ports[0], log2.input_ports[0])
        builder.connect(four.output_ports[0], log10.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        y = log.output_ports[0].eval(ctx)
        y2 = log2.output_ports[0].eval(ctx)
        y10 = log10.output_ports[0].eval(ctx)

        assert jnp.allclose(y, jnp.log(4.0))
        assert y.dtype == jnp.float64
        assert jnp.allclose(y2, jnp.log2(4.0))
        assert jnp.allclose(y10, jnp.log10(4.0))

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.Logarithm(base=2, name="Logarithm")
        # Success! The test failed as expected.
        print(e)
        assert (
            "Logarithm block Logarithm has invalid selection 2 for 'base'. Valid selections: 10, 2, natural"
            in str(e)
        )


class TestOffset:
    def test_floats(self):
        four = library.Constant(4.0)
        offset = library.Offset(offset=2.0)

        builder = collimator.DiagramBuilder()
        builder.add(four, offset)
        builder.connect(four.output_ports[0], offset.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        y = offset.output_ports[0].eval(ctx)

        assert jnp.allclose(y, 4.0 + 2.0)
        assert y.dtype == jnp.float64


class TestPower:
    def test_floats(self):
        four = library.Constant(4.0)
        power = library.Power(exponent=2.0)

        builder = collimator.DiagramBuilder()
        builder.add(four, power)
        builder.connect(four.output_ports[0], power.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        y = power.output_ports[0].eval(ctx)

        assert jnp.allclose(y, 4.0**2.0)
        assert y.dtype == jnp.float64


class TestProduct:
    scalars = [[1], [2], [3]]
    vectors2 = [[1, 2]] * 3
    vectors3 = [[1, 2, 3]] * 3
    mat3x3 = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]] * 3
    mat2x4 = [[[2, 3], [2, 3], [2, 3], [2, 3]]] * 3

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    @pytest.mark.parametrize("inputs", [scalars, vectors2, vectors3, mat3x3, mat2x4])
    @pytest.mark.parametrize("operators", [None, "**/"])
    def test_different_inputs(self, dtype, inputs, operators):
        prod = library.Product(3, operators=operators)
        assert len(prod.input_ports) == 3
        assert len(prod.output_ports) == 1

        Constant_0 = library.Constant(jnp.array(inputs[0], dtype=dtype), name="const1")
        Constant_1 = library.Constant(jnp.array(inputs[1], dtype=dtype), name="const2")
        Constant_2 = library.Constant(jnp.array(inputs[2], dtype=dtype), name="const3")

        builder = collimator.DiagramBuilder()
        builder.add(prod, Constant_0, Constant_1, Constant_2)

        builder.connect(Constant_0.output_ports[0], prod.input_ports[0])
        builder.connect(Constant_1.output_ports[0], prod.input_ports[1])
        builder.connect(Constant_2.output_ports[0], prod.input_ports[2])

        diagram = builder.build()
        ctx = diagram.create_context()
        y = prod.output_ports[0].eval(ctx)

        # compute expected solution
        in0 = np.array(inputs[0], dtype=dtype)
        in1 = np.array(inputs[1], dtype=dtype)
        in2 = np.array(inputs[2], dtype=dtype)
        if operators is None:
            y_sol = in0 * in1 * in2
        else:
            y_sol = in0 * in1 / in2

        assert jnp.allclose(y, y_sol)
        # assert y.dtype == dtype # jnp does some type promotion. let's not bother validating that.

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_operators(self, dtype):
        prod = library.Product(3, operators="**/")
        assert len(prod.input_ports) == 3
        assert len(prod.output_ports) == 1

        Constant_0 = library.Constant(jnp.array([6], dtype=dtype), name="const1")
        Constant_1 = library.Constant(jnp.array([2], dtype=dtype), name="const2")
        Constant_2 = library.Constant(jnp.array([3], dtype=dtype), name="const3")

        builder = collimator.DiagramBuilder()
        builder.add(prod, Constant_0, Constant_1, Constant_2)

        builder.connect(Constant_0.output_ports[0], prod.input_ports[0])
        builder.connect(Constant_1.output_ports[0], prod.input_ports[1])
        builder.connect(Constant_2.output_ports[0], prod.input_ports[2])

        diagram = builder.build()
        ctx = diagram.create_context()
        y = prod.output_ports[0].eval(ctx)

        assert jnp.allclose(y, np.array(4, dtype=dtype))
        # assert y.dtype == dtype # jnp does some type promotion. let's not bother validating that.

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.Product(2, operators="+-", name="Product")
        # Success! The test failed as expected.
        print(e)
        assert (
            "Product block Product has invalid operators +-. Can only contain '*' and '/'"
            in str(e)
        )


class TestProductOfElements:
    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_scalar_vector_inputs(self, dtype):
        prod_s = library.ProductOfElements()
        prod_v = library.ProductOfElements()

        Constant_0 = library.Constant(jnp.array([1, 2, 3], dtype=dtype), name="const1")
        Constant_1 = library.Constant(
            jnp.array([[1, 2, 3], [1, 2, 3]], dtype=dtype), name="const2"
        )

        builder = collimator.DiagramBuilder()
        builder.add(prod_s, prod_v, Constant_0, Constant_1)

        builder.connect(Constant_0.output_ports[0], prod_s.input_ports[0])
        builder.connect(Constant_1.output_ports[0], prod_v.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        y_s = prod_s.output_ports[0].eval(ctx)
        y_v = prod_v.output_ports[0].eval(ctx)

        assert np.allclose(y_s, 6)
        # assert y_s.dtype == dtype # jnp does some type promotion. let's not bother validating that.
        assert np.allclose(y_v, 36)


class TestReciprocal:
    def _make_diagram(self, Constant_0, Reciprocal_0):
        builder = collimator.DiagramBuilder()

        builder.add(Constant_0, Reciprocal_0)
        builder.connect(Constant_0.output_ports[0], Reciprocal_0.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        reciprocal_ctx = ctx[Reciprocal_0.system_id]
        return diagram, ctx, reciprocal_ctx

    def test_floats(self):
        Constant_0 = library.Constant(2.0, name="Constant_0")
        Reciprocal_0 = library.Reciprocal(name="Reciprocal_0")

        diagram, ctx, reciprocal_ctx = self._make_diagram(Constant_0, Reciprocal_0)
        diagram.pprint(logger.debug)

        y = Reciprocal_0.output_ports[0].eval(ctx)  # 0.5
        assert y == 0.5
        print(y)
        print(type(y))
        assert y.dtype == jnp.float64


class TestConcat:
    def _make_diagram(self, Constant_0, Constant_1, Concat_0):
        builder = collimator.DiagramBuilder()

        builder.add(Constant_0, Constant_1, Concat_0)
        builder.connect(Constant_0.output_ports[0], Concat_0.input_ports[0])
        builder.connect(Constant_1.output_ports[0], Concat_0.input_ports[1])

        diagram = builder.build()
        ctx = diagram.create_context()
        concat_ctx = ctx[Concat_0.system_id]
        return diagram, ctx, concat_ctx

    def test_floats(self):
        cv0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        cv1 = jnp.array([[1.0, 2.0], [9.0, 9.0]])
        Constant_0 = library.Constant(cv0, name="Constant_0")
        Constant_1 = library.Constant(cv1, name="Constant_1")
        Concat_0 = library.MatrixConcatenation(name="Concat_0")

        diagram, ctx, concat_ctx = self._make_diagram(Constant_0, Constant_1, Concat_0)
        diagram.pprint(logger.debug)

        print(cv0)
        print(cv0.shape)

        y = Concat_0.output_ports[0].eval(ctx)
        y_sol = jnp.concatenate((cv0, cv1))
        print(y)
        print(y.shape)
        print(y_sol)
        print(y_sol.shape)
        assert y.shape == y_sol.shape
        assert y.dtype == jnp.float64


class TestScalarBroadcast:
    def _make_diagram(self, Constant_0, ScalarBroadcast_0):
        builder = collimator.DiagramBuilder()

        builder.add(Constant_0, ScalarBroadcast_0)
        builder.connect(Constant_0.output_ports[0], ScalarBroadcast_0.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        sb_ctx = ctx[ScalarBroadcast_0.system_id]
        return diagram, ctx, sb_ctx

    @pytest.mark.parametrize("m, n", [(2, 3), (3, 1), (1, 2)])
    def test_floats(self, m, n):
        Constant_0 = library.Constant(2.0, name="Constant_0")
        ScalarBroadcast_0 = library.ScalarBroadcast(name="ScalarBroadcast_0", m=m, n=n)

        diagram, ctx, sb_ctx = self._make_diagram(Constant_0, ScalarBroadcast_0)
        diagram.pprint(logger.debug)

        y = ScalarBroadcast_0.output_ports[0].eval(ctx)  # np.ones((m,n))*2.0
        assert jnp.allclose(y, jnp.ones((m, n)) * 2.0)
        assert y.dtype == jnp.float64

    @pytest.mark.parametrize("m, n", [(3, None), (None, 2), (2, 0), (0, 3)])
    def test_one_dimensional_floats(self, m, n):
        Constant_0 = library.Constant(2.0, name="Constant_0")
        ScalarBroadcast_0 = library.ScalarBroadcast(name="ScalarBroadcast_0", m=m, n=n)

        diagram, ctx, sb_ctx = self._make_diagram(Constant_0, ScalarBroadcast_0)
        diagram.pprint(logger.debug)

        y = ScalarBroadcast_0.output_ports[0].eval(ctx)
        if m is None or m == 0:
            assert jnp.allclose(y, jnp.ones(n) * 2.0)
        elif n is None or n == 0:
            assert jnp.allclose(y, jnp.ones(m) * 2.0)
        assert y.dtype == jnp.float64

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.ScalarBroadcast(m=None, n=0, name="ScalarBroadcast")
        # Success! The test failed as expected.
        print(e)
        assert (
            "ScalarBroadcast block ScalarBroadcast at least m or n must not be None or Zero"
            in str(e)
        )


class TestMatrixConcatenation:
    @pytest.fixture
    def input_matrix_A_1d(self):
        return jnp.array([1, 2])

    @pytest.fixture
    def input_matrix_A_2d(self):
        return jnp.array([[1, 2], [3, 4]])

    @pytest.fixture
    def input_matrix_B_1d(self):
        return jnp.array([3, 4])

    @pytest.fixture
    def input_matrix_B_2d(self):
        return jnp.array([[5, 6], [7, 8]])

    def evaluate_concatenation_output(self, input_matrix_a, input_matrix_b, axis=None):
        builder = collimator.DiagramBuilder()
        concatenation_block = builder.add(
            library.MatrixConcatenation(
                axis=axis,
            )
            if axis is not None
            else library.MatrixConcatenation()
        )
        input_x_block = builder.add(library.Constant(input_matrix_a))
        input_y_block = builder.add(library.Constant(input_matrix_b))
        builder.connect(
            input_x_block.output_ports[0], concatenation_block.input_ports[0]
        )
        builder.connect(
            input_y_block.output_ports[0], concatenation_block.input_ports[1]
        )
        diagram = builder.build()
        ctx = diagram.create_context()
        return concatenation_block.output_ports[0].eval(ctx)

    @pytest.mark.parametrize(
        "axis, expected",
        [
            (1, jnp.array([[1, 2, 5, 6], [3, 4, 7, 8]])),
            (0, jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])),
            (1, jnp.array([[1, 2, 5, 6], [3, 4, 7, 8]])),
            (0, jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])),
        ],
    )
    def test_concatenation_2d(
        self,
        axis,
        expected,
        input_matrix_A_2d,
        input_matrix_B_2d,
    ):
        result = self.evaluate_concatenation_output(
            input_matrix_A_2d,
            input_matrix_B_2d,
            axis=axis,
        )
        assert jnp.array_equal(result, expected)

    @pytest.mark.parametrize(
        "axis, expected",
        [
            (0, jnp.array([1, 2, 3, 4])),
            (0, jnp.array([1, 2, 3, 4])),
        ],
    )
    def test_vertical_concatenation_1d(
        self,
        axis,
        expected,
        input_matrix_A_1d,
        input_matrix_B_1d,
    ):
        result = self.evaluate_concatenation_output(
            input_matrix_A_1d,
            input_matrix_B_1d,
            axis=axis,
        )
        assert jnp.array_equal(result, expected)

    def test_invalid_concatenation_orientation_1d_2d(
        self, input_matrix_A_1d, input_matrix_B_2d
    ):
        with pytest.raises(StaticError):
            _ = self.evaluate_concatenation_output(
                input_matrix_A_1d,
                input_matrix_B_2d,
                # axis=0,  # test default
            )

    def test_invalid_concatenation_orientation_1d_1d_horizontal(
        self, input_matrix_A_1d, input_matrix_B_1d
    ):
        with pytest.raises(StaticError):
            _ = self.evaluate_concatenation_output(
                input_matrix_A_1d,
                input_matrix_B_1d,
                axis=1,
            )


class TestMatrixInversion:
    def evaluate_matrix_inversion(self, input):
        builder = collimator.DiagramBuilder()
        inversion_block = builder.add(library.MatrixInversion())
        input_block = builder.add(library.Constant(input))
        builder.connect(input_block.output_ports[0], inversion_block.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        return inversion_block.output_ports[0].eval(ctx)

    @pytest.mark.parametrize(
        "matrix,expected",
        [
            (
                jnp.array([[1, 2], [3, 4]]),
                jnp.array([[-2, 1], [1.5, -0.5]]),
            ),
            (jnp.eye(4), jnp.eye(4)),
        ],
    )
    def test_known_inverse(self, matrix, expected):
        assert jnp.allclose(
            self.evaluate_matrix_inversion(matrix), expected
        ), "Inverse does not match expected result for known matrix."

    def test_singular_matrix(self):
        singular_matrix = jnp.array([[1, 2], [2, 4]])  # This matrix is singular
        assert jnp.all(jnp.isinf(self.evaluate_matrix_inversion(singular_matrix)))

    @pytest.mark.parametrize("size", [3, 5, 10])
    def test_random_matrix(self, size):
        from jax import random

        key = random.PRNGKey(0)
        matrix = random.uniform(key, shape=(size, size))
        matrix = matrix @ matrix.T
        inv_matrix = self.evaluate_matrix_inversion(matrix)
        assert jnp.allclose(
            jnp.dot(matrix, inv_matrix), jnp.eye(size)
        ), "AA^-1 != I for random symmetric matrix."
        assert jnp.allclose(
            jnp.dot(inv_matrix, matrix), jnp.eye(size)
        ), "A^-1A != I for random symmetric matrix."


class TestMatrixMultiplication:
    def evaluate_matrix_multiplication(
        self,
        input_matrix_a,
        input_matrix_b,
    ):
        builder = collimator.DiagramBuilder()
        multiplication_block = builder.add(library.MatrixMultiplication())
        input_x_block = builder.add(library.Constant(input_matrix_a))
        input_y_block = builder.add(library.Constant(input_matrix_b))
        builder.connect(
            input_x_block.output_ports[0], multiplication_block.input_ports[0]
        )
        builder.connect(
            input_y_block.output_ports[0], multiplication_block.input_ports[1]
        )
        diagram = builder.build()
        ctx = diagram.create_context()
        return multiplication_block.output_ports[0].eval(ctx)

    @pytest.mark.parametrize(
        "A,B,expected",
        [
            # Test square matrices
            (
                jnp.array([[1, 2], [3, 4]]),
                jnp.array([[2, 0], [1, 2]]),
                jnp.array([[4, 4], [10, 8]]),
            ),
            # Test identity matrix
            (
                jnp.eye(3),
                jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            ),
            # Test zero matrix
            (
                jnp.array([[0, 0], [0, 0]]),
                jnp.array([[1, 2], [3, 4]]),
                jnp.array([[0, 0], [0, 0]]),
            ),
        ],
    )
    def test_known_results(self, A, B, expected):
        assert jnp.allclose(
            self.evaluate_matrix_multiplication(A, B), expected
        ), "Multiplication does not match expected result."

    def test_non_square_matrices(self):
        A = jnp.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        B = jnp.array([[1, 4], [2, 5], [3, 6]])  # 3x2 matrix
        expected = jnp.array([[14, 32], [32, 77]])  # Result of multiplication
        assert jnp.allclose(
            self.evaluate_matrix_multiplication(A, B), expected
        ), "Multiplication of non-square matrices does not match expected result."

    def test_multiplication_by_identity(self):
        A = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        eye = jnp.eye(3)
        assert jnp.allclose(
            self.evaluate_matrix_multiplication(A, eye), A
        ), "Multiplication by identity matrix failed."
        assert jnp.allclose(
            self.evaluate_matrix_multiplication(eye, A), A
        ), "Multiplication by identity matrix failed."

    def test_dimension_mismatch(self):
        A = jnp.array([[1, 2], [3, 4]])  # 2x2 matrix
        B = jnp.array([[1, 2, 3]])  # 1x3 matrix
        with pytest.raises(StaticError):
            self.evaluate_matrix_multiplication(A, B)


class TestMatrixTransposition:
    def evaluate_matrix_transpose(self, input):
        builder = collimator.DiagramBuilder()
        transpose_block = builder.add(library.MatrixTransposition())
        input_block = builder.add(library.Constant(input))
        builder.connect(input_block.output_ports[0], transpose_block.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        return transpose_block.output_ports[0].eval(ctx)

    @pytest.mark.parametrize(
        "A,expected",
        [
            # Test square matrix
            (jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 3], [2, 4]])),
            # Test non-square matrix
            (jnp.array([[1, 2, 3], [4, 5, 6]]), jnp.array([[1, 4], [2, 5], [3, 6]])),
            # Test row vector
            (jnp.array([[1, 2, 3]]), jnp.array([[1], [2], [3]])),
            # Test column vector
            (jnp.array([[1], [2], [3]]), jnp.array([[1, 2, 3]])),
            # Test empty matrix
            (jnp.array([]), jnp.array([])),
        ],
    )
    def test_known_transpose(self, A, expected):
        assert jnp.allclose(
            self.evaluate_matrix_transpose(A), expected
        ), "Transpose does not match expected result."

    def test_identity_matrix(self):
        eye = jnp.eye(3)
        assert jnp.allclose(
            self.evaluate_matrix_transpose(eye), eye
        ), "Transpose of an identity matrix should be itself."

    def test_double_transpose(self):
        A = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert jnp.allclose(
            self.evaluate_matrix_transpose(self.evaluate_matrix_transpose(A)), A
        ), "Double transpose should return to original matrix."


class TestSlice:
    @pytest.mark.parametrize(
        "input,slice_,sol",
        [
            (np.arange(10), "1:3", np.arange(10)[1:3]),
            (np.arange(9).reshape(3, 3), "2,2", np.arange(9).reshape(3, 3)[2, 2]),
            (
                np.arange(8).reshape(2, 2, 2),
                ":,0,0",
                np.arange(8).reshape(2, 2, 2)[:, 0, 0],
            ),
        ],
    )
    def test_ops(self, input, slice_, sol):
        builder = collimator.DiagramBuilder()
        inp = builder.add(library.Constant(input))
        sl = builder.add(library.Slice(slice_=slice_))
        builder.connect(inp.output_ports[0], sl.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        y = sl.output_ports[0].eval(context)

        assert np.allclose(y, sol)

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.Slice(slice_="3-5", name="Slice_0")

        # Success! The test failed as expected.
        print(e)
        assert (
            "Slice block Slice_0 detected invalid slice operator 3-5. [] are optional. Valid examples: '1:3,4', '[:,4:10]'"
            in str(e)
        )


class TestStack:
    def test_ops(self):
        builder = collimator.DiagramBuilder()
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        inp = builder.add(library.Constant(arr))
        st20 = builder.add(library.Stack(n_in=2, axis=0))
        st21 = builder.add(library.Stack(n_in=2, axis=1))
        st30 = builder.add(library.Stack(n_in=3, axis=0))
        builder.connect(inp.output_ports[0], st20.input_ports[0])
        builder.connect(inp.output_ports[0], st20.input_ports[1])

        builder.connect(inp.output_ports[0], st21.input_ports[0])
        builder.connect(inp.output_ports[0], st21.input_ports[1])

        builder.connect(inp.output_ports[0], st30.input_ports[0])
        builder.connect(inp.output_ports[0], st30.input_ports[1])
        builder.connect(inp.output_ports[0], st30.input_ports[2])

        diagram = builder.build()
        context = diagram.create_context()

        st20_ = st20.output_ports[0].eval(context)
        st21_ = st21.output_ports[0].eval(context)
        st30_ = st30.output_ports[0].eval(context)

        st20_sol = np.stack((arr, arr), axis=0)
        st21_sol = np.stack((arr, arr), axis=1)
        st30_sol = np.stack((arr, arr, arr), axis=0)

        assert np.allclose(st20_, st20_sol)
        assert np.allclose(st21_, st21_sol)
        assert np.allclose(st30_, st30_sol)


class TestSumOfElements:
    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_scalar_vector_inputs(self, dtype):
        sum_s = library.SumOfElements()
        sum_v = library.SumOfElements()

        Constant_0 = library.Constant(jnp.array([1, 2, 3], dtype=dtype), name="const1")
        Constant_1 = library.Constant(
            jnp.array([[1, 2, 3], [1, 2, 3]], dtype=dtype), name="const2"
        )

        builder = collimator.DiagramBuilder()
        builder.add(sum_s, sum_v, Constant_0, Constant_1)

        builder.connect(Constant_0.output_ports[0], sum_s.input_ports[0])
        builder.connect(Constant_1.output_ports[0], sum_v.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        y_s = sum_s.output_ports[0].eval(ctx)
        y_v = sum_v.output_ports[0].eval(ctx)

        assert np.allclose(y_s, 6)
        # assert y_s.dtype == dtype # jnp does some type promotion. let's not bother validating that.
        assert np.allclose(y_v, 12)


class TestSquareRoot:
    def _make_diagram(self, Constant_0, SquareRoot_0):
        builder = collimator.DiagramBuilder()

        builder.add(Constant_0, SquareRoot_0)
        builder.connect(Constant_0.output_ports[0], SquareRoot_0.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        sqrt_ctx = ctx[SquareRoot_0.system_id]
        return diagram, ctx, sqrt_ctx

    def test_floats(self):
        Constant_0 = library.Constant(4.0, name="Constant_0")
        SquareRoot_0 = library.SquareRoot(name="SquareRoot_0")

        diagram, ctx, sqrt_ctx = self._make_diagram(Constant_0, SquareRoot_0)
        diagram.pprint(logger.debug)

        y = SquareRoot_0.output_ports[0].eval(ctx)
        assert y == 2.0
        assert y.dtype == jnp.float64


class TestTrigonometric:
    @pytest.mark.parametrize(
        "fun_name,fun_sol",
        [
            ("sin", jnp.sin),
            ("cos", jnp.cos),
            ("tan", jnp.tan),
            ("asin", jnp.arcsin),
            ("acos", jnp.arccos),
            ("atan", jnp.arctan),
            ("sinh", jnp.sinh),
            ("cosh", jnp.cosh),
            ("tanh", jnp.tanh),
            ("asinh", jnp.arcsinh),
            ("acosh", jnp.arccosh),
            ("atanh", jnp.arctanh),
        ],
    )
    def test_ops(self, fun_name, fun_sol):
        builder = collimator.DiagramBuilder()
        one = builder.add(library.Constant(1.0))
        trig = builder.add(library.Trigonometric(function=fun_name))
        builder.connect(one.output_ports[0], trig.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        y = trig.output_ports[0].eval(context)

        assert np.allclose(y, fun_sol(1.0))

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.Trigonometric(function="sine")
        # Success! The test failed as expected.
        print(e)
        assert (
            "Valid options: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh"
            in str(e)
        )


if __name__ == "__main__":
    # TestAdder().test_invalid_input()
    # TestCrossProduct().test_cross_product()
    # TestDotProduct().test_dot_product()
    # TestExponent().test_floats()
    # TestExponent().test_invalid_input()
    # TestLogarithm().test_floats()
    # TestLogarithm().test_invalid_input()
    # TestOffset().test_floats()
    # TestPower().test_floats()
    operators = "**/"
    # operators = None
    inputs = [[1], [2], [3]]
    TestProduct().test_different_inputs(
        dtype=jnp.float64, inputs=inputs, operators=operators
    )
    inputs = [[1, 2, 3]] * 3
    TestProduct().test_different_inputs(
        dtype=jnp.float64, inputs=inputs, operators=operators
    )
    inputs = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]] * 3
    TestProduct().test_different_inputs(
        dtype=jnp.float64, inputs=inputs, operators=operators
    )
    # TestProduct().test_operators(dtype=jnp.float64)
    # TestProduct().test_invalid_input()
    # TestProductOfElements().test_scalar_vector_inputs(dtype=np.float64)
    # TestSumOfElements().test_scalar_vector_inputs(dtype=np.float64)
    # TestScalarBroadcast().test_invalid_input()
    # TestSquareRoot().test_floats()
    # TestTrigonometric().test_ops(fun_name="sin", fun_sol=jnp.sin)
    # TestTrigonometric().test_invalid_input()
    # TestSlice().test_ops(np.arange(10), "1:3", np.arange(10)[1:3])
    # TestSlice().test_invalid_input()
    # TestStack().test_ops()
