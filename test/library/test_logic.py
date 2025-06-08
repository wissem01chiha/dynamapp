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

"""Test blocks that apply logical and bit operations to the signal.

Contains tests for:
- Logical Operator
- Logical Reducer
"""

import pytest
import numpy as np
import collimator
from collimator import library
from collimator.framework.error import BlockParameterError

pytestmark = pytest.mark.minimal


class TestLogicalOperator:
    def test_scalar(self):
        builder = collimator.DiagramBuilder()

        true_ = builder.add(library.Constant(name="true", value=True))
        false_ = builder.add(library.Constant(name="false", value=False))
        or_ = builder.add(library.LogicalOperator(function="or", name="or"))
        and_ = builder.add(library.LogicalOperator(function="and", name="and"))
        not_ = builder.add(library.LogicalOperator(function="not", name="not"))
        xor_ = builder.add(library.LogicalOperator(function="xor", name="xor"))
        nor_ = builder.add(library.LogicalOperator(function="nor", name="nor"))
        nand_ = builder.add(library.LogicalOperator(function="nand", name="nand"))

        builder.connect(true_.output_ports[0], or_.input_ports[0])
        builder.connect(false_.output_ports[0], or_.input_ports[1])

        builder.connect(true_.output_ports[0], and_.input_ports[0])
        builder.connect(false_.output_ports[0], and_.input_ports[1])

        builder.connect(true_.output_ports[0], not_.input_ports[0])

        builder.connect(true_.output_ports[0], xor_.input_ports[0])
        builder.connect(false_.output_ports[0], xor_.input_ports[1])

        builder.connect(true_.output_ports[0], nor_.input_ports[0])
        builder.connect(false_.output_ports[0], nor_.input_ports[1])

        builder.connect(true_.output_ports[0], nand_.input_ports[0])
        builder.connect(false_.output_ports[0], nand_.input_ports[1])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "or_": or_.output_ports[0],
            "and_": and_.output_ports[0],
            "not_": not_.output_ports[0],
            "xor_": xor_.output_ports[0],
            "nor_": nor_.output_ports[0],
            "nand_": nand_.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 0.1), recorded_signals=recorded_signals
        )

        assert results.outputs["or_"][0]
        assert not results.outputs["and_"][0]
        assert not results.outputs["not_"][0]
        assert results.outputs["xor_"][0]
        assert not results.outputs["nor_"][0]
        assert results.outputs["nand_"][0]

    # @pytest.mark.xfail(reason="haven't resolved WC-227")
    def test_array(self):
        builder = collimator.DiagramBuilder()

        arr_true = np.array([True, True, True])
        arr_false = np.array([False, False, False])
        trues = builder.add(library.Constant(name="true", value=arr_true))
        falses = builder.add(library.Constant(name="false", value=arr_false))

        or_ = builder.add(library.LogicalOperator(function="or", name="or"))
        and_ = builder.add(library.LogicalOperator(function="and", name="and"))
        not_ = builder.add(library.LogicalOperator(function="not", name="not"))
        xor_ = builder.add(library.LogicalOperator(function="xor", name="xor"))
        nor_ = builder.add(library.LogicalOperator(function="nor", name="nor"))
        nand_ = builder.add(library.LogicalOperator(function="nand", name="nand"))

        builder.connect(trues.output_ports[0], or_.input_ports[0])
        builder.connect(falses.output_ports[0], or_.input_ports[1])

        builder.connect(trues.output_ports[0], and_.input_ports[0])
        builder.connect(falses.output_ports[0], and_.input_ports[1])

        builder.connect(trues.output_ports[0], not_.input_ports[0])

        builder.connect(trues.output_ports[0], xor_.input_ports[0])
        builder.connect(falses.output_ports[0], xor_.input_ports[1])

        builder.connect(trues.output_ports[0], nor_.input_ports[0])
        builder.connect(falses.output_ports[0], nor_.input_ports[1])

        builder.connect(trues.output_ports[0], nand_.input_ports[0])
        builder.connect(falses.output_ports[0], nand_.input_ports[1])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "or_": or_.output_ports[0],
            "and_": and_.output_ports[0],
            "not_": not_.output_ports[0],
            "xor_": xor_.output_ports[0],
            "nor_": nor_.output_ports[0],
            "nand_": nand_.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 0.1), recorded_signals=recorded_signals
        )

        or_sol = np.logical_or(arr_true, arr_false)
        and_sol = np.logical_and(arr_true, arr_false)
        not_sol = np.logical_not(arr_true)
        xor_sol = np.logical_xor(arr_true, arr_false)
        nor_sol = np.logical_not(np.logical_or(arr_true, arr_false))
        nand_sol = np.logical_not(np.logical_and(arr_true, arr_false))

        print(results.time)
        print(results.outputs["or_"][0])
        print(or_sol)

        assert np.all(results.outputs["or_"][0] == or_sol)
        assert np.all(results.outputs["and_"][0] == and_sol)
        assert np.all(results.outputs["not_"][0] == not_sol)
        assert np.all(results.outputs["xor_"][0] == xor_sol)
        assert np.all(results.outputs["nor_"][0] == nor_sol)
        assert np.all(results.outputs["nand_"][0] == nand_sol)

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.LogicalOperator(function="some", name="LogicalOperator")
        print(e)
        # Success! The test failed as expected.
        assert (
            "LogicalOperator block LogicalOperator has invalid selection some for 'function'. Valid options: or, and, not, xor, nor, nand"
            in str(e)
        )


class TestLogicalReducer:
    def test_ops(self):
        builder = collimator.DiagramBuilder()

        true_ = builder.add(library.Constant(name="true", value=True))
        arr_true = np.array([True, True, True])
        trues = builder.add(library.Constant(name="trues", value=arr_true))

        any_s = builder.add(library.LogicalReduce(function="any", name="any_s"))
        all_s = builder.add(library.LogicalReduce(function="all", name="all_s"))
        any_arr = builder.add(library.LogicalReduce(function="any", name="any_arr"))
        all_arr = builder.add(library.LogicalReduce(function="all", name="all_arr"))

        builder.connect(true_.output_ports[0], any_s.input_ports[0])
        builder.connect(trues.output_ports[0], any_arr.input_ports[0])

        builder.connect(true_.output_ports[0], all_s.input_ports[0])
        builder.connect(trues.output_ports[0], all_arr.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "any_s": any_s.output_ports[0],
            "all_s": all_s.output_ports[0],
            "any_arr": any_arr.output_ports[0],
            "all_arr": all_arr.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 0.1), recorded_signals=recorded_signals
        )

        assert results.outputs["any_s"][0]
        assert results.outputs["all_s"][0]

        any_arr_sol = np.any(arr_true)
        all_arr_sol = np.all(arr_true)

        assert np.all(results.outputs["any_arr"][0] == any_arr_sol)
        assert np.all(results.outputs["all_arr"][0] == all_arr_sol)

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.LogicalReduce(function="some", name="LogicalReduce")
        print(e)
        # Success! The test failed as expected.
        assert (
            "LogicalReduce block LogicalReduce has invalid selection some for 'function'. Valid options: any, all"
            in str(e)
        )


if __name__ == "__main__":
    TestLogicalOperator().test_scalar()
    TestLogicalOperator().test_array()
    TestLogicalOperator().test_invalid_input()

    TestLogicalReducer().test_ops()
    TestLogicalReducer().test_invalid_input()
