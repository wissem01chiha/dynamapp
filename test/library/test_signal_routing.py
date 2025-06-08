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

"""Test blocks that combine and separate signals.

Contains tests for:
- IfThenElse
- Multiplexer
- Demultiplexer

In the future, this is also where buses should be tested.
"""

import pytest
import numpy as np

import collimator
from collimator import library

pytest.mark.minimal


float_dtypes = [
    np.float64,
    np.float32,
    np.float16,
]

int_dtypes = [
    np.int64,
    np.int32,
    np.int16,
]


class TestIfTheElse:
    def test_if_then_else(self):
        step = library.Step(start_value=False, end_value=True)
        ud = library.UnitDelay(dt=0.1, initial_state=0.0)
        ite = library.IfThenElse()
        one = library.Constant(1.0)
        zero = library.Constant(0.0)

        ite_vec = library.IfThenElse()
        ones = library.Constant(np.ones(3))
        zeros = library.Constant(np.zeros(3))

        builder = collimator.DiagramBuilder()
        builder.add(step, ud, ite, one, zero, ite_vec, ones, zeros)

        builder.connect(one.output_ports[0], ud.input_ports[0])

        builder.connect(step.output_ports[0], ite.input_ports[0])
        builder.connect(one.output_ports[0], ite.input_ports[1])
        builder.connect(zero.output_ports[0], ite.input_ports[2])

        builder.connect(step.output_ports[0], ite_vec.input_ports[0])
        builder.connect(ones.output_ports[0], ite_vec.input_ports[1])
        builder.connect(zeros.output_ports[0], ite_vec.input_ports[2])

        diagram = builder.build()
        context = diagram.create_context()

        recorded_signals = {
            "ite": ite.output_ports[0],
            "ite_vec": ite_vec.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        ts, ite_, ite_vec_ = (
            results.time,
            results.outputs["ite"],
            results.outputs["ite_vec"],
        )

        ite_sol = np.zeros_like(ts)
        ite_sol[10:] = 1.0

        print(ite_)
        print(ite_sol)

        assert np.allclose(ite_sol, ite_)
        assert np.allclose(ite_sol, ite_vec_[:, 0])


class TestMux:
    def test_mixed_inputs(self):
        Mux_0 = library.Multiplexer(3, name="mux3")
        assert len(Mux_0.input_ports) == 3
        assert len(Mux_0.output_ports) == 1

        Constant_0 = library.Constant(np.array([0.5]), name="const1")
        Constant_1 = library.Constant(1.5, name="const2")
        Constant_2 = library.Constant([2.5, 3.5], name="const3")

        builder = collimator.DiagramBuilder()
        builder.add(Mux_0, Constant_0, Constant_1, Constant_2)

        builder.connect(Constant_0.output_ports[0], Mux_0.input_ports[0])
        builder.connect(Constant_1.output_ports[0], Mux_0.input_ports[1])
        builder.connect(Constant_2.output_ports[0], Mux_0.input_ports[2])

        diagram = builder.build()

        ctx = diagram.create_context()

        diagram.pprint()

        y = Mux_0.output_ports[0].eval(ctx)
        print(f"{y=}")

        assert np.allclose(y, np.array([0.5, 1.5, 2.5, 3.5]))

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_scalar_inputs(self, dtype):
        Mux_0 = library.Multiplexer(3, name="mux3")
        assert len(Mux_0.input_ports) == 3
        assert len(Mux_0.output_ports) == 1

        Constant_0 = library.Constant(np.array([1], dtype=dtype), name="const1")
        Constant_1 = library.Constant(np.array([2], dtype=dtype), name="const2")
        Constant_2 = library.Constant(np.array([3, 4], dtype=dtype), name="const3")

        builder = collimator.DiagramBuilder()
        builder.add(Mux_0, Constant_0, Constant_1, Constant_2)

        builder.connect(Constant_0.output_ports[0], Mux_0.input_ports[0])
        builder.connect(Constant_1.output_ports[0], Mux_0.input_ports[1])
        builder.connect(Constant_2.output_ports[0], Mux_0.input_ports[2])

        diagram = builder.build()

        ctx = diagram.create_context()

        diagram.pprint()

        y = Mux_0.output_ports[0].eval(ctx)

        assert np.allclose(y, np.array([1, 2, 3, 4]))
        assert y.dtype == dtype


class TestDemux:
    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_basic(self, dtype):
        Demux_0 = library.Demultiplexer(3)
        assert len(Demux_0.input_ports) == 1
        assert len(Demux_0.output_ports) == 3

        u = np.array([1, 2, 3], dtype=dtype)
        Constant_0 = library.Constant(u)

        builder = collimator.DiagramBuilder()
        builder.add(Demux_0, Constant_0)
        builder.connect(Constant_0.output_ports[0], Demux_0.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        y = []
        for port in Demux_0.output_ports:
            y_i = port.eval(ctx)
            y.append(y_i)

        for i in range(3):
            assert np.allclose(y[i], u[i])
            assert y[i].dtype == dtype


if __name__ == "__main__":
    TestIfTheElse().test_if_then_else()
