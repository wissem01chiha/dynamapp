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

import pytest

import numpy as np

import collimator
from collimator.library import Constant, Integrator, Stop, Comparator
from collimator.simulation import SimulatorOptions

# from collimator import logging


# logging.set_log_handlers(to_file="test.log")

pytestmark = pytest.mark.minimal


def test_stop_with_integrator():
    sim_stop_time = 1.0
    expected_stop_time = 0.123456

    builder = collimator.DiagramBuilder()
    c = builder.add(Constant(value=1.0))
    i = builder.add(Integrator(initial_state=0.0))
    cmp = builder.add(Comparator(operator=">="))
    thr = builder.add(Constant(value=expected_stop_time))
    s = builder.add(Stop())
    builder.connect(c.output_ports[0], i.input_ports[0])
    builder.connect(i.output_ports[0], cmp.input_ports[0])
    builder.connect(thr.output_ports[0], cmp.input_ports[1])
    builder.connect(cmp.output_ports[0], s.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded_signals = {"int": i.output_ports[0], "cmp": cmp.output_ports[0]}

    dt = 0.1  # needed to catch zero_crossing events
    options = SimulatorOptions(
        max_major_step_length=dt,
        atol=1e-8,
        rtol=1e-6,
    )
    results = collimator.simulate(
        diagram,
        context,
        (0.0, sim_stop_time),
        recorded_signals=recorded_signals,
        options=options,
    )

    print(results.time)

    assert np.allclose(results.context.time, expected_stop_time)


if __name__ == "__main__":
    test_stop_with_integrator()
