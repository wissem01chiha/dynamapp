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

import collimator
from collimator import DiagramBuilder, LeafSystem, simulate
from collimator.framework import parameters, Parameter
from collimator.library import Constant


class GainWithStaticParam(LeafSystem):
    @parameters(static=["p"])
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    def initialize(self, p):
        self.configure_output_port(
            self._output_port_idx,
            lambda time, state, u: p * u,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )


# FIXME: Make it work for jax too
@pytest.mark.parametrize("backend", ["numpy"])
def test_update_static_param(backend):
    collimator.set_backend(backend)
    p = Parameter(1.0)
    builder = DiagramBuilder()
    constant = builder.add(Constant(1.0))
    gain = builder.add(GainWithStaticParam(p))
    builder.connect(constant.output_ports[0], gain.input_ports[0])

    diagram = builder.build()
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "output": gain.output_ports[0],
        },
    )
    assert results.outputs["output"][0] == 1.0

    p.set(2.0)
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "output": gain.output_ports[0],
        },
    )
    assert results.outputs["output"][0] == 2.0
