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

import jax.numpy as jnp

import collimator
from collimator.library import (
    Integrator,
    Sine,
)

from collimator import logging


logging.set_file_handler("test.log")

pytestmark = pytest.mark.minimal


def test_double_integrator(dtype=jnp.float64):
    builder = collimator.DiagramBuilder()
    Sin_0 = builder.add(Sine(name="Sin_0"))

    x0 = dtype(0.0)
    v0 = dtype(-1.0)

    Integrator_0 = builder.add(Integrator(v0))  # v
    Integrator_1 = builder.add(Integrator(x0))  # x

    builder.connect(Sin_0.output_ports[0], Integrator_0.input_ports[0])
    builder.connect(Integrator_0.output_ports[0], Integrator_1.input_ports[0])

    diagram = builder.build()
    ctx = diagram.create_context()

    print([(p.name, p.system) for p in Sin_0.output_ports])
    print([(p.name, p.system) for p in Integrator_0.input_ports])
    print([(p.name, p.system) for p in Integrator_0.output_ports])
    print([(p.name, p.system) for p in Integrator_1.input_ports])

    t = jnp.linspace(0.0, 10.0, 100, dtype=dtype)
    options = collimator.SimulatorOptions(atol=1e-8, rtol=1e-6)
    recorded_signals = {
        "x": Integrator_1.output_ports[0],
        "v": Integrator_0.output_ports[0],
    }
    sol = collimator.simulate(
        diagram,
        ctx,
        (t[0], t[-1]),
        options=options,
        recorded_signals=recorded_signals,
    )
    x, v = sol.outputs["x"], sol.outputs["v"]
    t = sol.time

    print(x)
    print(v)
    print(jnp.sin(t))
    print(jnp.cos(t))
    print(jnp.std(x - jnp.sin(t)))
    print(jnp.std(x + jnp.sin(t)))
    assert jnp.allclose(x, -jnp.sin(t), rtol=1e-4, atol=1e-6)
    assert jnp.allclose(v, -jnp.cos(t), rtol=1e-4, atol=1e-6)

    assert x.dtype == dtype
    assert v.dtype == dtype
