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

"""Test that numerical backend settings result in the correct active backend."""

import pytest

import numpy as np

import collimator
from collimator import library
from collimator.backend import numpy_api as cnp

pytestmark = pytest.mark.minimal


def _psb_ode_diagram(use_jax):
    """A simple Clock -> PythonScript -> Integrator system"""
    PythonScript = library.CustomJaxBlock if use_jax else library.CustomPythonBlock

    builder = collimator.DiagramBuilder()
    clock = builder.add(library.Clock(name="Clock_0"))
    psb = builder.add(
        PythonScript(
            user_statements="out_0 = in_0",
            inputs=["in_0"],
            outputs=["out_0"],
            time_mode="agnostic",
            name="PythonScript_0",
        )
    )
    integrator = builder.add(library.Integrator(0.0, name="Integrator_0"))
    builder.connect(clock.output_ports[0], psb.input_ports[0])
    builder.connect(psb.output_ports[0], integrator.input_ports[0])
    return builder.build()


@pytest.mark.parametrize("use_jax", [True, False])
def test_set_jax_global_backend(use_jax):
    # Check that setting the backend ahead of time works as expected.
    cnp.set_backend("jax" if use_jax else "numpy")
    system = _psb_ode_diagram(use_jax=use_jax)
    context = system.create_context()
    recorded_signals = {"x": system["Integrator_0"].output_ports[0]}
    tf = 1.0
    results = collimator.simulate(
        system,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
    )
    assert results.time[-1] == tf
    assert np.allclose(results.outputs["x"], 0.5 * results.time**2)

    expected_backend = "jax" if use_jax else "numpy"
    assert cnp.active_backend == expected_backend


@pytest.mark.parametrize("use_jax", [True, False])
def test_set_np_global_backend(use_jax):
    # Note that due to array conversions in CustomJaxBlock this will not
    # work if use_jax=True.
    cnp.set_backend("numpy")
    system = _psb_ode_diagram(use_jax=use_jax)
    context = system.create_context()
    recorded_signals = {"x": system["Integrator_0"].output_ports[0]}
    tf = 1.0
    results = collimator.simulate(
        system,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
    )
    assert results.time[-1] == tf
    assert np.allclose(results.outputs["x"], 0.5 * results.time**2)
    assert cnp.active_backend == "numpy"


@pytest.mark.parametrize("use_jax", [True, False])
def test_set_jax_backend_options(use_jax):
    # Check that setting the backend via options works as expected.
    # First set the opposite backend globally.
    cnp.set_backend("numpy" if use_jax else "jax")
    expected_backend = "jax" if use_jax else "numpy"
    system = _psb_ode_diagram(use_jax=use_jax)
    context = system.create_context()
    recorded_signals = {"x": system["Integrator_0"].output_ports[0]}
    options = collimator.SimulatorOptions(math_backend=expected_backend)
    tf = 1.0
    results = collimator.simulate(
        system,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
        options=options,
    )
    assert results.time[-1] == tf
    assert np.allclose(results.outputs["x"], 0.5 * results.time**2)
    assert cnp.active_backend == expected_backend


def test_set_np_backend_options_untraced():
    # Test setting the numpy backend via options.
    # First set the opposite backend globally.
    cnp.set_backend("jax")
    system = _psb_ode_diagram(use_jax=False)
    context = system.create_context()
    recorded_signals = {"x": system["Integrator_0"].output_ports[0]}
    options = collimator.SimulatorOptions(math_backend="numpy")
    tf = 1.0
    results = collimator.simulate(
        system,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
        options=options,
    )
    assert results.time[-1] == tf
    assert np.allclose(results.outputs["x"], 0.5 * results.time**2)
    assert cnp.active_backend == "numpy"
