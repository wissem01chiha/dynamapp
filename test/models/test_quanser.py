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

import numpy as np

import collimator
from collimator.library.quanser import QubeServoModel


def test_qube_simulation(show_plots=False):
    system = QubeServoModel(x0=[0.0, 0.5, 0.0, 0.0])
    system.input_ports[0].fix_value(0.0)
    context = system.create_context()

    recorded_signals = {"y": system.output_ports[0]}
    results = collimator.simulate(
        system,
        context,
        (0.0, 5.0),
        recorded_signals=recorded_signals,
    )

    if show_plots:
        import matplotlib.pyplot as plt

        t = results.time
        y = results.outputs["y"]

        plt.figure(figsize=(7, 2))
        plt.plot(t, y)
        plt.xlabel("Time (s)")
        plt.ylabel("State")
        plt.title("Qube Simulation")
        plt.show()


def test_qube_linearization():
    xd = np.array([0.0, 0.0, 0.0, 0.0])
    xu = np.array([0.0, np.pi, 0.0, 0.0])
    u0 = np.array([0.0])

    # Check that the "down" fixed point is stable
    system = QubeServoModel()
    system.input_ports[0].fix_value(u0)
    context = system.create_context()

    context = context.with_continuous_state(xd)
    xdot = system.eval_time_derivatives(context)
    assert np.allclose(xdot, 0.0)

    lin_sys = collimator.library.linearize(system, context)
    evals = np.linalg.eigvals(lin_sys.A)

    # One zero eigenvalue for the rotor degree of freedom
    assert sum(e == 0 for e in evals) == 1

    # Three stable modes
    assert sum(e.real < 0 for e in evals) == 3

    # Check that the "up" fixed point is unstable
    context = context.with_continuous_state(xu)
    xdot = system.eval_time_derivatives(context)
    assert np.allclose(xdot, 0.0)

    lin_sys = collimator.library.linearize(system, context)
    evals = np.linalg.eigvals(lin_sys.A)

    # Expect one positive eigenvalue for the unstable "falling" mode
    assert sum(e.real > 0 for e in evals) == 1

    # One zero eigenvalue for the rotor degree of freedom
    assert sum(e == 0 for e in evals) == 1

    # Two stable modes
    assert sum(e.real < 0 for e in evals) == 2


if __name__ == "__main__":
    test_qube_simulation(show_plots=True)
