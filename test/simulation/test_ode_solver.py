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
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import solve_ivp
import collimator
from collimator.models import EulerRigidBody, ArenstorfOrbit, Lorenz, Pleiades
from collimator.backend import ODESolver, numpy_api as cnp

pytestmark = pytest.mark.minimal

ODE_SOLVERS = ["RK45", "BDF"]


class TestHairerSystems:
    def _run_ode_test(self, system, t_span, method, rtol, atol):
        cnp.set_backend("jax")
        context = system.create_context()
        recorded_signals = {"x": system.output_ports[0]}
        options = collimator.SimulatorOptions(
            rtol=rtol,
            atol=atol,
            ode_solver_method=method,
        )
        results = collimator.simulate(
            system,
            context,
            t_span,
            recorded_signals=recorded_signals,
            options=options,
        )
        x = results.outputs["x"]
        t = results.time

        # Compare with the SciPy solution
        # Since we can't directly match time stamps using the `simulate` interface,
        # instead call the SciPy solve with results interpolated at the specified points

        cnp.set_backend("numpy")
        context = system.create_context()
        scipy_solver = ODESolver(system)
        scipy_solver.initialize(context)

        def f(t, y):
            return scipy_solver.flat_ode_rhs(y, t, context)

        xc0 = scipy_solver._ravel(context.continuous_state)
        scipy_sol = solve_ivp(
            f, t_span, xc0, atol=atol, rtol=rtol, method=method, t_eval=t
        )
        x_scipy = scipy_sol.y.T

        assert np.allclose(x_scipy, x, rtol=1e-4, atol=1e-4)
        return t, x, x_scipy

    @pytest.mark.parametrize("method", ODE_SOLVERS)
    def test_euler(self, method, show_plot=False):
        # Euler's equation of rotation for a rigid body
        system = EulerRigidBody()
        t_span = (0.0, 20.0)
        t, x, x_scipy = self._run_ode_test(system, t_span, method, rtol=1e-8, atol=1e-6)

        if show_plot:
            fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
            axs[0].plot(t, x_scipy, ".-")
            axs[0].set_title("Scipy")
            axs[1].plot(t, x, ".-")
            axs[1].set_title("Collimator")
            plt.show()

    @pytest.mark.parametrize("method", ODE_SOLVERS)
    def test_arenstorf(self, method, show_plot=False):
        # Restricted three-body problem
        system = ArenstorfOrbit()
        t_span = (0.0, 17.0652165601579625588917206249)

        t, x, x_scipy = self._run_ode_test(
            system, t_span, method, rtol=1e-10, atol=1e-12
        )

        if show_plot:
            fig, axs = plt.subplots(figsize=(4, 4), sharex=True)
            axs.plot(x_scipy[:, 0], x_scipy[:, 1], "-", label="Scipy")
            axs.plot(x[:, 0], x[:, 1], "--", label="Collimator")
            plt.show()

    @pytest.mark.parametrize("method", ODE_SOLVERS)
    def test_lorenz(self, method, show_plot=False):
        system = Lorenz()

        # Very chaotic - only compare over short times
        t_span = (0.0, 1.0)
        t, x, x_scipy = self._run_ode_test(
            system, t_span, method, rtol=1e-12, atol=1e-14
        )

        if show_plot:
            fig, axs = plt.subplots(3, 1, figsize=(7, 4), sharex=True)
            for i in range(3):
                axs[i].plot(t, x_scipy[:, i], label="scipy")
                axs[i].plot(t, x[:, i], "--", label="collimator")
            plt.show()

    @pytest.mark.skip(reason="Too slow to be useful as a CI test. Use for dev only")
    def test_pleiades(self, show_plot=False):
        t_span = (0.0, 3.0)
        system = Pleiades()
        t, x, x_scipy = self._run_ode_test(
            system, t_span, "rk45", rtol=1e-8, atol=1e-10
        )

        if show_plot:
            fig, axs = plt.subplots(figsize=(4, 4), sharex=True)
            for i in range(7):
                axs.plot(x_scipy[:, i], x_scipy[:, i + 7], "-", label="Scipy")
                axs.plot(x[:, i], x[:, i + 7], "--", label="Collimator")
            plt.show()


if __name__ == "__main__":
    TestHairerSystems().test_euler("rk45", show_plot=True)
    # TestHairerSystems().test_arenstorf(show_plot=True)
    # TestHairerSystems().test_lorenz(show_plot=True)
    # TestHairerSystems().test_pleiades(show_plot=True)
