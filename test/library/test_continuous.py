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

"""Test continuous-time systems

Contains tests for:
- BatteryCell
- Derivative
- Integrator
- PID
- LTISystem
"""

import pytest
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import control
from jax.scipy.linalg import expm
from collimator.framework.error import StaticError
import collimator
from collimator import library
from collimator.backend import numpy_api as cnp
from collimator.framework import Parameter


float_dtypes = [
    cnp.float64,
    cnp.float32,
    cnp.float16,
]

int_dtypes = [
    cnp.int64,
    cnp.int32,
    cnp.int16,
]


# Simplified discrete-time implementation of the battery model.
class bm_lion:
    def __init__(
        self, E0=3.366, K=0.0076, Q=2.3, R=0.01, tau=30.0, A=0.26422, B=26.5487
    ):
        self.E0 = E0
        self.K = K
        self.Q = Q
        self.R = R
        self.tau = tau
        self.A = A
        self.B = B
        self.it = 1e-4  # 0 means fully charged, so start very close to fully charged
        self.istar = 0.0
        self.istar_st = 0.0

    def v_polarization(self):
        return self.K * (self.Q / (self.Q - self.it)) * self.it

    def v_exponential(self):
        return self.A * np.exp(-1 * self.B * self.it)

    def v_sat(self, v):
        # upper limit is not active because lower limiting 'it' to 0 ensures the upper limit is respected.
        # lower limit is needed otherwise the voltage will go negative for extreme discharge scenarios.
        return np.minimum(2 * self.E0, np.maximum(v, 0))

    def it_sat(self):
        # saturate the integrand 'it' to between its reasonable physical limits
        # to prevent unreasonable voltage values resulting from 'it' being outside it physical limits
        # llim = 1e-4
        llim = 0
        it_ulim = self.Q * (1 - llim)
        self.it = np.minimum(it_ulim, np.maximum(self.it, llim))

    def v_dcharge(self, i):
        v_pol = self.v_polarization()
        v_exp = self.v_exponential()
        Vbatt = (
            self.E0
            - self.R * i
            - self.K * (self.Q / (self.Q - self.it)) * self.istar
            - v_pol
            + v_exp
        )
        # Vbatt = self.v_sat(Vbatt)
        return Vbatt

    def v_charge(self, i):
        v_pol = self.v_polarization()
        v_exp = self.v_exponential()
        Vbatt = (
            self.E0
            - self.R * i
            - self.K * (self.Q / (self.it + 0.1 * self.Q)) * self.istar
            - v_pol
            + v_exp
        )
        # Vbatt = self.v_sat(Vbatt)
        return Vbatt

    def update(self, dt, i):
        # alpha = dt/(self.tau + dt)
        # self.istar = self.istar*(1-alpha) + i*alpha # discrete time first order filter
        self.istar_st = self.istar_st + dt * (-0.05 * self.istar_st + i)
        self.istar = 0.05 * self.istar_st
        self.it = self.it + i * dt / 3600  # convert A*s to A*h
        self.it_sat()
        if self.istar >= 0:
            return self.v_dcharge(i)
        else:
            return self.v_charge(i)


class CurrentDraw(library.SourceBlock):
    def __init__(self, **kwargs):
        super().__init__(self._func, **kwargs)

    def _func(self, t):
        current_sign = jnp.where(t >= 20, 1, 0)
        current_sign = jnp.where(t >= 200, 0, current_sign)
        current_sign = jnp.where(t >= 220, -1, current_sign)
        current_sign = jnp.where(t >= 400, 0, current_sign)
        return current_sign * 10


@pytest.mark.minimal
class TestBatteryCell:
    def test_battery_cell(self, plot=False):
        # collimator.set_backend("numpy")

        builder = collimator.DiagramBuilder()
        current = builder.add(CurrentDraw())
        battery_cell = builder.add(library.BatteryCell())
        builder.connect(current.output_ports[0], battery_cell.input_ports[0])

        system = builder.build()
        context = system.create_context()

        # define test parameters
        tf = 500.0

        recorded_signals = {
            "current": current.output_ports[0],
            "voltage": battery_cell.output_ports[0],
            "soc": battery_cell.output_ports[1],
        }
        results = collimator.simulate(
            system,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        # generate the alternative solution
        time = results.time
        b_volts_sim = results.outputs["voltage"]
        dt_v = np.diff(time)
        current_gain = 10

        b = bm_lion()
        b_volts_sol = np.zeros(np.shape(time))
        b_it = np.zeros(np.shape(time))
        b_A = np.zeros(np.shape(time))
        b_istar = np.zeros(np.shape(time))
        for idx in range(len(time) - 1):
            t = time[idx]
            dt = dt_v[idx]
            current_sign = 0
            if t >= 20:
                current_sign = 1
            if t >= 200:
                current_sign = 0
            if t >= 220:
                current_sign = -1
            if t >= 400:
                current_sign = 0

            b_A[idx] = current_gain * current_sign
            b_volts_sol[idx] = b.update(dt, b_A[idx])
            b_it[idx] = b.it
            b_istar[idx] = b.istar

        b_volts_sol[-1] = b_volts_sim[-1]

        if plot:
            plt.figure()
            plt.plot(time, b_volts_sol, label="Discrete-time model")
            plt.plot(time, b_volts_sim, label="Collimator model")
            plt.show()

        # CMLC tested mean error with a limit of 0.004
        err = abs(b_volts_sol - b_volts_sim).mean()
        assert err < 0.004
        assert time[-1] == tf


@pytest.mark.minimal
class TestIntegrator:
    def test_port_eval_scalar(self):
        x0 = 0.5
        integrator = library.Integrator(x0)
        integrator.input_ports[0].fix_value(0.0)

        ctx = integrator.create_context()

        x_eval = integrator.output_ports[0].eval(ctx)  # x0
        assert x_eval == x0

        # Integrator converts things to jax arrays internally
        assert isinstance(x_eval, cnp.ndarray)
        assert x_eval.shape == ()
        assert x_eval.dtype == cnp.float64

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_port_eval_array(self, dtype):
        x0 = np.array([1, 2], dtype=dtype)
        integrator = library.Integrator(x0)
        integrator.input_ports[0].fix_value(0 * x0)
        ctx = integrator.create_context()

        x_eval = integrator.output_ports[0].eval(ctx)  # x0
        assert np.allclose(x_eval, x0)
        assert isinstance(x_eval, cnp.ndarray)
        assert x_eval.shape == x0.shape
        assert x_eval.dtype == dtype

    def _make_sin_integrator(self, Integrator_0):
        Sin_0 = library.Sine(name="Sin_0")
        builder = collimator.DiagramBuilder()
        builder.add(Integrator_0, Sin_0)
        builder.connect(Sin_0.output_ports[0], Integrator_0.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        return diagram, ctx

    def test_time_derivatives_scalar(self):
        x0 = 0.5
        Integrator_0 = library.Integrator(x0, name="Integrator_0")
        diagram, ctx = self._make_sin_integrator(Integrator_0)

        xdot = Integrator_0.eval_time_derivatives(ctx)
        assert xdot == 0.0

        # Check at a different time
        ctx = ctx.with_time(1.0)
        xdot = Integrator_0.eval_time_derivatives(ctx)
        assert xdot == np.sin(1.0)

    def test_integrate_sin(self):
        x0 = -1.0
        Integrator_0 = library.Integrator(x0)
        diagram, ctx = self._make_sin_integrator(Integrator_0)

        t0, tf = 0.0, 10.0
        results = collimator.simulate(
            diagram,
            ctx,
            (t0, tf),
        )
        xf = results.context[
            Integrator_0.system_id
        ].continuous_state  # Second state component is empty

        assert np.allclose(xf, -np.cos(tf), rtol=1e-6, atol=1e-8)

    def _make_sawtooth(self):
        x_reset = 0.5

        builder = collimator.DiagramBuilder()
        integrator = builder.add(
            library.Integrator(0.0, enable_reset=True, name="integrator")
        )
        constant = builder.add(library.Constant(1.0, name="constant"))
        comparator = builder.add(
            library.FeedthroughBlock(lambda x: x > x_reset, name="comparator")
        )

        builder.connect(constant.output_ports[0], integrator.input_ports[0])
        builder.connect(integrator.output_ports[0], comparator.input_ports[0])
        builder.connect(comparator.output_ports[0], integrator.input_ports[1])

        return builder.build()

    def test_single_reset(self):
        diagram = self._make_sawtooth()
        ctx = diagram.create_context()

        tf = 0.75
        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        results = collimator.simulate(diagram, ctx, (0.0, tf), options=options)
        ctx = results.context
        integrator = diagram["integrator"]
        assert np.allclose(ctx[integrator.system_id].continuous_state, 0.25)

    def test_multiple_resets(self):
        diagram = self._make_sawtooth()
        ctx = diagram.create_context()

        tf = 2.75
        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        results = collimator.simulate(diagram, ctx, (0.0, tf), options=options)
        ctx = results.context
        integrator = diagram["integrator"]
        assert np.allclose(ctx[integrator.system_id].continuous_state, 0.25, atol=1e-4)

    x_reset = 1.5

    def _make_sin_sawtooth(self):
        builder = collimator.DiagramBuilder()
        integrator = builder.add(
            library.Integrator(
                0.0,
                name="integrator",
                enable_reset=True,
                enable_external_reset=True,
            )
        )
        constant = builder.add(library.Constant(1.0, name="constant"))
        sin = builder.add(library.Sine(name="sin"))
        comparator = builder.add(
            library.FeedthroughBlock(lambda x: x > self.x_reset, name="comparator")
        )

        builder.connect(constant.output_ports[0], integrator.input_ports[0])

        builder.connect(integrator.output_ports[0], comparator.input_ports[0])
        builder.connect(
            comparator.output_ports[0], integrator.input_ports[1]
        )  # Reset trigger
        builder.connect(sin.output_ports[0], integrator.input_ports[2])  # Reset value

        return builder.build()

    def test_single_reset_external(self):
        diagram = self._make_sin_sawtooth()
        integrator = diagram["integrator"]

        ctx = diagram.create_context()

        tf = 2.0
        t1 = self.x_reset
        x1 = np.sin(t1)  # Should reset to this value at t1
        xf = x1 + (tf - t1)

        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        results = collimator.simulate(diagram, ctx, (0.0, tf), options=options)
        ctx = results.context
        assert np.allclose(ctx[integrator.system_id].continuous_state, xf, atol=1e-3)

    def test_multiple_resets_external(self):
        diagram = self._make_sin_sawtooth()
        ctx = diagram.create_context()

        tf = 2.5

        t1 = self.x_reset
        x1 = np.sin(t1)  # Should reset to this value at t1

        t2 = t1 + (self.x_reset - x1)  # Second time x -> x_reset
        x2 = np.sin(t2)  # Should reset to this value at t2

        xf = x2 + (tf - t2)
        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        results = collimator.simulate(diagram, ctx, (0.0, tf), options=options)
        ctx = results.context
        integrator = diagram["integrator"]
        assert np.allclose(ctx[integrator.system_id].continuous_state, xf, atol=1e-3)

    def _make_limits_test_model(self, const=1.0, lower_limit=None, upper_limit=None):
        builder = collimator.DiagramBuilder()
        integrator = builder.add(
            library.Integrator(
                0.0,
                name="integrator",
                enable_limits=True,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
        )
        constant = builder.add(library.Constant(const, name="constant"))
        builder.connect(constant.output_ports[0], integrator.input_ports[0])
        return builder.build()

    def test_upper_limit(self, upper_limit=2.0, show_plot=False):
        diagram = self._make_limits_test_model(upper_limit=upper_limit)
        int_ = diagram["integrator"]
        ctx = diagram.create_context()

        tf = upper_limit * 2.0
        recorded_signals = {
            "int_": int_.output_ports[0],
        }
        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        res = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals, options=options
        )

        time = res.time
        int__ = res.outputs["int_"]
        int__sol = np.array(res.time)
        cutoff_idx = np.argmin(np.abs(res.time - upper_limit))
        int__sol[cutoff_idx:] = upper_limit

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, int__, label="int__", marker="_")
            ax1.plot(time, int__sol, label="int__sol", marker="o")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert np.allclose(int__, int__sol)

    def test_lower_limit(self, lower_limit=-2.0, show_plot=False):
        diagram = self._make_limits_test_model(const=-1.0, lower_limit=lower_limit)
        int_ = diagram["integrator"]
        ctx = diagram.create_context()

        t1 = np.abs(lower_limit)

        tf = t1 * 2.0
        recorded_signals = {
            "int_": int_.output_ports[0],
        }
        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        res = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals, options=options
        )

        time = res.time
        int__ = res.outputs["int_"]
        int__sol = np.array(res.time * -1.0)
        cutoff_idx = np.argmin(np.abs(res.time - t1))
        int__sol[cutoff_idx:] = lower_limit

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, int__, label="int__", marker="_")
            ax1.plot(time, int__sol, label="int__sol", marker="o")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert np.allclose(int__, int__sol)

    def test_hold(self, show_plot=False):
        t1 = 1.0

        builder = collimator.DiagramBuilder()
        int_ = builder.add(
            library.Integrator(
                0.0,
                name="integrator",
                enable_hold=True,
            )
        )
        constant = builder.add(library.Constant(t1, name="constant"))
        ramp = builder.add(library.Ramp(start_time=0.0))
        cmp = builder.add(library.Comparator(operator=">="))
        builder.connect(constant.output_ports[0], int_.input_ports[0])
        builder.connect(ramp.output_ports[0], cmp.input_ports[0])
        builder.connect(constant.output_ports[0], cmp.input_ports[1])
        builder.connect(cmp.output_ports[0], int_.input_ports[1])

        diagram = builder.build()
        ctx = diagram.create_context()

        tf = t1 * 2.0
        recorded_signals = {
            "int_": int_.output_ports[0],
            "ramp_": ramp.output_ports[0],
            "cmp_": cmp.output_ports[0],
        }
        options = collimator.SimulatorOptions(
            atol=1e-10,
            rtol=1e-8,
        )
        res = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals, options=options
        )

        time = res.time
        int__ = res.outputs["int_"]
        int__sol = np.array(res.time)
        cutoff_idx = np.argmin(np.abs(res.time - t1))
        int__sol[cutoff_idx:] = t1

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, res.outputs["ramp_"], label="ramp_")
            ax1.plot(time, res.outputs["cmp_"], label="cmp_")
            ax1.plot(time, int__, label="int__", marker="_")
            ax1.plot(time, int__sol, label="int__sol", marker="o")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert np.allclose(int__, int__sol)

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_array_integrate(self, backend):
        dim0 = 4
        dim1 = 2
        st_shape = (dim0, dim1)
        dim2 = dim0 * dim1
        cnp.set_backend(backend)
        x0 = cnp.arange(dim2, dtype=np.float64).reshape(st_shape)
        builder = collimator.DiagramBuilder()
        int_ = builder.add(library.Integrator(x0, name="int_"))
        builder.connect(int_.output_ports[0], int_.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        recorded_signals = {"x": int_.output_ports[0]}
        results = collimator.simulate(
            diagram,
            ctx,
            (0.0, 1.0),
            recorded_signals=recorded_signals,
        )

        x = results.outputs["x"]
        x_last = x[0, :]

        assert x_last.shape == st_shape


class TestPID:
    def _make_pid_diagram(self, source, kp, ki, kd, n):
        builder = collimator.DiagramBuilder()

        pid = builder.add(library.PID(kp=kp, ki=ki, kd=kd, n=n, name="pid"))
        source = builder.add(source)
        builder.connect(source.output_ports[0], pid.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()

        return diagram, context

    def test_pid_derivative(self):
        source = library.Sine()
        diagram, context = self._make_pid_diagram(
            source=source, kp=0.0, ki=0.0, kd=1.0, n=100.0
        )

        recorded_signals = {
            "u": diagram["pid"].output_ports[0],
        }

        options = collimator.SimulatorOptions(
            rtol=1e-6,
            atol=1e-8,
        )
        results = collimator.simulate(
            diagram,
            context,
            (0.0, 10.0),
            options=options,
            recorded_signals=recorded_signals,
        )

        # The derivative is only a filtered estimate, so we have to be a bit generous
        # with tolerances here. It also takes a while to converge.
        assert np.allclose(
            results.outputs["u"][-10:],
            np.cos(results.time[-10:]),
            atol=1e-2,
        )

    def test_pid_integral(self):
        source = library.Constant(value=1.0)
        diagram, context = self._make_pid_diagram(
            source=source, kp=0.0, ki=1.0, kd=0.0, n=0.0
        )

        recorded_signals = {
            "u": diagram["pid"].output_ports[0],
        }

        results = collimator.simulate(
            diagram,
            context,
            (0.0, 10.0),
            recorded_signals=recorded_signals,
        )

        assert np.allclose(
            results.outputs["u"],
            results.time,
        )

    def test_pid_proportional(self):
        kp = 2.5
        source = library.Sine()
        diagram, context = self._make_pid_diagram(
            source=source, kp=kp, ki=0.0, kd=0.0, n=0.0
        )

        recorded_signals = {
            "u": diagram["pid"].output_ports[0],
        }

        results = collimator.simulate(
            diagram,
            context,
            (0.0, 10.0),
            recorded_signals=recorded_signals,
        )

        assert np.allclose(
            results.outputs["u"],
            kp * np.sin(results.time),
        )

    def test_pid_param(self):
        kp = Parameter(value=1.0)

        builder = collimator.DiagramBuilder()
        constant = builder.add(library.Constant(1.0))
        pid = builder.add(library.PID(kp, 1.0, 1.0, 1.0))

        builder.connect(constant.output_ports[0], pid.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()

        out = pid.output_ports[0].eval(context)
        assert out == 2.0

        kp.set(2.0)
        context = diagram.create_context()
        out = pid.output_ports[0].eval(context)
        assert out == 3.0


class TestLTISystem:
    def test_feedthrough(self):
        A, B, C, D = (
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([[0.1]]),
            np.array([[2.0]]),
        )
        block = library.LTISystem(A, B, C, D)
        block.initialize(A=A, B=B, C=C, D=D)
        assert block.get_feedthrough() == [(0, 0)]

        D = np.array([[0.0]])
        block.initialize(A=A, B=B, C=C, D=D)
        assert block.get_feedthrough() == []

    @pytest.mark.parametrize(
        "A, B, C, D, x0, u",
        [
            (
                np.array([[1, 0.1], [-0.1, 1]]),
                np.array([[0.0], [1.0]]),
                np.array([[1.0, 0.0]]),
                np.array([[0.0]]),
                np.array([0.0, 0.0]),
                np.array([1.0]),
            ),
            (
                np.array([[1, 0.1, 0], [0, 1, 0.1], [-0.1, 0, 1]]),
                np.array([[1, 0], [0, 1], [0.5, 0.5]]),
                np.array([[1.0, 0, 0], [0, 1, 0]]),
                np.array([[0, 0.1], [0.1, 0]]),
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0]),
            ),
        ],
    )
    def test_lti_system(self, A, B, C, D, x0, u, tf=1.0):
        builder = collimator.DiagramBuilder()

        control = builder.add(library.Constant(u))
        lti = builder.add(library.LTISystem(A, B, C, D, x0))
        builder.connect(control.output_ports[0], lti.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        recorded_signals = {
            "y": lti.output_ports[0],
        }
        sol = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        time_points = sol.time
        y = sol.outputs["y"]

        # Computed expected solution

        # Precompute the inverse of A if it's needed for the integral part
        if np.linalg.det(A) != 0:  # Check if A is invertible
            A_inv = np.linalg.inv(A)
        else:
            raise ValueError("Matrix A is not invertible.")

        y_expected = []

        for t in time_points:
            Phi_t = expm(A * t)
            integral_part = np.dot(A_inv, (Phi_t - np.eye(A.shape[0]))) @ B @ u
            x_t = np.dot(Phi_t, x0) + integral_part
            y_t = np.dot(C, x_t) + np.dot(D, u)
            y_expected.append(y_t.ravel())

        y_expected = np.array(y_expected)

        if C.shape[0] == 1:
            y_expected = y_expected.ravel()

        # User lower tolerances due to difference in `expm` and ode solvers
        assert np.allclose(y, y_expected, atol=1e-04)

    def test_initial_states(self):
        builder = collimator.DiagramBuilder()

        control = builder.add(library.Constant(1.0))
        A = [0.0]
        B = [1.0]
        C = [1.0]
        D = [0.0]
        x0 = [10.0]
        lti = builder.add(library.LTISystem(A, B, C, D, x0))
        builder.connect(control.output_ports[0], lti.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()

        y0 = lti.output_ports[0].eval(ctx)
        print(y0)
        assert np.allclose(y0, x0)


class TestTransferFunction:
    def test_feedthrough(self):
        block = library.TransferFunction(num=[2], den=[1])
        block.initialize(num=[2], den=[1])
        assert block.get_feedthrough() == [(0, 0)]

        block.initialize(num=[0.1], den=[1, -1])
        assert block.get_feedthrough() == []

    @pytest.mark.parametrize(
        "num, den, u",
        [
            (
                [0.1],
                [1.0, -2.0, 1.01],
                np.array([1.0]),
            ),
            (
                [2.0, 5.0, 3.0, 1.0],
                [1.0, 4.0, 6.0, 4.0],
                np.array([1.0]),
            ),
        ],
    )
    def test_transfer_function(self, num, den, u, tf=1.0):
        builder = collimator.DiagramBuilder()

        control_block = builder.add(library.Constant(u))
        block = builder.add(library.TransferFunction(num, den))
        builder.connect(control_block.output_ports[0], block.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        recorded_signals = {
            "y": block.output_ports[0],
        }
        sol = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        time_points = sol.time
        y = sol.outputs["y"]

        # Get linear system representation
        linear_system = control.tf2ss(num, den)
        A, B, C, D = linear_system.A, linear_system.B, linear_system.C, linear_system.D
        # Computed expected solution for the linear system

        # Precompute the inverse of A if it's needed for the integral part
        if np.linalg.det(A) != 0:  # Check if A is invertible
            A_inv = np.linalg.inv(A)
        else:
            raise ValueError("Matrix A is not invertible.")

        x0 = np.zeros(A.shape[0])
        y_expected = []

        for t in time_points:
            Phi_t = expm(A * t)
            integral_part = np.dot(A_inv, (Phi_t - np.eye(A.shape[0]))) @ B @ u
            x_t = np.dot(Phi_t, x0) + integral_part
            y_t = np.dot(C, x_t) + np.dot(D, u)
            y_expected.append(y_t.ravel())

        y_expected = np.array(y_expected)

        if C.shape[0] == 1:
            y_expected = y_expected.ravel()

        # User lower tolerances due to difference in `expm` and ode solvers
        assert np.allclose(y, y_expected, atol=1e-04)

    def test_transfer_function_num_longer_than_den(self):
        num = [3, 2, 1, 0.1]
        den = [1.0, -2.0, 1.01]
        with pytest.raises(ValueError):
            _ = library.TransferFunction(num, den)


class TestDerivative:
    def test_feedthrough(self):
        filter_coefficient = 100
        block = library.Derivative(filter_coefficient)
        block.initialize(filter_coefficient=filter_coefficient)
        assert block.get_feedthrough() == [(0, 0)]

    def test_derivative_sine(self, show_plot=False):
        builder = collimator.DiagramBuilder()

        filter_coefficient = 100
        sine = builder.add(library.Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        der = builder.add(library.Derivative(filter_coefficient))
        builder.connect(sine.output_ports[0], der.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()

        recorded_signals = {
            "u": sine.output_ports[0],
            "y": der.output_ports[0],
        }
        options = collimator.SimulatorOptions(
            rtol=1e-6,
            atol=1e-8,
        )
        results = collimator.simulate(
            diagram,
            context,
            (0.0, 2.0),
            recorded_signals=recorded_signals,
            options=options,
        )
        ts, ys = results.time, results.outputs["y"]

        # Compute finite difference solution
        der_sol = np.cos(ts)

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(ts, results.outputs["u"], label="u", marker="x")
            ax1.plot(ts, results.outputs["y"], label="y", marker="x")
            ax1.plot(ts, der_sol, label="der_sol", marker="o")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        # we cannot expect very good adherence to the analytical solution.
        idx_where_compare_might_work = 10
        assert np.allclose(
            ys[idx_where_compare_might_work:],
            der_sol[idx_where_compare_might_work:],
            atol=0.05,
        )

    def test_derivative_of_integral(self, show_plot=False):
        builder = collimator.DiagramBuilder()

        filter_coefficient = 100
        A = 1.0
        const = builder.add(library.Constant(A))
        integrator = builder.add(library.Integrator(initial_state=np.zeros_like(A)))
        derivative = builder.add(library.Derivative(filter_coefficient))

        builder.connect(const.output_ports[0], integrator.input_ports[0])
        builder.connect(integrator.output_ports[0], derivative.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "y": derivative.output_ports[0],
        }
        options = collimator.SimulatorOptions(
            rtol=1e-6,
            atol=1e-8,
        )
        results = collimator.simulate(
            diagram,
            context,
            (0.0, 2.0),
            recorded_signals=recorded_signals,
            options=options,
        )
        ts, ys = results.time, results.outputs["y"]

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(ts, ys, label="y", marker="x")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        # we cannot expect very good adherence to the analytical solution.
        idx_where_compare_might_work = 10
        assert np.allclose(
            ys[idx_where_compare_might_work:],
            A,
            atol=0.01,
        )

    def test_vector_input_error(self):
        builder = collimator.DiagramBuilder()

        filter_coefficient = 20
        A = np.array([0.0, 1.0])
        const = builder.add(library.Constant(A))
        integrator = builder.add(library.Integrator(initial_state=np.zeros_like(A)))
        derivative = builder.add(library.Derivative(filter_coefficient))

        builder.connect(const.output_ports[0], integrator.input_ports[0])
        builder.connect(integrator.output_ports[0], derivative.input_ports[0])

        with pytest.raises(StaticError) as e:
            diagram = builder.build()
            diagram.create_context()
        # Success! The test failed as expected.
        print(e)
        assert "Derivative must have scalar input." in str(e)


if __name__ == "__main__":
    # TestIntegrator().test_upper_limit(show_plot=False)
    TestIntegrator().test_hold(show_plot=True)
    # TestDerivative().test_feedthrough()
    # TestDerivative().test_derivative_sine(show_plot=True)
    # TestDerivative().test_derivative_of_integral(show_plot=True)
    # TestDerivative().test_vector_input_error()
    # TestLTISystem().test_initial_states()
