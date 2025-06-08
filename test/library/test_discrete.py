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

"""Test discrete system blocks.

Contains tests for:
- DerivativeDiscrete
- DiscreteInitializer
- EdgeDetection
- FilterDiscrete
- IntegratorDiscrete
- PIDDiscrete
- TransferFunctionDiscrete
- LTISystemDiscrete
- UnitDelay
- ZeroOrderHold
"""

import pytest

import numpy as np
import jax.numpy as jnp
from scipy import signal

import matplotlib.pyplot as plt

import collimator
from collimator import library

# from collimator import logging
# logging.set_log_level(logging.DEBUG)

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


# TODO:
# - Test filter options (forward euler, backward euler, bilinear)
class TestDerivativeDiscrete:
    def test_feedthrough(self):
        dt = 0.1
        block = library.DerivativeDiscrete(dt)
        block.initialize()
        assert block.get_feedthrough() == [(0, 0)]

    def test_discrete_derivative_sine(self, show_plot=False):
        builder = collimator.DiagramBuilder()

        dt = 0.1
        sine = builder.add(library.Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        derivative_dt = builder.add(library.DerivativeDiscrete(dt=dt))
        builder.connect(sine.output_ports[0], derivative_dt.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()

        recorded_signals = {
            "y": derivative_dt.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        ts, ys = results.time, results.outputs["y"]

        # Compute finite difference solution
        u = np.sin(ts)
        y_sol = np.zeros_like(u)
        u_prev = 0.0
        y_prev = 0.0
        t_prev = 0.0
        for i in range(len(ts)):
            y_sol[i] = y_prev
            if abs(ts[i] - (t_prev)) < 1e-6:
                t_prev += dt
                y_sol[i] = (u[i] - u_prev) / dt
                y_prev = y_sol[i]
                u_prev = u[i]

        if show_plot:
            fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

            ax1.plot(ts, results.outputs["y"], label="y", marker="x")
            ax1.plot(ts, y_sol, label="y_sol", marker="o")
            ax1.grid(True)
            ax1.set_ylim(bottom=-1, top=11)
            ax1.legend()

            # ax2.plot(time, results.outputs["ud"], label="ud", marker="x")
            # ax2.plot(time_sol, ud_sol, label="ud_sol", marker="o")
            # ax2.grid(True)
            # ax2.set_ylim(bottom=-1, top=11)
            # ax2.legend()

            plt.show()

        assert jnp.allclose(ys, y_sol)

    def test_discrete_derivative_of_integral_scalar(self):
        builder = collimator.DiagramBuilder()

        dt = 0.1
        clock = builder.add(library.DiscreteClock(dt=dt))
        integrator = builder.add(library.IntegratorDiscrete(dt=dt, initial_state=0.0))

        derivative = builder.add(library.DerivativeDiscrete(dt=dt))

        builder.connect(clock.output_ports[0], integrator.input_ports[0])
        builder.connect(integrator.output_ports[0], derivative.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "clock": clock.output_ports[0],
            "derivative": derivative.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )

        # Check that the derivative is the same as the clock delayed by two samples
        #  (one for the finite difference derivative calculation, and one for the
        #  convention of publishing the x⁻ values at the beginning of the step).
        assert jnp.allclose(
            results.outputs["derivative"][1:], results.outputs["clock"][:-1]
        )

    def test_discrete_derivative_of_integral_vector(self):
        builder = collimator.DiagramBuilder()

        dt = 0.1
        A = jnp.array([0.0, 1.0])
        const = builder.add(library.Constant(A))
        integrator = builder.add(
            library.IntegratorDiscrete(dt=dt, initial_state=jnp.zeros_like(A))
        )
        derivative = builder.add(library.DerivativeDiscrete(dt=dt))

        builder.connect(const.output_ports[0], integrator.input_ports[0])
        builder.connect(integrator.output_ports[0], derivative.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "derivative": derivative.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        ys = results.outputs["derivative"]

        # Check that the derivative is a constant after two samples
        #  (one for the finite difference derivative calculation, and one for the
        #  convention of publishing the x⁻ values at the beginning of the step).
        assert jnp.allclose(ys[2:], A)


class TestDiscreteInitializer:
    def test_discrete_initializer(self, show_plot=False):
        builder = collimator.DiagramBuilder()

        dt = 0.1
        di = builder.add(library.DiscreteInitializer(dt=dt))
        di_vec = builder.add(
            library.DiscreteInitializer(dt=dt, initial_state=np.array([True, False]))
        )

        diagram = builder.build()
        context = diagram.create_context()

        recorded_signals = {
            "di": di.output_ports[0],
            "di_vec": di_vec.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 0.5), recorded_signals=recorded_signals
        )
        ts, ys, ys_vec = results.time, results.outputs["di"], results.outputs["di_vec"]

        # Compute finite difference solution
        y_sol = np.array([False] * len(ts))
        y_sol[0] = True
        y_sol2 = np.logical_not(y_sol)

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(ts, ys, label="y", marker="x")
            ax1.plot(ts, y_sol, label="y_sol", marker="o")
            ax1.plot(ts, ys_vec[:, 1], label="ys_vec[1]", marker="x")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert jnp.all(ys == y_sol)
        assert jnp.all(ys_vec[:, 0] == y_sol)
        assert jnp.all(ys_vec[:, 1] == y_sol2)


class TestEdgeDetection:
    def test_edge_detection(self, show_plot=False):
        builder = collimator.DiagramBuilder()

        dt = 0.1
        step_up = builder.add(
            library.Step(start_value=False, end_value=True, step_time=0.5)
        )
        step_down = builder.add(
            library.Step(start_value=True, end_value=False, step_time=1.5)
        )
        stepper = builder.add(library.LogicalOperator(function="and"))
        rising = builder.add(library.EdgeDetection(dt=dt, edge_detection="rising"))
        falling = builder.add(library.EdgeDetection(dt=dt, edge_detection="falling"))
        either = builder.add(library.EdgeDetection(dt=dt, edge_detection="either"))

        builder.connect(step_up.output_ports[0], stepper.input_ports[0])
        builder.connect(step_down.output_ports[0], stepper.input_ports[1])
        builder.connect(stepper.output_ports[0], rising.input_ports[0])
        builder.connect(stepper.output_ports[0], falling.input_ports[0])
        builder.connect(stepper.output_ports[0], either.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()

        recorded_signals = {
            "stepper": stepper.output_ports[0],
            "rising": rising.output_ports[0],
            "falling": falling.output_ports[0],
            "either": either.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        ts = results.time
        stepper_ = results.outputs["stepper"]
        rising_ = results.outputs["rising"]
        falling_ = results.outputs["falling"]
        either_ = results.outputs["either"]

        # Compute finite difference solution
        sol_tmp = np.array([False] * len(ts))
        rising_sol = sol_tmp.copy()
        rising_sol[5] = True

        falling_sol = sol_tmp.copy()
        falling_sol[15] = True

        either_sol = sol_tmp.copy()
        either_sol[5] = True
        either_sol[15] = True

        if show_plot:
            fig02, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9, 12))
            ax1.plot(ts, stepper_, label="stepper_")
            ax1.grid(True)
            ax1.legend()

            ax2.plot(ts, rising_, label="rising_")
            ax2.plot(ts, rising_sol, label="rising_sol")
            ax2.grid(True)
            ax2.legend()

            ax3.plot(ts, falling_, label="falling_")
            ax3.plot(ts, falling_sol, label="falling_sol")
            ax3.grid(True)
            ax3.legend()

            ax4.plot(ts, either_, label="either_")
            ax4.plot(ts, either_sol, label="either_sol")
            ax4.grid(True)
            ax4.legend()

            plt.show()

        assert jnp.all(rising_ == rising_sol)
        assert jnp.all(falling_ == falling_sol)
        assert jnp.all(either_ == either_sol)


class TestFilterDiscrete:
    def test_feedthrough(self):
        dt = 0.1
        filter_N = 10
        b = np.ones(filter_N) / filter_N
        block = library.FilterDiscrete(dt, b_coefficients=b)
        block.initialize(b_coefficients=b)
        assert block.is_feedthrough
        assert block.get_feedthrough() == [(0, 0)]

        # Not feedthrough if b[0] = 0
        b[0] = 0.0
        block = library.FilterDiscrete(dt, b_coefficients=b)
        assert not block.is_feedthrough
        assert block.get_feedthrough() == []

    def test_filter_discrete_fir(self, show_plot=False):
        """
        This test passes same input to both feedthrough and discrete implementation
        of the FilterDiscrete block.
        """
        sim_stop_time = 3.0
        dt = 0.1

        filter_N = 10
        b = jnp.ones(filter_N) / filter_N

        builder = collimator.DiagramBuilder()
        step = builder.add(library.Step(step_time=0.1))
        fir = builder.add(library.FilterDiscrete(dt=dt, b_coefficients=b))

        builder.connect(step.output_ports[0], fir.input_ports[0])

        recorded_signals = {
            "step": step.output_ports[0],
            "fir": fir.output_ports[0],
        }
        diagram = builder.build()
        context = diagram.create_context()
        options = collimator.SimulatorOptions(enable_tracing=False)
        res = collimator.simulate(
            diagram,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
            options=options,
        )

        time = np.array(res.time)
        step_ = np.array(res.outputs["step"])
        fir_ = np.array(res.outputs["fir"])
        fir_sol = signal.lfilter(b, 1, step_)

        print(f"time=\n{time}")
        print(f"step_=\n{step_}")
        print(f"fir_=\n{fir_}")
        print(f"fir_sol=\n{fir_sol}")

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, step_, label="step_", marker="_")
            ax1.plot(time, fir_, label="fir_", marker="o")
            ax1.plot(time, fir_sol, label="fir_sol", marker="x")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert jnp.allclose(fir_, fir_sol)


scalar_testdata = []
for x0 in [0, 1, -1]:
    for dtype in float_dtypes:
        scalar_testdata.append((x0, dtype))

array_testdata = []
for A in [[1, 2, 3], [[1, 2, 3]], [[1], [2], [3]], [[1, 2], [3, 4]]]:
    for dtype in float_dtypes:
        array_testdata.append((A, dtype))

# Need to relax the tolerance here to allow for floating point error
dtype_atol = {
    np.float64: 1e-14,
    np.float32: 1e-6,
    np.float16: 1e-2,
}

debug_testdata = [(0.0, np.float64)]

limits_data = [
    (None, None, 1.0),
    (1.0, None, 1.0),
    (None, -1.0, -1.0),
    (1.0, -1.0, 1.0),
]


class TestIntegratorDiscrete:
    def test_feedthrough(self):
        dt = 0.1
        block = library.IntegratorDiscrete(dt, 0.0)
        assert block.get_feedthrough() == []

        # If the integrator has reset, it will be feedthrough
        # from the reset trigger to the output
        block = library.IntegratorDiscrete(dt, 0.0, enable_reset=True)
        assert block.get_feedthrough() == [(1, 0)]

        # If external resets are also enabled, it is feedthrough
        # from the value to the output as well
        block = library.IntegratorDiscrete(
            dt, 0.0, enable_reset=True, enable_external_reset=True
        )
        assert block.get_feedthrough() == [(1, 0), (2, 0)]

    @pytest.mark.parametrize("x0,dtype", scalar_testdata)
    def test_discrete_scalar_constant(self, x0, dtype, dt=0.1, tf=1.0):
        x0 = dtype(x0)
        builder = collimator.DiagramBuilder()

        const = builder.add(library.Constant(dtype(1.0), name="const"))
        integrator = builder.add(
            library.IntegratorDiscrete(dt=dt, initial_state=x0, name="integrator")
        )
        builder.connect(const.output_ports[0], integrator.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        print(ctx.discrete_state)
        recorded_signals = {"y": integrator.output_ports[0]}

        results = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        xd = ctx[integrator.system_id].discrete_state
        assert xd.dtype == dtype
        assert results.outputs["y"].dtype == dtype
        assert np.allclose(
            results.outputs["y"], x0 + results.time, atol=dtype_atol[dtype]
        )

    @pytest.mark.parametrize("A,dtype", array_testdata)
    def test_discrete_array_constant(self, A, dtype, dt=0.1, tf=1.0):
        A = jnp.asarray(A, dtype=dtype)
        builder = collimator.DiagramBuilder()

        const = builder.add(library.Constant(A, name="const"))
        integrator = builder.add(
            library.IntegratorDiscrete(
                dt=dt, initial_state=jnp.zeros_like(A), name="integrator"
            )
        )
        builder.connect(const.output_ports[0], integrator.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()
        recorded_signals = {"y": integrator.output_ports[0]}
        results = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        y_sol = jnp.outer(results.time, A).reshape(results.outputs["y"].shape)

        assert results.outputs["y"].dtype == dtype
        assert jnp.allclose(results.outputs["y"], y_sol, atol=dtype_atol[dtype])

    # @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize("dtype", [jnp.float64])
    def test_discrete_reset(self, dtype, dt=0.1, tf=1.0):
        x0 = np.asarray([1, 0, 0], dtype=dtype)
        A = np.asarray([0, 1, 2], dtype=dtype)
        step_time = 0.5

        builder = collimator.DiagramBuilder()
        vec = builder.add(library.Constant(A, name="vec"))
        step = builder.add(
            library.Step(start_value=0, end_value=1, step_time=step_time)
        )
        const = builder.add(library.Constant(0.2, name="const"))
        comparator = builder.add(library.Comparator(operator=">", name="comparator"))

        integrator = builder.add(
            library.IntegratorDiscrete(
                dt=dt,
                initial_state=x0,
                name="integrator",
                enable_reset=True,
            )
        )

        builder.connect(vec.output_ports[0], integrator.input_ports[0])
        builder.connect(step.output_ports[0], comparator.input_ports[0])
        builder.connect(const.output_ports[0], comparator.input_ports[1])
        builder.connect(comparator.output_ports[0], integrator.input_ports[1])

        diagram = builder.build()
        ctx = diagram.create_context()
        recorded_signals = {"y": integrator.output_ports[0]}
        results = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        y_sol = x0 + np.outer(results.time, A).reshape(results.outputs["y"].shape)

        # Handle reset values
        for i in range(len(results.time)):
            if results.time[i] >= step_time:
                y_sol[i] = x0

        assert results.outputs["y"].dtype == dtype
        assert jnp.allclose(results.outputs["y"], y_sol, atol=dtype_atol[dtype])

    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_discrete_external_reset(self, dtype, dt=0.1, tf=1.0):
        x0 = np.asarray([1, 0, 0], dtype=dtype)
        A = np.asarray([0, 1, 2], dtype=dtype)
        B = np.asarray([4, 5, 6], dtype=dtype)
        step_time = 0.5

        builder = collimator.DiagramBuilder()
        vec = builder.add(library.Constant(A, name="vec"))
        step = builder.add(
            library.Step(start_value=0, end_value=1, step_time=step_time)
        )
        const = builder.add(library.Constant(0.2, name="const"))
        comparator = builder.add(library.Comparator(operator=">", name="comparator"))
        external_reset_vec = builder.add(library.Constant(B, name="external_reset_vec"))

        integrator = builder.add(
            library.IntegratorDiscrete(
                dt=dt,
                initial_state=x0,
                name="integrator",
                enable_reset=True,
                enable_external_reset=True,
            )
        )

        builder.connect(vec.output_ports[0], integrator.input_ports[0])
        builder.connect(step.output_ports[0], comparator.input_ports[0])
        builder.connect(const.output_ports[0], comparator.input_ports[1])
        builder.connect(comparator.output_ports[0], integrator.input_ports[1])
        builder.connect(external_reset_vec.output_ports[0], integrator.input_ports[2])

        diagram = builder.build()
        ctx = diagram.create_context()
        recorded_signals = {"y": integrator.output_ports[0]}
        results = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        y_sol = x0 + np.outer(results.time, A).reshape(results.outputs["y"].shape)

        # Handle reset values
        for i in range(len(results.time)):
            if results.time[i] >= step_time:
                y_sol[i] = B

        assert results.outputs["y"].dtype == dtype
        assert jnp.allclose(results.outputs["y"], y_sol, atol=dtype_atol[dtype])

    def test_hold_discrete(self, show_plot=False):
        t1 = 1.0
        dt = 0.1
        builder = collimator.DiagramBuilder()
        int_ = builder.add(
            library.IntegratorDiscrete(
                dt,
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
        res = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        time = res.time
        int__ = res.outputs["int_"]
        int__sol = np.array(res.time)
        # Note that the outputs react to "hold" one time step
        # later, since the update equations are as follows:
        # t[n]:
        #   y[n] = x[n]
        #   x[n+1] = x[n] + h*u[n]
        # t[n+1]: hold input is now true
        #   y[n+1] = x[n+1]
        #   x[n+2] = x[n+1]
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

        assert jnp.allclose(int__, int__sol)

    @pytest.mark.parametrize("ulim,llim,const", limits_data)
    def test_limit_discrete(self, ulim, llim, const, show_plot=False):
        t1 = 1.0
        dt = 0.1
        builder = collimator.DiagramBuilder()
        int_ = builder.add(
            library.IntegratorDiscrete(
                dt,
                0.0,
                name="integrator",
                enable_limits=True,
                upper_limit=ulim,
                lower_limit=llim,
            )
        )
        constant = builder.add(library.Constant(const, name="constant"))

        builder.connect(constant.output_ports[0], int_.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        tf = t1 * 2.0
        recorded_signals = {
            "int_": int_.output_ports[0],
        }
        res = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        time = res.time
        int__ = res.outputs["int_"]

        if llim is None and ulim is None:
            int__sol = time.copy()
        else:
            int__sol = np.array(res.time) * const
            cutoff_idx = np.argmin(np.abs(res.time - t1)) + 1
            int__sol[cutoff_idx:] = (t1) * const

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, int__, label="int__", marker="o")
            ax1.plot(time, int__sol, label="int__sol")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert jnp.allclose(int__, int__sol)


# TODO:
# - Test filter options for the derivative filter
class TestPIDDiscrete:
    # Tests equivalence between the PID controller and one constructed from
    # primitive blocks.  However, note that because of the way the
    # IntegratorDiscrete block is initialized, the two are not equivalent unless
    # the initial value of the signal zero (hence the use of the sine input)

    def test_feedthrough(self):
        dt = 0.1
        block = library.PIDDiscrete(dt)
        assert block.get_feedthrough() == [(0, 0)]

    def _test_open_loop(self, kp=0.0, ki=0.0, kd=0.0, dt=0.1):
        builder = collimator.DiagramBuilder()

        sine = builder.add(library.Sine(frequency=1.0, amplitude=1.0, phase=0.0))

        # Here the PID and Derivative blocks are  discrete-time, so we do not need to
        # zero-order hold their outputs.  However, we do need to zero-order hold the
        # gain block for the "p" term so that the delay is consistent with the signal
        # timing of the IntegratorDiscrete and DerivativeDiscrete.

        pid = builder.add(library.PIDDiscrete(kp=kp, ki=ki, kd=kd, dt=dt))
        builder.connect(sine.output_ports[0], pid.input_ports[0])

        derivative = builder.add(library.DerivativeDiscrete(dt=dt))
        integral = builder.add(library.IntegratorDiscrete(dt=dt, initial_state=0.0))

        gain_p = builder.add(library.Gain(kp, name="gain_p"))
        gain_i = builder.add(library.Gain(ki, name="gain_i"))
        gain_d = builder.add(library.Gain(kd, name="gain_d"))

        pid_sol = builder.add(library.Adder(3, name="pid_sol"))
        zoh = builder.add(library.ZeroOrderHold(dt=dt, name="zoh"))

        builder.connect(sine.output_ports[0], zoh.input_ports[0])
        builder.connect(zoh.output_ports[0], gain_p.input_ports[0])
        builder.connect(sine.output_ports[0], integral.input_ports[0])
        builder.connect(integral.output_ports[0], gain_i.input_ports[0])
        builder.connect(sine.output_ports[0], derivative.input_ports[0])
        builder.connect(derivative.output_ports[0], gain_d.input_ports[0])

        builder.connect(gain_p.output_ports[0], pid_sol.input_ports[0])
        builder.connect(gain_i.output_ports[0], pid_sol.input_ports[1])
        builder.connect(gain_d.output_ports[0], pid_sol.input_ports[2])

        diagram = builder.build()
        context = diagram.create_context()

        recorded_signals = {
            "pid": pid.output_ports[0],
            "pid_sol": pid_sol.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        assert jnp.allclose(results.outputs["pid"], results.outputs["pid_sol"])

    def test_p(self):
        self._test_open_loop(kp=1.0)

    def test_pi(self):
        self._test_open_loop(kp=1.0, ki=10.0)

    def test_pd(self):
        self._test_open_loop(kp=1.0, kd=0.1)

    def test_pid(self):
        self._test_open_loop(kp=1.0, ki=10.0, kd=0.1)

    def test_pid_consistent_with_derivative_discrete(self):
        # "Integration" test that the PIDDiscrete block is consistent with the
        # DerivativeDiscrete block when the proportional and integral gains are
        # zero.

        builder = collimator.DiagramBuilder()

        dt = 0.1
        sine = builder.add(library.Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        pid = builder.add(library.PIDDiscrete(kp=0.0, ki=0.0, kd=1.0, dt=dt))
        builder.connect(sine.output_ports[0], pid.input_ports[0])
        derivative = builder.add(library.DerivativeDiscrete(dt=dt))
        builder.connect(sine.output_ports[0], derivative.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()

        recorded_signals = {
            "pid": pid.output_ports[0],
            "derivative": derivative.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )

        assert jnp.allclose(results.outputs["pid"], results.outputs["derivative"])

    def test_nondefault_initial_state(self):
        dt = 0.1
        initial_state = 1.0
        builder = collimator.DiagramBuilder()
        sine = builder.add(library.Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        pid = builder.add(
            library.PIDDiscrete(kp=0.0, ki=0.0, kd=1.0, dt=dt, initial_state=1.0)
        )
        builder.connect(sine.output_ports[0], pid.input_ports[0])

        system = builder.build()
        context = system.create_context()
        assert context[pid.system_id].discrete_state.integral == initial_state
        assert context[pid.system_id].discrete_state.e_prev == 0.0
        assert context[pid.system_id].discrete_state.e_dot_prev == 0.0

    def test_external_initial_state(self):
        dt = 0.1
        initial_state = 1.0
        builder = collimator.DiagramBuilder()
        sine = builder.add(library.Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        const = builder.add(library.Constant(initial_state))
        pid = builder.add(
            library.PIDDiscrete(
                kp=0.0,
                ki=0.0,
                kd=1.0,
                dt=dt,
                enable_external_initial_state=True,
            )
        )
        builder.connect(sine.output_ports[0], pid.input_ports[0])
        builder.connect(const.output_ports[0], pid.input_ports[1])

        system = builder.build()
        context = system.create_context()
        assert context[pid.system_id].discrete_state.integral == initial_state
        assert context[pid.system_id].discrete_state.e_prev == 0.0
        assert context[pid.system_id].discrete_state.e_dot_prev == 0.0


class TestTransferFunctionDiscrete:
    def test_feedthrough(self):
        dt = 0.1
        block = library.TransferFunctionDiscrete(dt, num=[2], den=[1])
        block.initialize(num=[2], den=[1])
        assert block.get_feedthrough() == [(0, 0)]

        block.initialize(num=[0.1], den=[1, -1])
        assert block.get_feedthrough() == []

    def test_transfer_function_discrete(self, show_plot=False):
        t1 = 1.0
        dt = 0.1
        builder = collimator.DiagramBuilder()

        ramp = builder.add(library.Ramp())
        # zoh = builder.add(library.ZeroOrderHold(dt))
        tf_gain = builder.add(library.TransferFunctionDiscrete(dt, num=[2], den=[1]))
        tf_gain_sol = builder.add(library.Gain(gain=2.0))
        tf_int = builder.add(
            library.TransferFunctionDiscrete(dt, num=[0.1], den=[1, -1])
        )
        tf_int_sol = builder.add(library.IntegratorDiscrete(dt, initial_state=0.0))
        tf_ud = builder.add(library.TransferFunctionDiscrete(dt, num=[1], den=[1, 0]))
        tf_ud_sol = builder.add(library.UnitDelay(dt, initial_state=0.0))
        tf_0 = builder.add(library.TransferFunctionDiscrete(dt, num=[4], den=[2, 3]))
        add0 = builder.add(library.Adder(2, operators="+-"))
        gain_p5 = builder.add(library.Gain(gain=0.5))
        ud_x1 = builder.add(library.UnitDelay(dt, initial_state=0.0))
        gain_3 = builder.add(library.Gain(gain=3.0))
        tf_0_sol = builder.add(library.Gain(gain=4.0))

        # builder.connect(ramp.output_ports[0], zoh.input_ports[0])
        builder.connect(ramp.output_ports[0], tf_gain.input_ports[0])
        builder.connect(ramp.output_ports[0], tf_gain_sol.input_ports[0])
        builder.connect(ramp.output_ports[0], tf_int.input_ports[0])
        builder.connect(ramp.output_ports[0], tf_int_sol.input_ports[0])
        builder.connect(ramp.output_ports[0], tf_ud.input_ports[0])
        builder.connect(ramp.output_ports[0], tf_ud_sol.input_ports[0])
        builder.connect(ramp.output_ports[0], tf_0.input_ports[0])

        builder.connect(ramp.output_ports[0], add0.input_ports[0])
        builder.connect(gain_3.output_ports[0], add0.input_ports[1])
        builder.connect(ud_x1.output_ports[0], gain_3.input_ports[0])
        builder.connect(add0.output_ports[0], gain_p5.input_ports[0])
        builder.connect(gain_p5.output_ports[0], ud_x1.input_ports[0])
        builder.connect(ud_x1.output_ports[0], tf_0_sol.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        tf = t1 * 2.0
        recorded_signals = {
            "tf_gain": tf_gain.output_ports[0],
            "tf_gain_sol": tf_gain_sol.output_ports[0],
            "tf_int": tf_int.output_ports[0],
            "tf_int_sol": tf_int_sol.output_ports[0],
            "tf_ud": tf_ud.output_ports[0],
            "tf_ud_sol": tf_ud_sol.output_ports[0],
            "tf_0": tf_0.output_ports[0],
            "tf_0_sol": tf_0_sol.output_ports[0],
        }
        res = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        time = res.time
        if show_plot:
            fig02, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9, 12))
            ax1.plot(time, res.outputs["tf_gain"], label="tf_gain")
            ax1.plot(time, res.outputs["tf_gain_sol"], label="tf_gain_sol")
            ax1.grid(True)
            ax1.legend()

            ax2.plot(time, res.outputs["tf_int"], label="tf_int")
            ax2.plot(time, res.outputs["tf_int_sol"], label="tf_int_sol")
            ax2.grid(True)
            ax2.legend()

            ax3.plot(time, res.outputs["tf_ud"], label="tf_ud")
            ax3.plot(time, res.outputs["tf_ud_sol"], label="tf_ud_sol")
            ax3.grid(True)
            ax3.legend()

            ax4.plot(time, res.outputs["tf_0"], label="tf_0")
            ax4.plot(time, res.outputs["tf_0_sol"], label="tf_0_sol")
            ax4.grid(True)
            ax4.legend()

            plt.show()

        assert jnp.allclose(res.outputs["tf_gain"], res.outputs["tf_gain_sol"])
        assert jnp.allclose(res.outputs["tf_int"], res.outputs["tf_int_sol"])
        assert jnp.allclose(res.outputs["tf_ud"], res.outputs["tf_ud_sol"])
        assert jnp.allclose(res.outputs["tf_0"], res.outputs["tf_0_sol"])


class TestLTISystemDiscrete:
    def test_feedthrough(self):
        dt = 0.1
        A, B, C, D = (
            jnp.array([[1.0]]),
            jnp.array([[1.0]]),
            jnp.array([[0.1]]),
            jnp.array([[2.0]]),
        )
        block = library.LTISystemDiscrete(A, B, C, D, dt)
        block.initialize(A=A, B=B, C=C, D=D)
        assert block.get_feedthrough() == [(0, 0)]

        D = jnp.array([[0.0]])
        block.initialize(A=A, B=B, C=C, D=D)
        assert block.get_feedthrough() == []

    @pytest.mark.parametrize(
        "A, B, C, D, x0, u",
        [
            (
                jnp.array([[1, 0.1], [-0.1, 1]]),
                jnp.array([[0.0], [1.0]]),
                jnp.array([[1.0, 0.0]]),
                jnp.array([[0.0]]),
                jnp.array([0.0, 0.0]),
                jnp.array([1.0]),
            ),
            (
                jnp.array([[1, 0.1, 0], [0, 1, 0.1], [-0.1, 0, 1]]),
                jnp.array([[1, 0], [0, 1], [0.5, 0.5]]),
                jnp.array([[1.0, 0, 0], [0, 1, 0]]),
                jnp.array([[0, 0.1], [0.1, 0]]),
                jnp.array([0.0, 0.0, 0.0]),
                jnp.array([1.0, 1.0]),
            ),
        ],
    )
    def test_lti_system_discrete(self, A, B, C, D, x0, u, dt=0.1, num_steps=100):
        dt = 0.1
        tf = dt * num_steps
        builder = collimator.DiagramBuilder()

        control = builder.add(library.Constant(u))
        lti = builder.add(library.LTISystemDiscrete(A, B, C, D, dt, x0))
        builder.connect(control.output_ports[0], lti.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        recorded_signals = {
            "y": lti.output_ports[0],
        }
        sol = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals
        )

        # Computed expected solution
        x_k = x0
        y_expected = []

        for _ in range(num_steps + 1):
            # Compute next state
            x_k_plus_1 = A.dot(x_k) + B.dot(u)
            # Compute output
            y_k = C.dot(x_k) + D.dot(u)
            # Store output
            y_expected.append(y_k)
            # Update state
            x_k = x_k_plus_1

        y_expected = jnp.array(y_expected)

        if C.shape[0] == 1:
            y_expected = y_expected.ravel()

        assert jnp.allclose(sol.outputs["y"], y_expected)


class TestUnitDelay:
    def test_feedthrough(self):
        dt = 0.1
        block = library.UnitDelay(dt, 0.0)
        block.initialize(initial_state=0.0)
        assert block.get_feedthrough() == []

    def test_unit_delay_scalar(self):
        """
        Test that a unit delay delays a continuous-time signal by one sample.
        """
        builder = collimator.DiagramBuilder()

        dt = 0.1
        z0 = 0.0
        ramp = builder.add(
            library.Ramp(slope=1.0, start_time=0.0, start_value=z0)
        )  # Time
        delay = builder.add(library.UnitDelay(dt=dt, initial_state=0.0))
        builder.connect(ramp.output_ports[0], delay.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {"y": delay.output_ports[0]}
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )

        assert jnp.allclose(results.outputs["y"][1:], results.time[:-1])

    def test_unit_delay_vec(self, show_plot=False):
        """
        This test case is similar to above, but with an array of inputs.
        """
        builder = collimator.DiagramBuilder()

        dt = 0.1
        tf = 2.0
        ramp = builder.add(
            library.Ramp(slope=1.0, start_time=0.0, start_value=0.0)
        )  # Time
        chirp = builder.add(library.Chirp(f0=0.0, f1=10.0, stop_time=tf, phi=0.0))
        mux = builder.add(library.Multiplexer(2))
        builder.connect(ramp.output_ports[0], mux.input_ports[0])
        builder.connect(chirp.output_ports[0], mux.input_ports[1])

        delay = builder.add(library.UnitDelay(dt=dt, initial_state=[0.0, 0.0]))
        builder.connect(mux.output_ports[0], delay.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "y": delay.output_ports[0],
            "ramp": ramp.output_ports[0],
            "chirp": chirp.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )

        if show_plot:
            time = results.time
            fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

            ax1.plot(time, results.outputs["y"][:, 0], label="ud[0]", marker="x")
            ax1.plot(time, results.outputs["ramp"], label="sol", marker="o")
            ax1.grid(True)
            ax1.set_ylim(bottom=-1, top=11)
            ax1.legend()

            ax2.plot(time, results.outputs["y"][:, 1], label="ud[1]", marker="x")
            ax2.plot(time, results.outputs["chirp"], label="sol", marker="o")
            ax2.grid(True)
            ax2.set_ylim(bottom=-1, top=11)
            ax2.legend()

            plt.show()

        ud_chirp_sol = results.outputs["chirp"][:-1]
        ud_chirp_sol[0] *= 0.0
        assert jnp.allclose(results.outputs["y"][1:, 0], results.outputs["ramp"][:-1])
        assert jnp.allclose(results.outputs["y"][1:, 1], ud_chirp_sol)


class TestZeroOrderHold:
    def test_zoh_bool(self):
        builder = collimator.DiagramBuilder()

        dt = 0.1
        tf = 2.0

        clk = builder.add(library.Clock())
        cmp = builder.add(library.Comparator(operator=">"))
        const = builder.add(library.Constant(0.2))
        zoh = builder.add(library.ZeroOrderHold(dt=dt))

        builder.connect(clk.output_ports[0], cmp.input_ports[0])
        builder.connect(const.output_ports[0], cmp.input_ports[1])
        builder.connect(cmp.output_ports[0], zoh.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "zoh": zoh.output_ports[0],
            "cmp": cmp.output_ports[0],
        }

        results = collimator.simulate(
            diagram,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        time = results.time
        print(f"time=\n{time}")
        assert jnp.allclose(results.time, jnp.arange(0, tf + dt, dt))

        zoh_y = results.outputs["zoh"]
        cmp_y = results.outputs["cmp"]

        assert cmp_y.dtype == jnp.bool_
        assert zoh_y.dtype == jnp.bool_

        # The ZOH should hold the value of the comparator input until the next
        # sample time with no delays
        assert jnp.allclose(zoh_y, cmp_y)

    def test_zoh_int(self):
        # Test zero-order hold of an integrated signal
        builder = collimator.DiagramBuilder()

        dt = 0.1
        tf = 2.0

        # Discrete-time clock for checking hold values
        dclk = builder.add(library.DiscreteClock(dt=dt))

        const = builder.add(library.Constant([0.0, 1.0]))
        int_ = builder.add(library.Integrator(initial_state=[0.0, 0.0]))
        zoh = builder.add(library.ZeroOrderHold(dt=dt))

        builder.connect(const.output_ports[0], int_.input_ports[0])
        builder.connect(int_.output_ports[0], zoh.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()

        recorded_signals = {
            "dclk": dclk.output_ports[0],
            "int_": int_.output_ports[0],
            "zoh": zoh.output_ports[0],
        }

        results = collimator.simulate(
            diagram,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        time = results.time
        print(f"time=\n{time}")

        assert jnp.allclose(results.outputs["int_"][:, 0], 0.0)
        assert jnp.allclose(results.outputs["int_"][:, 1], results.time)

        # Check that the output of the ZOH is equal to the integrator state,
        # held at the last sample time.  This will also be the value of the
        # discrete-time clock output
        print(f"zoh=\n{results.outputs['zoh'][:, 1]}")
        assert jnp.allclose(results.outputs["zoh"][:, 0], 0.0)
        assert jnp.allclose(results.outputs["zoh"][:, 1], results.outputs["dclk"])

    def test_zoh_zoh(self):
        """Check that two ZOH blocks in series don't do anything."""
        builder = collimator.DiagramBuilder()

        dt = 0.1
        tf = 2.0

        clk = builder.add(library.DiscreteClock(dt=dt))
        zoh1 = builder.add(library.ZeroOrderHold(dt=dt))
        zoh2 = builder.add(library.ZeroOrderHold(dt=dt))

        builder.connect(clk.output_ports[0], zoh1.input_ports[0])
        builder.connect(zoh1.output_ports[0], zoh2.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "zoh1": zoh1.output_ports[0],
            "zoh2": zoh2.output_ports[0],
        }

        results = collimator.simulate(
            diagram,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        time = results.time
        print(f"time=\n{time}")

        assert jnp.allclose(results.time, jnp.arange(0, tf + dt, dt))
        assert jnp.allclose(results.outputs["zoh1"], results.time)
        assert jnp.allclose(results.outputs["zoh1"], results.outputs["zoh2"])

    def test_zoh_mixed_vec(self, show_plot=False):
        """Test zoh of a mixed time-mode signal.

        The mux signal has mix of discrete and non-discrete - a discrete clock
        and a continuous clock.  The ZOH should hold the continuous clock signal
        and have no effect on the discrete clock signal, so both components of
        the vector output signal are the same.
        """
        builder = collimator.DiagramBuilder()

        dt = 0.1
        tf = 2.0
        clk = builder.add(library.Clock())
        dclk = builder.add(library.DiscreteClock(dt=dt))
        mux = builder.add(library.Multiplexer(2))
        builder.connect(clk.output_ports[0], mux.input_ports[0])
        builder.connect(dclk.output_ports[0], mux.input_ports[1])

        zoh = builder.add(library.ZeroOrderHold(dt=dt))
        builder.connect(mux.output_ports[0], zoh.input_ports[0])

        # Add a second discrete clock to oversample the output.
        # This ensures that the "hold" is working as expected on
        # the continuous component of the signal.
        builder.add(library.DiscreteClock(dt=0.1 * dt))

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "zoh": zoh.output_ports[0],
            "clk": clk.output_ports[0],
            "dclk": dclk.output_ports[0],
        }
        results = collimator.simulate(
            diagram,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )
        time = results.time
        print(f"time=\n{time}")

        if show_plot:
            fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

            ax1.plot(time, results.outputs["zoh"][:, 0], label="zoh[0]", marker="x")
            ax1.plot(time, results.outputs["ud"][:, 0], label="ud[0]", marker="x")
            ax1.plot(time, results.outputs["clk"], label="clk", marker="+")
            ax1.grid(True)
            ax1.set_ylim(bottom=-1, top=3)
            ax1.legend()

            ax2.plot(time, results.outputs["zoh"][:, 1], label="zoh[1]", marker="x")
            ax2.plot(time, results.outputs["ud"][:, 1], label="ud[1]", marker="x")
            ax2.plot(time, results.outputs["dclk"], label="dclk", marker="+")
            ax2.grid(True)
            ax2.set_ylim(bottom=-1, top=3)
            ax2.legend()

            plt.show()

        # Discrete clock is equivalent to ZOH applied to continuous clock
        assert jnp.allclose(results.outputs["dclk"], results.outputs["zoh"][:, 0])

        # Zero-order hold has no effect on discrete clock
        assert jnp.allclose(results.outputs["dclk"], results.outputs["zoh"][:, 1])

    # @pytest.mark.xfail(reason="Fix was reverted because it crashed prod models: #5957")
    def test_zoh_group_init(self):
        # Tests recursion error in static initialization ordering. See:
        # https://collimator.atlassian.net/browse/WC-266

        dt = 0.1

        # Construct the subdiagram
        builder = collimator.DiagramBuilder()
        zoh = builder.add(library.ZeroOrderHold(dt=dt, name="zoh"))
        delay = builder.add(library.UnitDelay(dt=dt, initial_state=0.0, name="delay"))
        builder.connect(zoh.output_ports[0], delay.input_ports[0])
        builder.export_input(zoh.input_ports[0])
        builder.export_output(delay.output_ports[0])

        group = builder.build(name="group")

        # Construct the root diagram
        builder = collimator.DiagramBuilder()
        builder.add(group)
        builder.connect(group.output_ports[0], group.input_ports[0])

        system = builder.build()
        system.create_context()


class TestRateLimiter:
    def test_feedthrough(self):
        dt = 0.1
        block = library.RateLimiter(dt)
        assert block.get_feedthrough() == [(0, 0)]

    def test_rate_limiter_scalar(self, show_plot=False):
        """
        Basic test case for a rate limiter
        """
        builder = collimator.DiagramBuilder()

        dt = 0.1
        ramp = builder.add(library.Ramp(start_time=0.0))
        ramp_sol = builder.add(library.Ramp(slope=0.5, start_time=0.0))
        rt = builder.add(library.RateLimiter(dt=dt, upper_limit=0.5))
        builder.connect(ramp.output_ports[0], rt.input_ports[0])

        ulim = builder.add(library.Constant(value=0.5))
        rt_dyn = builder.add(
            library.RateLimiter(dt=dt, enable_dynamic_upper_limit=True)
        )
        builder.connect(ramp.output_ports[0], rt_dyn.input_ports[0])
        builder.connect(ulim.output_ports[0], rt_dyn.input_ports[1])

        ramp_d = builder.add(library.Ramp(slope=-1.0, start_time=0.0))
        ramp_d_sol = builder.add(library.Ramp(slope=-0.5, start_time=0.0))
        rt_d = builder.add(library.RateLimiter(dt=dt, lower_limit=-0.5))
        builder.connect(ramp_d.output_ports[0], rt_d.input_ports[0])

        llim = builder.add(library.Constant(value=-0.5))
        rt_d_dyn = builder.add(
            library.RateLimiter(dt=dt, enable_dynamic_lower_limit=True)
        )
        builder.connect(ramp_d.output_ports[0], rt_d_dyn.input_ports[0])
        builder.connect(llim.output_ports[0], rt_d_dyn.input_ports[1])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "ramp": ramp.output_ports[0],
            "ramp_sol": ramp_sol.output_ports[0],
            "rt": rt.output_ports[0],
            "ramp_d": ramp_d.output_ports[0],
            "ramp_d_sol": ramp_d_sol.output_ports[0],
            "rt_d": rt_d.output_ports[0],
            "rt_dyn": rt_dyn.output_ports[0],
            "rt_d_dyn": rt_d_dyn.output_ports[0],
        }
        results = collimator.simulate(
            diagram,
            context,
            (0.0, 2.0),
            recorded_signals=recorded_signals,
        )

        if show_plot:
            time = results.time
            fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

            ax1.plot(time, results.outputs["ramp"], label="ramp", marker="o")
            ax1.plot(time, results.outputs["rt"], label="rt", marker="x")
            ax1.plot(time, results.outputs["ramp_sol"], label="ramp_sol")
            ax1.grid(True)
            ax1.legend()

            ax2.plot(time, results.outputs["ramp_d"], label="ramp_d", marker="o")
            ax2.plot(time, results.outputs["rt_d_dyn"], label="rt_d_dyn", marker="x")
            ax2.plot(time, results.outputs["ramp_d_sol"], label="ramp_d_sol")
            ax2.grid(True)
            ax2.legend()

            plt.show()

        assert jnp.allclose(results.outputs["rt"], results.outputs["ramp_sol"])
        assert jnp.allclose(results.outputs["rt_dyn"], results.outputs["ramp_sol"])
        assert jnp.allclose(results.outputs["rt_d"], results.outputs["ramp_d_sol"])
        assert jnp.allclose(results.outputs["rt_d_dyn"], results.outputs["ramp_d_sol"])

    def test_rate_limiter_vec(self, show_plot=False):
        """
        This test case is similar to above, but with an array of inputs.
        """
        builder = collimator.DiagramBuilder()

        dt = 0.1
        tf = 2.0
        ramp = builder.add(library.Ramp(start_time=0.0))
        ramp_sol = builder.add(library.Ramp(slope=0.5, start_time=0.0))
        chirp = builder.add(library.Chirp(f0=0.0, f1=10.0, stop_time=tf, phi=0.0))
        ramp_d = builder.add(library.Ramp(slope=-1.0, start_time=0.0))
        ramp_d_sol = builder.add(library.Ramp(slope=-0.5, start_time=0.0))

        mux = builder.add(library.Multiplexer(3))
        builder.connect(ramp.output_ports[0], mux.input_ports[0])
        builder.connect(chirp.output_ports[0], mux.input_ports[1])
        builder.connect(ramp_d.output_ports[0], mux.input_ports[2])

        rt = builder.add(library.RateLimiter(dt=dt, upper_limit=0.5, lower_limit=-0.5))
        builder.connect(mux.output_ports[0], rt.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "ramp": ramp.output_ports[0],
            "ramp_sol": ramp_sol.output_ports[0],
            "rt": rt.output_ports[0],
            "ramp_d": ramp_d.output_ports[0],
            "ramp_d_sol": ramp_d_sol.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )

        if show_plot:
            time = results.time
            fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

            ax1.plot(time, results.outputs["ramp"], label="ramp", marker="o")
            ax1.plot(time, results.outputs["rt"][:, 0], label="rt[0]", marker="x")
            ax1.plot(time, results.outputs["ramp_sol"], label="ramp_sol")
            ax1.grid(True)
            ax1.legend()

            ax2.plot(time, results.outputs["ramp_d"], label="ramp_d", marker="o")
            ax2.plot(time, results.outputs["rt"][:, 2], label="rt[2]", marker="x")
            ax2.plot(time, results.outputs["ramp_d_sol"], label="ramp_d_sol")
            ax2.grid(True)
            ax2.legend()

            plt.show()

        assert jnp.allclose(results.outputs["rt"][:, 0], results.outputs["ramp_sol"])
        assert jnp.allclose(results.outputs["rt"][:, 2], results.outputs["ramp_d_sol"])


if __name__ == "__main__":
    # TestTransferFunctionDiscrete().test_transfer_function_discrete(show_plot=True)
    # TestRateLimiter().test_rate_limiter_scalar(show_plot=True)
    # TestRateLimiter().test_rate_limiter_vec(show_plot=True)
    # TestDiscreteInitializer().test_discrete_initializer(show_plot=True)
    TestEdgeDetection().test_edge_detection(show_plot=True)
