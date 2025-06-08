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

"""Test source blocks that generate signals.

Contains tests for:
- Generic SourceBlock
- Chirp
- Clock (continuous and discrete)
- Constant
- DataSource
- Pulse
- RandomNumber
- Ramp
- Sawtooth
- Sine (and cosine, by phase delay)
- Step
"""

import pytest
import os
import numpy as np
from ..gen_datasource_csv import gen_files
import shutil
import collimator
from collimator import library
from collimator.backend import numpy_api as cnp

pytestmark = pytest.mark.minimal

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


# Test the generic SourceBlock
class TestSourceBlock:
    def test_trig(self):
        t = np.linspace(0.0, 10.0, 100)

        # Could also use pytest.mark.parameterize and a list of functions
        for func in [np.sin, np.cos, np.tan]:
            Source_0 = library.SourceBlock(func, name="Source_0")
            ctx = Source_0.create_context()

            ctx = ctx.with_time(t)
            y = Source_0.output_ports[0].eval(ctx)

            assert np.allclose(y, func(t))


class TestChirp:
    def test_chirp_simulation(self):
        sim_stop_time = 20.0
        dt = 0.05

        f0 = 0.01
        f1 = 1.0
        chirp_stop_time = 2.0

        system = library.Chirp(f0=f0, f1=f1, stop_time=chirp_stop_time, phi=0.0)

        context = system.create_context()
        recorded_signals = {"y": system.output_ports[0]}
        options = collimator.SimulatorOptions(
            max_major_step_length=dt,
        )
        results = collimator.simulate(
            system,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
            options=options,
        )

        # Exact solution
        f_t = f0 + (f1 - f0) * 0.5 * results.time / chirp_stop_time
        chirp_sol = np.cos(f_t * results.time)

        assert np.allclose(results.context.time, sim_stop_time)
        assert np.allclose(results.outputs["y"], chirp_sol)


class TestDataSource:
    def test_block(self):
        # create directory for temp files, and then create some csv files
        # in that directory
        abspath = os.path.abspath(__file__)
        library_dir, this_test_dir = os.path.split(abspath)
        this_test_dir = os.path.splitext(this_test_dir)[0]
        test_dir, this_test_parent_dir = os.path.split(library_dir)
        print(f"test_dir={test_dir}")
        workdir = os.path.join(
            test_dir, "workdir", this_test_parent_dir, this_test_dir, "DataSource"
        )
        print(f"workdir={workdir}")
        if os.path.exists(workdir):
            # os.rmdir(workdir)
            shutil.rmtree(workdir)
        os.makedirs(workdir)

        times, sw, times_t6, cw, filenames = gen_files(workdir, 10.0)

        # create DataSource blocks that ingest the CSV files created to verify
        # ingestion functionality.
        builder = collimator.DiagramBuilder()

        ds_f0_o1 = builder.add(
            library.DataSource(
                file_name=filenames[0],
                data_columns="1",
            )
        )
        ds_f0_o12 = builder.add(
            library.DataSource(
                file_name=filenames[0],
                data_columns="1,2",
            )
        )
        ds_f0_oAll = builder.add(
            library.DataSource(
                file_name=filenames[0],
                data_columns="0:3",
            )
        )
        ds_f1_o2 = builder.add(
            library.DataSource(
                file_name=filenames[1],
                time_samples_as_column=True,
                data_columns="2",
            )
        )
        ds_f1_oAll = builder.add(
            library.DataSource(
                file_name=filenames[1],
                time_samples_as_column=True,
                data_columns="1:4",
            )
        )
        builder.add(
            library.DataSource(
                file_name=filenames[2],
                header_as_first_row=True,
                data_columns="1",
            )
        )
        builder.add(
            library.DataSource(
                file_name=filenames[3],
                header_as_first_row=True,
                time_samples_as_column=True,
                data_columns="2",
            )
        )
        builder.add(
            library.DataSource(
                file_name=filenames[4],
                time_samples_as_column=True,
                time_column="1",
                data_columns="3",
            )
        )
        builder.add(
            library.DataSource(
                file_name=filenames[5],
                header_as_first_row=True,
                time_samples_as_column=True,
                time_column="3",
            )
        )
        builder.add(
            library.DataSource(
                file_name=filenames[6],
                time_samples_as_column=True,
                data_columns="1",
            )
        )
        builder.add(
            library.DataSource(
                file_name=filenames[7],
                header_as_first_row=True,
                time_samples_as_column=True,
                time_column="1",
                data_columns="2",  # FIXME why cant this be "1"
            )
        )

        diagram = builder.build()
        context = diagram.create_context()

        ds_f0_o1_ = ds_f0_o1.output_ports[0].eval(context)
        assert ds_f0_o1_.shape == ()
        print(f"ds_f0_o1_={ds_f0_o1_} shape={ds_f0_o1_.shape}")

        ds_f0_o12_ = ds_f0_o12.output_ports[0].eval(context)
        assert ds_f0_o12_.shape == (2,)
        print(f"ds_f0_o12_={ds_f0_o12_} shape={ds_f0_o12_.shape}")

        ds_f0_oAll_ = ds_f0_oAll.output_ports[0].eval(context)
        assert ds_f0_oAll_.shape == (3,)
        print(f"ds_f0_oAll_={ds_f0_oAll_} shape={ds_f0_oAll_.shape}")

        ds_f1_o2_ = ds_f1_o2.output_ports[0].eval(context)
        assert ds_f1_o2_.shape == ()
        print(f"ds_f1_o2_={ds_f1_o2_} shape={ds_f1_o2_.shape}")

        ds_f1_oAll_ = ds_f1_oAll.output_ports[0].eval(context)
        assert ds_f1_oAll_.shape == (3,)
        print(f"ds_f1_oAll_={ds_f1_oAll_} shape={ds_f1_oAll_.shape}")


class TestClock:
    def test_clock_simulation(self):
        builder = collimator.DiagramBuilder()
        clock = builder.add(library.Clock())
        integrator = builder.add(library.Integrator(0.0))
        builder.connect(clock.output_ports[0], integrator.input_ports[0])
        diagram = builder.build()

        context = diagram.create_context()
        diagram.check_types(context)

        results = collimator.simulate(
            diagram,
            context,
            (0.0, 1.0),
            recorded_signals={"clock": clock.output_ports[0]},
        )

        np.testing.assert_equal(np.asarray(results.outputs["clock"][0]), 0.0)
        np.testing.assert_equal(np.asarray(results.outputs["clock"][-1]), 1.0)

    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_clock_continuous(self, dtype):
        t = np.linspace(0.0, 10.0, 100)
        clock = library.Clock(name="Clock_0", dtype=dtype)

        # Pass in the initial time to get the right data type.
        ctx = clock.create_context()

        ctx = ctx.with_time(t)
        x = clock.output_ports[0].eval(ctx)

        assert np.allclose(x, t.astype(dtype))
        assert isinstance(x, cnp.ndarray)
        assert x.shape == t.shape
        assert x.dtype == dtype

    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_clock_discrete(self, dtype):
        dt = 1.0
        t = np.linspace(0.0, 10.0, 10)
        clock = library.DiscreteClock(dt, dtype=dtype, name="Clock_0")
        ctx = clock.create_context()

        recorded_signals = {"y": clock.output_ports[0]}
        results = collimator.simulate(
            clock, ctx, (t[0], t[-1]), recorded_signals=recorded_signals
        )

        t = results.time
        x = results.outputs["y"]
        assert np.allclose(x, t.astype(dtype))
        assert x.shape == t.shape
        assert x.dtype == dtype


class TestConstant:
    def test_float(self, x=3.0):
        Constant_0 = library.Constant(x)
        ctx = Constant_0.create_context()
        val = Constant_0.output_ports[0].eval(ctx)
        assert val == x

        assert val.dtype == np.float64

    def test_constant_int(self, x=3):
        Constant_0 = library.Constant(x)
        ctx = Constant_0.create_context()
        val = Constant_0.output_ports[0].eval(ctx)

        assert val == x
        assert val.dtype == np.int64

    def test_bool(self, x=True):
        Constant_0 = library.Constant(x)
        ctx = Constant_0.create_context()
        val = Constant_0.output_ports[0].eval(ctx)
        assert val == x

        assert val.dtype == np.bool_

    @pytest.mark.parametrize("dtype", [*float_dtypes, *int_dtypes])
    def test_array(self, dtype, x=np.array([1, 2, 3])):
        Constant_0 = library.Constant(np.asarray(x, dtype=dtype))
        ctx = Constant_0.create_context()
        val = Constant_0.output_ports[0].eval(ctx)
        assert np.allclose(val, x)
        assert isinstance(val, cnp.ndarray)
        assert val.shape == x.shape
        assert val.dtype == dtype

    def test_array_bool(self, x=np.array([True, False, True])):
        Constant_0 = library.Constant(x)
        ctx = Constant_0.create_context()
        val = Constant_0.output_ports[0].eval(ctx)
        assert np.allclose(val, x)
        assert isinstance(val, cnp.ndarray)
        assert val.shape == x.shape
        assert val.dtype == np.bool_


class TestPulse:
    def _run_pulse_test(self, amplitude, period, pulse_width):
        pulse = library.Pulse(
            amplitude=amplitude,
            period=period,
            pulse_width=pulse_width,
            name="pulse",
        )

        context = pulse.create_context()
        recorded_signals = {"y": pulse.output_ports[0]}

        dt = 0.01
        sim_stop_time = 2.0
        options = collimator.SimulatorOptions(
            max_major_step_length=dt,
        )
        results = collimator.simulate(
            pulse,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
            options=options,
        )
        return results.time, results.outputs["y"]

    def _check_pulse(
        self,
        time,
        pulse,
        amplitude,
        period,
        pulse_width,
    ):
        tol = period * pulse_width * 1e-6
        tol_amplitude = amplitude * 1e-6
        # get pulse rising and falling edges
        pulse_diff = np.diff(pulse)
        rising_edge_idx = np.argwhere(pulse_diff > 0)
        falling_edge_idx = np.argwhere(pulse_diff < 0)

        # if these lists are empty, the test failed because the test expects all pulse signals to have many edges in the test duration.
        assert len(rising_edge_idx) > 0
        assert len(rising_edge_idx) > 0

        rising_edge_idx = rising_edge_idx + 1
        falling_edge_idx = falling_edge_idx + 1
        print(f"[pulse_error] tol={tol}")
        print(f"pulse={pulse}")
        print(f"pulse_diff={pulse_diff}")
        print(f"time={time}")
        print(f"rising_edge_idx={rising_edge_idx}")
        print(f"falling_edge_idx={falling_edge_idx}")
        # get pulse rising and falling edge times
        rising_times = time[rising_edge_idx]
        falling_times = time[falling_edge_idx]
        print(f"rising_times={rising_times}")
        print(f"falling_times={falling_times}")

        # check that all rising edge times are close to integer multiples of the period
        for idx in range(len(rising_times)):
            rise_time = rising_times[idx]
            rising_mod = np.fmod(rise_time, period)
            rise_time_incorrect = (np.abs(rising_mod) >= tol) and (
                np.abs(rising_mod - period) >= tol
            )
            if rise_time_incorrect:
                print(f"plus idx={idx},rise_time={rise_time},rising_mod={rising_mod}")
            assert not rise_time_incorrect

        # check that all falling edge times, minus the period*pulse_width,  are close to integer multiples of the period
        falling_times_offset = falling_times - (period * pulse_width)
        for idx in range(len(falling_times_offset)):
            falling_time_offset = falling_times_offset[idx]
            falling_mod = np.fmod(falling_time_offset, period)
            fall_time_incorrect = (np.abs(falling_mod) >= tol) and (
                np.abs(falling_mod - period) >= tol
            )
            if fall_time_incorrect:
                print(f"plus idx={idx},rise_time={rise_time},rising_mod={rising_mod}")
            assert not fall_time_incorrect

        amplitude_measured = np.max(pulse)
        assert np.abs(amplitude_measured - amplitude) < tol_amplitude

    def test_pulse_a1_pw05_p01(self):
        t, y = self._run_pulse_test(amplitude=1.0, period=0.1, pulse_width=0.5)
        self._check_pulse(t, y, amplitude=1.0, period=0.1, pulse_width=0.5)

    def test_pulse_a1e20_pw05_p01(self):
        t, y = self._run_pulse_test(amplitude=1e20, period=0.1, pulse_width=0.5)
        self._check_pulse(t, y, amplitude=1e20, period=0.1, pulse_width=0.5)

    def test_pulse_a1_pw01_p01(self):
        t, y = self._run_pulse_test(amplitude=1.0, period=0.1, pulse_width=0.1)
        self._check_pulse(t, y, amplitude=1.0, period=0.1, pulse_width=0.1)

    def test_pulse_a1_pw05_p02(self):
        t, y = self._run_pulse_test(amplitude=1.0, period=0.2, pulse_width=0.5)
        self._check_pulse(t, y, amplitude=1.0, period=0.2, pulse_width=0.5)


class TestRandomNumber:
    def _run_random_test(
        self, dt, tf, distribution, distribution_parameters, seed=None
    ):
        system = library.RandomNumber(
            dt=dt,
            distribution=distribution,
            seed=seed,
            **distribution_parameters,
        )

        context = system.create_context()
        recorded_signals = {
            "x": system.output_ports[0],
        }

        result = collimator.simulate(
            system,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
        )

        x = result.outputs["x"]
        t = result.time
        return t, x

    def test_random_normal(self):
        dt = 0.1
        tf = 100.0
        shape = (2,)
        dtype = np.float64
        seed = 42
        t1, x1 = self._run_random_test(
            dt=dt,
            tf=tf,
            distribution="normal",
            distribution_parameters={"shape": shape, "dtype": dtype},
            seed=seed,
        )

        print(t1, t1.shape, x1.shape)

        assert x1.shape == (int(tf / dt) + 1, 2)

        # Check uncorrelated in time
        assert np.allclose(np.corrcoef(x1[1:, 0], x1[:-1, 0]), np.eye(2), atol=1e-1)

        # Check covariance
        assert np.allclose(np.cov(x1.T), np.eye(2), atol=1e-1)

        # Check mean
        assert np.allclose(np.mean(x1, axis=0), np.zeros(2), atol=1e-1)

        # Check that the random numbers are deterministic with the same seed
        _, x2 = self._run_random_test(
            dt=dt,
            tf=tf,
            distribution="normal",
            distribution_parameters={"shape": shape, "dtype": dtype},
            seed=seed,
        )

        assert np.allclose(x1, x2)

        # Check that the random numbers are uncorrelated with a different seed
        _, x3 = self._run_random_test(
            dt=dt,
            tf=tf,
            distribution="normal",
            distribution_parameters={"shape": shape, "dtype": dtype},
            seed=seed + 1,
        )

        assert np.allclose(np.corrcoef(x1.T, x3.T), np.eye(4), atol=1e-1)

    def test_random_gamma(self):
        # Since the parameters are just passed to JAX, no need to test every
        # distribution in detail.  Just make sure that the distribution parameters
        # are passed correctly for one distribution
        dt = 0.1
        tf = 100.0
        alpha = 2.5
        distribution_parameters = {
            "shape": (),
            "dtype": np.float64,
            "a": alpha,
        }
        seed = 42
        _, x = self._run_random_test(
            dt=dt,
            tf=tf,
            distribution="gamma",
            distribution_parameters=distribution_parameters,
            seed=seed,
        )

        # Check mean is close to alpha
        assert np.allclose(np.mean(x), alpha, atol=1e-1)

        # Check variance is close to alpha
        assert np.allclose(np.var(x), alpha, atol=1e-1)

        # Check that the numbers are uncorrelated in time
        assert np.allclose(np.corrcoef(x[1:], x[:-1]), np.eye(2), atol=1e-1)


class TestRamp:
    def _run_ramp_test(self, slope, start_value, start_time):
        ramp = library.Ramp(
            start_value=start_value,
            slope=slope,
            start_time=start_time,
            name="ramp",
        )

        context = ramp.create_context()
        recorded_signals = {"y": ramp.output_ports[0]}

        dt = 0.1
        sim_stop_time = 10.0
        options = collimator.SimulatorOptions(
            max_major_step_length=dt,
        )
        results = collimator.simulate(
            ramp,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
            options=options,
        )
        return results.time, results.outputs["y"]

    def test_ramp_time(self):
        t, y = self._run_ramp_test(slope=1.0, start_value=0.0, start_time=0.0)
        assert np.allclose(t, y)

    def test_ramp_const(self):
        t, y = self._run_ramp_test(slope=0.0, start_value=0.0, start_time=0.0)
        assert np.allclose(y, 0.0)

    def test_ramp_delay(self):
        t, y = self._run_ramp_test(slope=1.0, start_value=0.0, start_time=0.05)
        assert np.allclose(y, np.clip(t - 0.05, 0.0, np.inf))

    def test_ramp_early(self):
        t, y = self._run_ramp_test(slope=1.0, start_value=0.0, start_time=-0.05)
        assert np.allclose(y, t + 0.05)

    def test_ramp_time_neg(self):
        t, y = self._run_ramp_test(slope=-1.0, start_value=0.0, start_time=0.0)
        assert np.allclose(y, -1.0 * t)

    def test_ramp_1e20(self):
        t, y = self._run_ramp_test(slope=1e20, start_value=0.0, start_time=0.0)
        assert np.allclose(y, t * 1e20, atol=3e3)

    def test_ramp_1eNeg20(self):
        t, y = self._run_ramp_test(slope=1e-20, start_value=0.0, start_time=0.0)
        assert np.allclose(y, t * 1e-20)

    def test_ramp_ic_1(self):
        t, y = self._run_ramp_test(slope=1.0, start_value=1.0, start_time=0.0)
        assert np.allclose(y, t + 1.0)

    def test_ramp_ic_neg1(self):
        t, y = self._run_ramp_test(slope=1.0, start_value=-1.0, start_time=0.0)
        assert np.allclose(y, t - 1.0)

    def test_ramp_delay_ic_1(self):
        t, y = self._run_ramp_test(slope=1.0, start_value=1.0, start_time=0.05)
        assert np.allclose(y, np.clip(t - 0.05, 0.0, np.inf) + 1.0)


class TestSawtooth:
    def _sawtooth(self, time, freq=1.0, amp=1.0, phase=0.0):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
        eps = 2 * np.finfo(np.asarray(time).dtype).eps
        return np.mod((time - phase) + eps, (1.0 / freq)) * amp

    def _run_sawtooth_test(self, amplitude, frequency, phase_delay):
        sawtooth = library.Sawtooth(
            amplitude=amplitude,
            frequency=frequency,
            phase_delay=phase_delay,
            name="sawtooth",
        )

        context = sawtooth.create_context()
        recorded_signals = {"y": sawtooth.output_ports[0]}

        dt = 0.1
        sim_stop_time = 10.0
        options = collimator.SimulatorOptions(
            max_major_step_length=dt,
        )
        results = collimator.simulate(
            sawtooth,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
            options=options,
        )
        return results.time, results.outputs["y"]

    def test_sawtooth_f1a1p0(self):
        t, y = self._run_sawtooth_test(amplitude=1.0, frequency=1.0, phase_delay=0.0)
        y_ex = self._sawtooth(t, freq=1.0, amp=1.0, phase=0.0)
        assert np.allclose(y, y_ex)

    def test_sawtooth_f10a1p0(self):
        t, y = self._run_sawtooth_test(amplitude=1.0, frequency=10.0, phase_delay=0.0)
        y_ex = self._sawtooth(t, freq=10.0, amp=1.0, phase=0.0)
        print(y - y_ex)
        assert np.allclose(y, y_ex)

    def test_sawtooth_f1a2p0(self):
        t, y = self._run_sawtooth_test(amplitude=2.0, frequency=1.0, phase_delay=0.0)
        y_ex = self._sawtooth(t, freq=1.0, amp=2.0, phase=0.0)
        assert np.allclose(y, y_ex)

    def test_sawtooth_f1aNeg1p0(self):
        t, y = self._run_sawtooth_test(amplitude=-1.0, frequency=1.0, phase_delay=0.0)
        y_ex = self._sawtooth(t, freq=1.0, amp=-1.0, phase=0.0)
        assert np.allclose(y, y_ex)

    def test_sawtooth_f1a1p0p1(self):
        t, y = self._run_sawtooth_test(amplitude=1.0, frequency=1.0, phase_delay=0.1)
        y_ex = self._sawtooth(t, freq=1.0, amp=1.0, phase=0.1)
        assert np.allclose(y, y_ex)

    def test_sawtooth_f1a1p0pNeg0p1(self):
        t, y = self._run_sawtooth_test(amplitude=1.0, frequency=1.0, phase_delay=-0.1)
        y_ex = self._sawtooth(t, freq=1.0, amp=1.0, phase=-0.1)
        assert np.allclose(y, y_ex)


class TestSine:
    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_sin(self, dtype):
        t = np.linspace(0.0, 10.0, 100, dtype=dtype)
        Sine_0 = library.Sine(name="Sine_0")
        ctx = Sine_0.create_context(time=t[0])

        ctx = ctx.with_time(t)
        x = Sine_0.output_ports[0].eval(ctx)

        assert np.allclose(x, np.sin(t))
        assert isinstance(x, cnp.ndarray)
        assert x.shape == t.shape
        assert x.dtype == dtype

    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_sin_with_parameters(self, dtype, A=3, f=2):
        t = np.linspace(0.0, 10.0, 100, dtype=dtype)
        Sine_0 = library.Sine(amplitude=A, frequency=f, name="Sine_0")
        ctx = Sine_0.create_context(time=t[0])

        ctx = ctx.with_time(t)
        x = Sine_0.output_ports[0].eval(ctx)

        assert np.allclose(x, A * np.sin(f * t))
        assert isinstance(x, cnp.ndarray)
        assert x.shape == t.shape
        assert x.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_cos(self, dtype):
        t = np.linspace(0.0, 10.0, 100, dtype=dtype)
        Cosine_0 = library.Sine(phase=np.pi / 2, name="Cosine_0")
        ctx = Cosine_0.create_context(time=t[0])

        ctx = ctx.with_time(t)
        x = Cosine_0.output_ports[0].eval(ctx)

        assert np.allclose(x, np.cos(t), atol=1e-6)
        assert isinstance(x, cnp.ndarray)
        assert x.shape == t.shape
        assert x.dtype == dtype


class TestStep:
    def _run_step_test(self, start_value, end_value, step_time):
        step = library.Step(
            start_value=start_value,
            end_value=end_value,
            step_time=step_time,
            name="step",
        )

        context = step.create_context()

        recorded_signals = {"y": step.output_ports[0]}
        results = collimator.simulate(
            step, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        return results.time, results.outputs["y"]

    def test_start0_end0_time0(self):
        t, y = self._run_step_test(start_value=0.0, end_value=0.0, step_time=0.0)
        assert np.allclose(y, 0.0)

    def test_start1_end0_time0(self):
        t, y = self._run_step_test(start_value=1.0, end_value=0.0, step_time=0.0)
        assert np.allclose(y, 0.0)

    def test_start0_end1_time0(self):
        t, y = self._run_step_test(start_value=0.0, end_value=1.0, step_time=0.0)
        assert np.allclose(y, 1.0)

    def test_start0_end0_time1(self):
        t, y = self._run_step_test(start_value=0.0, end_value=0.0, step_time=1.0)
        assert np.allclose(y, 0.0)

    def test_start1_end1_time0(self):
        t, y = self._run_step_test(start_value=1.0, end_value=1.0, step_time=0.0)
        assert np.allclose(y, 1.0)

    def test_start1_end1_time1(self):
        t, y = self._run_step_test(start_value=1.0, end_value=1.0, step_time=1.0)
        assert np.allclose(y, 1.0)

    def test_startNeg1_end0_time0(self):
        t, y = self._run_step_test(start_value=-1.0, end_value=0.0, step_time=0.0)
        assert np.allclose(y, 0.0)

    def test_start0_endNeg1_time0(self):
        t, y = self._run_step_test(start_value=0.0, end_value=-1.0, step_time=0.0)
        assert np.allclose(y, -1.0)

    def test_start0_end0_timeNeg1(self):
        t, y = self._run_step_test(start_value=0.0, end_value=0.0, step_time=-1.0)
        assert np.allclose(y, 0.0)

    def test_startNeg1_endNeg1_time0(self):
        t, y = self._run_step_test(start_value=-1.0, end_value=-1.0, step_time=0.0)
        assert np.allclose(y, -1.0)

    def test_startNeg1_end0_timeNeg1(self):
        t, y = self._run_step_test(start_value=-1.0, end_value=0.0, step_time=-1.0)
        assert np.allclose(y, 0.0)

    def test_start0_endNeg1_timeNeg1(self):
        t, y = self._run_step_test(start_value=0.0, end_value=-1.0, step_time=-1.0)
        assert np.allclose(y, -1.0)

    def test_startNeg1_endNeg1_timeNeg1(self):
        t, y = self._run_step_test(start_value=-1.0, end_value=-1.0, step_time=-1.0)
        assert np.allclose(y, -1.0)

    def test_start0_end1_time1(self):
        t, y = self._run_step_test(start_value=0.0, end_value=1.0, step_time=1.0)
        y_ex = np.where(t >= 1.0, 1.0, 0.0)
        assert np.allclose(y, y_ex)


class TestWhiteNoise:
    def test_white_noise_scalar(self):
        dt = 0.1
        fs = 10 / dt  # Sample 10x per update
        power = 0.2  # Arbitrary desired PSD power
        system = library.WhiteNoise(
            correlation_time=dt,
            noise_power=power,
            num_samples=10,
            seed=42,
        )

        context = system.create_context()

        recorded_signals = {"y": system.output_ports[0]}
        tf = 100.0
        options = collimator.SimulatorOptions(
            max_major_step_length=1 / fs,
            buffer_length=int(tf * fs) + 1,
        )
        results = collimator.simulate(
            system,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
            options=options,
        )

        y = results.outputs["y"]

        # Check mean and variance
        mean = np.mean(y)
        assert np.allclose(mean, 0.0, atol=1e-1)
        assert np.allclose(np.var(y), power, atol=0.02)

        # Check the correlation function
        N = len(y)
        acv = np.correlate(y - mean, y - mean, mode="full") / N
        rho = acv[N - 1 :] / acv[N - 1]

        assert rho[0] == 1.0

        # After 1 sample (10% of dt), the correlation should be reasonably high
        assert rho[1] > 0.9

        # After 10 samples (100% of dt), the correlation should be close to 0
        assert rho[10] < 0.1

    def test_white_noise_vector(self):
        dt = 0.1
        fs = 10 / dt  # Sample 10x per update
        system = library.WhiteNoise(
            correlation_time=dt,
            num_samples=10,
            seed=42,
            shape=(2,),
        )

        context = system.create_context()

        recorded_signals = {"y": system.output_ports[0]}
        tf = 100.0
        options = collimator.SimulatorOptions(
            max_major_step_length=1 / fs,
            buffer_length=int(tf * fs) + 1,
        )
        results = collimator.simulate(
            system,
            context,
            (0.0, tf),
            recorded_signals=recorded_signals,
            options=options,
        )

        # Check mean and covariance of the signal
        y = results.outputs["y"]
        assert np.allclose(np.mean(y), 0.0, atol=1e-1)
        assert np.allclose(np.cov(y.T), np.eye(2), atol=1e-1)


if __name__ == "__main__":
    TestDataSource().test_block()
