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

"""Test blocks that apply nonlinearities/discontinuities to the signal.

Contains tests for:
- Comparator
- Dead
- MinMax
- Quantizer
- Relay
- Saturate
"""

import pytest

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

import collimator
from collimator import library
from collimator.framework.error import BlockParameterError

pytestmark = pytest.mark.minimal


class TestComparator:
    def test_ops(self):
        builder = collimator.DiagramBuilder()

        one = builder.add(library.Constant(1.0))
        two = builder.add(library.Constant(2.0))
        gt = builder.add(library.Comparator(operator=">"))
        gteq = builder.add(library.Comparator(operator=">="))
        lt = builder.add(library.Comparator(operator="<"))
        lteq = builder.add(library.Comparator(operator="<="))
        neq = builder.add(library.Comparator(operator="!="))
        eq = builder.add(library.Comparator(operator="=="))

        int_ = builder.add(library.Integrator(0.0))

        builder.connect(one.output_ports[0], gt.input_ports[0])
        builder.connect(two.output_ports[0], gt.input_ports[1])

        builder.connect(one.output_ports[0], gteq.input_ports[0])
        builder.connect(two.output_ports[0], gteq.input_ports[1])

        builder.connect(one.output_ports[0], lt.input_ports[0])
        builder.connect(two.output_ports[0], lt.input_ports[1])

        builder.connect(one.output_ports[0], lteq.input_ports[0])
        builder.connect(two.output_ports[0], lteq.input_ports[1])

        builder.connect(one.output_ports[0], neq.input_ports[0])
        builder.connect(two.output_ports[0], neq.input_ports[1])

        builder.connect(one.output_ports[0], eq.input_ports[0])
        builder.connect(one.output_ports[0], eq.input_ports[1])

        builder.connect(one.output_ports[0], int_.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "int_": int_.output_ports[0],
            "gt": gt.output_ports[0],
            "gteq": gteq.output_ports[0],
            "lt": lt.output_ports[0],
            "lteq": lteq.output_ports[0],
            "neq": neq.output_ports[0],
            "eq": eq.output_ports[0],
        }
        r = collimator.simulate(
            diagram, context, (0.0, 0.1), recorded_signals=recorded_signals
        )
        assert not r.outputs["gt"][-1]
        assert not r.outputs["gteq"][-1]
        assert r.outputs["lt"][-1]
        assert r.outputs["lteq"][-1]
        assert r.outputs["neq"][-1]
        assert r.outputs["eq"][-1]

    def test_invalid_input(self):
        builder = collimator.DiagramBuilder()

        with pytest.raises(BlockParameterError) as e:
            builder.add(library.Comparator(operator="="))
        # Success! The test failed as expected.
        print(e)
        assert "Valid options: >,>=,<,<=,==,!=" in str(e)

    def test_zc(self):
        builder = collimator.DiagramBuilder()

        zc_val = 1.2345678
        one = builder.add(library.Constant(zc_val))
        ramp = builder.add(library.Ramp(start_time=0.0))
        gt = builder.add(library.Comparator(operator=">"))

        int_ = builder.add(library.Integrator(0.0, enable_reset=True))

        builder.connect(ramp.output_ports[0], gt.input_ports[0])
        builder.connect(one.output_ports[0], gt.input_ports[1])

        builder.connect(one.output_ports[0], int_.input_ports[0])
        builder.connect(gt.output_ports[0], int_.input_ports[1])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "int_": int_.output_ports[0],
            "gt": gt.output_ports[0],
            "ramp": ramp.output_ports[0],
        }
        r = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        zc_idx = np.argmax(r.outputs["gt"])
        zc_time = r.time[zc_idx]
        print(f"zc_time={zc_time}")
        ramp_ = r.outputs["ramp"]
        print(f"ramp ={ramp_}")
        gt_ = r.outputs["gt"]
        print(f"gt ={gt_}")
        assert np.allclose(zc_val, zc_time)


class TestDeadZone:
    def test_invalid_input(self):
        builder = collimator.DiagramBuilder()

        with pytest.raises(BlockParameterError) as e:
            builder.add(library.DeadZone(half_range=-1.0, name="DeadZone"))
        # Success! The test failed as expected.
        print(e)
        assert (
            "DeadZone block DeadZone has invalid half_range -1.0. Must be > 0" in str(e)
        )

    def test_zc(self, show_plot=False):
        builder = collimator.DiagramBuilder()

        half_range = 0.2589
        start_value = -2.0
        ramp = builder.add(library.Ramp(start_time=0.0, start_value=start_value))
        dz = builder.add(library.DeadZone(half_range=half_range))
        int_ = builder.add(library.Integrator(0.0))

        builder.connect(ramp.output_ports[0], dz.input_ports[0])
        builder.connect(dz.output_ports[0], int_.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "int_": int_.output_ports[0],
            "dz": dz.output_ports[0],
            "ramp": ramp.output_ports[0],
        }
        r = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )

        time = r.time
        dz_sol = np.zeros_like(r.time)
        ramp_ = r.outputs["ramp"]
        dz_sol[ramp_ < -half_range] = ramp_[ramp_ < -half_range]
        dz_sol[ramp_ > half_range] = ramp_[ramp_ > half_range]

        dz_ = r.outputs["dz"]

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, ramp_, label="ramp_", marker="_")
            ax1.plot(time, dz_, label="dz_", marker="o")
            ax1.plot(time, dz_sol, label="dz_sol", marker=">")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert np.allclose(dz_sol, dz_)

        # for some reason the array of indices is the first element
        # so need to index into it twice to get the first occurence index.
        first_0_idx = np.where(dz_ == 0)[0][0]
        print(f"first_0_idx={first_0_idx}")
        zc_time = time[first_0_idx]
        zc_time_sol = start_value * -1 - half_range
        print(f"zc_time={zc_time}")
        print(f"zc_time_sol={zc_time_sol}")
        assert np.allclose(zc_time, zc_time_sol)


class TestMinMax:
    def test_ops(self):
        builder = collimator.DiagramBuilder()
        ud = builder.add(library.UnitDelay(dt=0.2, initial_state=0.0))
        one = builder.add(library.Constant(1.0))
        ramp = builder.add(library.Ramp(start_time=0.0))
        min2 = builder.add(library.MinMax(n_in=2, operator="min"))
        max2 = builder.add(library.MinMax(n_in=2, operator="max"))
        sine = builder.add(library.Sine(frequency=100))
        min3 = builder.add(library.MinMax(n_in=3, operator="min"))

        builder.connect(ramp.output_ports[0], ud.input_ports[0])

        builder.connect(one.output_ports[0], min2.input_ports[0])
        builder.connect(ramp.output_ports[0], min2.input_ports[1])

        builder.connect(one.output_ports[0], max2.input_ports[0])
        builder.connect(ramp.output_ports[0], max2.input_ports[1])

        builder.connect(one.output_ports[0], min3.input_ports[0])
        builder.connect(ramp.output_ports[0], min3.input_ports[1])
        builder.connect(sine.output_ports[0], min3.input_ports[2])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "one": one.output_ports[0],
            "ramp": ramp.output_ports[0],
            "sine": sine.output_ports[0],
            "min2": min2.output_ports[0],
            "max2": max2.output_ports[0],
            "min3": min3.output_ports[0],
        }
        r = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )

        ramp_ = r.outputs["ramp"]

        # solutions
        min2_sol = np.ones_like(r.time)
        min2_sol[ramp_ < 1.0] = ramp_[ramp_ < 1.0]
        max2_sol = np.ones_like(r.time)
        max2_sol[ramp_ > 1.0] = ramp_[ramp_ > 1.0]

        min3_sol = np.ones_like(r.time)
        for idx, t in enumerate(r.time):
            inputs = np.array([1.0, r.outputs["ramp"][idx], r.outputs["sine"][idx]])
            min3_sol[idx] = np.min(inputs)

        assert np.allclose(r.outputs["min2"], min2_sol)
        assert np.allclose(r.outputs["max2"], max2_sol)
        assert np.allclose(r.outputs["min3"], min3_sol)

    def test_invalid_input(self):
        with pytest.raises(BlockParameterError) as e:
            library.MinMax(n_in=2, operator="mini")
        # Success! The test failed as expected.
        print(e)
        assert "Valid options: max, min" in str(e)

    def test_zc(self):
        builder = collimator.DiagramBuilder()

        zc_val = 1.2345678
        one = builder.add(library.Constant(zc_val))
        ramp = builder.add(library.Ramp(start_time=0.0))
        max_ = builder.add(library.MinMax(n_in=2, operator="max"))

        int_ = builder.add(library.Integrator(0.0))

        builder.connect(ramp.output_ports[0], max_.input_ports[0])
        builder.connect(one.output_ports[0], max_.input_ports[1])

        builder.connect(max_.output_ports[0], int_.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "int_": int_.output_ports[0],
            "max_": max_.output_ports[0],
            "ramp": ramp.output_ports[0],
        }
        r = collimator.simulate(
            diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
        )
        zc_idx = np.argmin(np.abs(r.time - zc_val))
        zc_time = r.time[zc_idx]
        print(f"zc_time={zc_time}")
        ramp_ = r.outputs["ramp"]
        print(f"ramp ={ramp_}")
        max_ = r.outputs["max_"]
        print(f"max_ ={max_}")
        assert np.allclose(zc_val, zc_time)


class TestQuantizer:
    def evaluate_quantizer_output(self, input, interval):
        builder = collimator.DiagramBuilder()
        quantizer_block = builder.add(library.Quantizer(interval))
        input_block = builder.add(library.Constant(input))
        builder.connect(input_block.output_ports[0], quantizer_block.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        return quantizer_block.output_ports[0].eval(ctx)

    @pytest.mark.parametrize(
        "x,interval,expected",
        [
            (10, 3, 9.0),  # (int, int), quantizes down
            (10, 5, 10.0),  # (int, int), exact match
            (11, 3, 12.0),  # (int, int), quantizes up
            (-10, 3, -9.0),  # (-int, int), quantizes to lesser absolute value
            (-11, 3, -12.0),  # (-int, int),quantizes to greater absolute value
            (10.0, 3, 9.0),  # (float, int), quantizes down
            (10.0, 5, 10.0),  # (float, int), exact match
            (11.0, 3, 12.0),  # (float, int), quantizes up
            (-10.0, 3, -9.0),  # (-float, int), quantizes to lesser absolute value
            (-11.0, 3, -12.0),  # (-float, int),quantizes to greater absolute value
            (10.0, 3.0, 9.0),  # (float, float), quantizes down
            (10.0, 5.0, 10.0),  # (float, float), exact match
            (11.0, 3.0, 12.0),  # (float, float), quantizes up
            (-10.0, 3.0, -9.0),  # (-float, float), quantizes to lesser absolute value
            (-11.0, 3.0, -12.0),  # (-float, float),quantizes to greater absolute value
            (2.5, 0.3, 2.4),  # Non-integer interval, quantizes down
            (2.7, 0.3, 2.7),  # Non-integer interval, exact match
            (2.8, 0.3, 2.7),  # Non-integer interval, quantizes up
        ],
    )
    def test_scalar_inputs(self, x, interval, expected):
        assert jnp.allclose(
            self.evaluate_quantizer_output(x, interval), expected
        ), "Scalar quantizing does not match expected result."

    @pytest.mark.parametrize(
        "x,interval,expected",
        [
            (
                jnp.array([1.5, 2.5, 3.5]),
                1,
                jnp.array([2.0, 2.0, 4.0]),
            ),  # Array input, integer interval
            (
                jnp.array([1.5, 2.5, 3.5]),
                0.5,
                jnp.array([1.5, 2.5, 3.5]),
            ),  # Array input, non-integer interval
            (
                jnp.array([-1.5, -2.5, -3.5]),
                1.0,
                jnp.array([-2.0, -2.0, -4.0]),
            ),  # Array input with negative values
        ],
    )
    def test_array_inputs(self, x, interval, expected):
        assert jnp.allclose(
            self.evaluate_quantizer_output(x, interval), expected
        ), "Array quantizing does not match expected result."

    def test_zero_interval(self):
        x = 10.0
        interval = 0.0
        assert jnp.isnan(self.evaluate_quantizer_output(x, interval))


class TestRelay:
    def test_relay_zc(self, show_plot=False):
        # When the model has a continuous state, the zero-crossing of the
        # relay block should be localized in time.
        sim_stop_time = 2.0

        builder = collimator.DiagramBuilder()
        sine = builder.add(library.Sine(frequency=10))
        rly = builder.add(
            library.Relay(
                on_threshold=0.5,
                off_threshold=-0.5,
                on_value=1.0,
                off_value=0.0,
                initial_state=0.0,
            )
        )
        int0 = builder.add(library.Integrator(initial_state=0.0))

        builder.connect(sine.output_ports[0], rly.input_ports[0])
        builder.connect(rly.output_ports[0], int0.input_ports[0])

        recorded_signals = {
            "sine": sine.output_ports[0],
            "rly": rly.output_ports[0],
            "int0": int0.output_ports[0],
        }
        diagram = builder.build()
        context = diagram.create_context()
        options = collimator.SimulatorOptions(
            max_major_step_length=0.2,
            atol=1e-8,
            rtol=1e-6,
        )
        res = collimator.simulate(
            diagram,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
            options=options,
        )

        time = np.array(res.time)
        sine_ = np.array(res.outputs["sine"])
        rly_ = np.array(res.outputs["rly"])
        int0_ = np.array(res.outputs["int0"])
        rly_sol = np.zeros_like(time)
        # this pass condition is kind hacky. but it's necessary because of
        # the event occurred results sample recorded just before applying reset maps.
        dly_up = 0
        dly_down = 0
        for idx, val in enumerate(sine_):
            if val >= 0.5:
                dly_down = 0
                if dly_up < 1:
                    dly_up += 1
                    rly_sol[idx] = rly_sol[idx - 1]
                else:
                    rly_sol[idx] = 1.0
            elif val <= -0.5:
                dly_up = 0
                if dly_down < 1:
                    dly_down += 1
                    rly_sol[idx] = rly_sol[idx - 1]
                else:
                    rly_sol[idx] = 0.0
            elif idx > 0:
                rly_sol[idx] = rly_sol[idx - 1]

        print(f"time=\n{time}")
        print(f"sine_=\n{sine_}")
        print(f"int0_=\n{int0_}")
        print(f"rly_=\n{rly_}")
        print(f"rly_sol=\n{rly_sol}")
        print(f"err={rly_sol - rly_}")

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, sine_, label="sine_", marker="_")
            ax1.plot(time, rly_, label="rly_", marker="o")
            ax1.plot(time, int0_, label="int0_", marker=">")
            ax1.plot(time, rly_sol, label="rly_sol", marker="x")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert jnp.allclose(rly_, rly_sol)

    def test_relay_ft(self, show_plot=False):
        # If the model is purely feedthrough, the zero-crossing of the relay
        # will not be localized
        sim_stop_time = 2.0

        builder = collimator.DiagramBuilder()
        sine = builder.add(library.Sine(frequency=10))
        rly = builder.add(
            library.Relay(
                on_threshold=0.5,
                off_threshold=-0.5,
                on_value=1.0,
                off_value=0.0,
                initial_state=0.0,
            )
        )

        builder.connect(sine.output_ports[0], rly.input_ports[0])

        recorded_signals = {
            "sine": sine.output_ports[0],
            "rly": rly.output_ports[0],
        }
        diagram = builder.build()
        context = diagram.create_context()
        res = collimator.simulate(
            diagram,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
        )

        time = np.array(res.time)
        sine_ = np.array(res.outputs["sine"])
        rly_ = np.array(res.outputs["rly"])
        rly_sol = np.zeros_like(time)
        for idx, val in enumerate(sine_):
            if val >= 0.5:
                rly_sol[idx] = 1.0
            elif val <= -0.5:
                rly_sol[idx] = 0.0
            elif idx > 0:
                rly_sol[idx] = rly_sol[idx - 1]

        print(f"time=\n{time}")
        print(f"sine_=\n{sine_}")
        print(f"rly_=\n{rly_}")
        print(f"rly_sol=\n{rly_sol}")

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, sine_, label="sine_", marker="_")
            ax1.plot(time, rly_, label="rly_", marker="o")
            ax1.plot(time, rly_sol, label="rly_sol", marker="x")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert jnp.allclose(rly_, rly_sol)

    def test_relay_discrete(self, show_plot=False):
        # If the model has only discrete state, the zero-crossing will not
        # be localized.
        sim_stop_time = 2.0
        dt = 0.1

        builder = collimator.DiagramBuilder()
        sine = builder.add(library.Sine(frequency=10.0))
        builder.add(
            library.DiscreteClock(dt)
        )  # Add a discrete clock tick to the diagram
        rly = builder.add(
            library.Relay(
                on_threshold=0.5,
                off_threshold=-0.5,
                on_value=1.0,
                off_value=0.0,
                initial_state=0.0,
            )
        )
        # rly = builder.add(Gain(gain=2.0))

        builder.connect(sine.output_ports[0], rly.input_ports[0])

        recorded_signals = {
            "sine": sine.output_ports[0],
            "rly": rly.output_ports[0],
        }
        diagram = builder.build()
        context = diagram.create_context()
        res = collimator.simulate(
            diagram,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
        )

        time = np.array(res.time)
        sine_ = np.array(res.outputs["sine"])
        rly_ = np.array(res.outputs["rly"])
        rly_sol = np.zeros_like(time)
        for idx, val in enumerate(sine_):
            if val >= 0.5:
                rly_sol[idx] = 1.0
            elif val <= -0.5:
                rly_sol[idx] = 0.0
            elif idx > 0:
                rly_sol[idx] = rly_sol[idx - 1]

        print(f"time=\n{time}")
        print(f"sine_=\n{sine_}")
        print(f"rly_=\n{rly_}")
        print(f"rly_sol=\n{rly_sol}")

        if show_plot:
            fig02, (ax1) = plt.subplots(1, 1, figsize=(9, 12))
            ax1.plot(time, sine_, label="sine_", marker="_")
            ax1.plot(time, rly_, label="rly_", marker="o")
            ax1.plot(time, rly_sol, label="rly_sol", marker="x")
            ax1.grid(True)
            ax1.legend()
            plt.show()

        assert jnp.allclose(rly_, rly_sol)

    def test_relay_hybrid(self, show_plot=False):
        # If the model has both discrete and continuous states, the relay blocks
        # should follow the time mode of the input signal
        sim_stop_time = 2.0
        dt = 0.1

        builder = collimator.DiagramBuilder()
        sine = builder.add(library.Sine(frequency=10))
        zoh = builder.add(library.ZeroOrderHold(dt))  # Discretize the signal in time
        builder.connect(sine.output_ports[0], zoh.input_ports[0])

        # Add a high-frequency clock to clearly distinguish the cases
        builder.add(
            library.DiscreteClock(0.1 * dt)
        )  # Add a discrete clock tick to the diagram

        on_threshold = 0.5
        off_threshold = -0.5
        on_value = 1.0
        off_value = 0.0
        initial_state = 0.0

        # This relay should localize zero-crossings in time
        rly_zc = builder.add(
            library.Relay(
                on_threshold=on_threshold,
                off_threshold=off_threshold,
                on_value=on_value,
                off_value=off_value,
                initial_state=initial_state,
            )
        )

        # This relay will not localize zero-crossings in time
        rly_dt = builder.add(
            library.Relay(
                on_threshold=on_threshold,
                off_threshold=off_threshold,
                on_value=on_value,
                off_value=off_value,
                initial_state=initial_state,
            )
        )

        builder.connect(sine.output_ports[0], rly_zc.input_ports[0])
        builder.connect(zoh.output_ports[0], rly_dt.input_ports[0])

        # Add a continuous state to the system
        int0 = builder.add(library.Integrator(initial_state=0.0))
        builder.connect(rly_zc.output_ports[0], int0.input_ports[0])

        recorded_signals = {
            "sine": sine.output_ports[0],
            "sine_dt": zoh.output_ports[0],
            "rly_zc": rly_zc.output_ports[0],
            "rly_dt": rly_dt.output_ports[0],
        }
        diagram = builder.build()
        context = diagram.create_context()
        res = collimator.simulate(
            diagram,
            context,
            (0.0, sim_stop_time),
            recorded_signals=recorded_signals,
        )

        time = np.array(res.time)
        sine_ = np.array(res.outputs["sine"])
        rly_zc = np.array(res.outputs["rly_zc"])
        rly_zc_sol = np.zeros_like(time)
        # this pass condition is kind hacky. but it's necessary because of
        # the event occurred results sample recorded just before applying reset maps.
        dly_up = 0
        dly_down = 0
        for idx, val in enumerate(sine_):
            if val >= 0.5:
                dly_down = 0
                if dly_up < 1:
                    dly_up += 1
                    rly_zc_sol[idx] = rly_zc_sol[idx - 1]
                else:
                    rly_zc_sol[idx] = 1.0
            elif val <= -0.5:
                dly_up = 0
                if dly_down < 1:
                    dly_down += 1
                    rly_zc_sol[idx] = rly_zc_sol[idx - 1]
                else:
                    rly_zc_sol[idx] = 0.0
            elif idx > 0:
                rly_zc_sol[idx] = rly_zc_sol[idx - 1]

        sine_dt = np.array(res.outputs["sine_dt"])
        rly_dt = np.array(res.outputs["rly_dt"])
        rly_dt_sol = np.zeros_like(time)
        for idx, val in enumerate(sine_dt):
            if val >= 0.5:
                rly_dt_sol[idx] = 1.0
            elif val <= -0.5:
                rly_dt_sol[idx] = 0.0
            elif idx > 0:
                rly_dt_sol[idx] = rly_dt_sol[idx - 1]

        print(f"time=\n{time}")
        print(f"sine_=\n{sine_}")
        print(f"rly_zc=\n{rly_zc}")
        print(f"rly_zc_sol=\n{rly_zc_sol}")

        print(f"sine_=\n{sine_dt}")
        print(f"rly_dt=\n{rly_dt}")
        print(f"rly_dt_sol=\n{rly_dt_sol}")

        if show_plot:
            fig02, axs = plt.subplots(3, 1, figsize=(9, 12))
            axs[0].plot(time, sine_, label="sine", marker="_")
            axs[0].plot(time, rly_zc, label="rly_zc", marker="o")
            axs[0].plot(time, rly_zc_sol, label="rly_zc_sol", marker="x")
            axs[0].grid(True)
            axs[0].legend()
            axs[1].step(time, sine_dt, label="zoh", marker="_", where="post")
            axs[1].plot(time, rly_dt, label="rly_dt", marker="o")
            axs[1].plot(time, rly_dt_sol, label="rly_dt_sol", marker="x")
            axs[1].grid(True)
            axs[1].legend()
            axs[2].plot(time, sine_, label="sine", marker="_")
            axs[2].plot(time, rly_dt, label="rly_dt", marker="o")
            axs[2].plot(time, rly_zc, label="rly_zc", marker="x")
            axs[2].grid(True)
            axs[2].legend()
            plt.show()

        assert jnp.allclose(rly_zc, rly_zc_sol)
        assert jnp.allclose(rly_dt, rly_dt_sol)


class TestSaturate:
    def test_saturate_llim2_ulim2(self):
        builder = collimator.DiagramBuilder()

        slope = 100.0
        llim = 2.0
        ulim = 2.0
        saturate = builder.add(
            library.Saturate(lower_limit=llim, upper_limit=ulim, name="saturate")
        )
        ramp = builder.add(
            library.Ramp(start_value=0.0, slope=slope, start_time=0.0, name="ramp")
        )

        builder.connect(ramp.output_ports[0], saturate.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {"y": saturate.output_ports[0]}
        results = collimator.simulate(
            diagram, context, (0.0, 10.0), recorded_signals=recorded_signals
        )
        assert jnp.allclose(
            results.outputs["y"], jnp.clip(results.time * slope, llim, ulim)
        )

    def test_saturate_llim2_ulim8(self):
        builder = collimator.DiagramBuilder()

        slope = 100.0
        llim = 2.0
        ulim = 8.0
        saturate = builder.add(
            library.Saturate(lower_limit=llim, upper_limit=ulim, name="saturate")
        )
        ramp = builder.add(
            library.Ramp(start_value=0.0, slope=slope, start_time=0.0, name="ramp")
        )

        builder.connect(ramp.output_ports[0], saturate.input_ports[0])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {"y": saturate.output_ports[0]}
        results = collimator.simulate(
            diagram, context, (0.0, 10.0), recorded_signals=recorded_signals
        )
        assert jnp.allclose(
            results.outputs["y"], jnp.clip(results.time * slope, llim, ulim)
        )

    def test_saturate_llim0_ulimd(self):
        builder = collimator.DiagramBuilder()

        slope = 100.0
        llim = 0.0
        ulim = 1.0
        saturate = builder.add(
            library.Saturate(
                lower_limit=llim, enable_dynamic_upper_limit=True, name="saturate"
            )
        )
        ramp = builder.add(
            library.Ramp(start_value=0.0, slope=slope, start_time=0.0, name="ramp")
        )
        const = builder.add(library.Constant(value=ulim, name="const"))

        builder.connect(ramp.output_ports[0], saturate.input_ports[0])
        builder.connect(const.output_ports[0], saturate.input_ports[1])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {"y": saturate.output_ports[0]}
        results = collimator.simulate(
            diagram, context, (0.0, 10.0), recorded_signals=recorded_signals
        )
        assert jnp.allclose(
            results.outputs["y"], jnp.clip(results.time * slope, llim, ulim)
        )

    def test_saturate_llimd_ulim8(self):
        builder = collimator.DiagramBuilder()
        a = 10.0
        b = 100.0

        def _source_func(time):
            return a - b * time

        source = builder.add(library.SourceBlock(_source_func, name="source"))
        saturate = builder.add(
            library.Saturate(
                upper_limit=8.0, enable_dynamic_lower_limit=True, name="saturate"
            )
        )
        builder.connect(source.output_ports[0], saturate.input_ports[0])

        llimd = builder.add(
            library.Ramp(slope=1.0, start_value=-1.0, start_time=0.0, name="llimd")
        )
        builder.connect(llimd.output_ports[0], saturate.input_ports[1])

        diagram = builder.build()
        context = diagram.create_context()
        recorded_signals = {
            "y": saturate.output_ports[0],
            "llimd": llimd.output_ports[0],
        }
        results = collimator.simulate(
            diagram, context, (0.0, 10.0), recorded_signals=recorded_signals
        )

        assert jnp.allclose(
            results.outputs["y"],
            jnp.clip(a - b * results.time, results.outputs["llimd"], 8.0),
        )


if __name__ == "__main__":
    TestRelay().test_relay_zc(show_plot=False)
    TestRelay().test_relay_hybrid(show_plot=False)
    # TestComparator().test_ops()
    # TestComparator().test_invalid_input()
    # TestComparator().test_zc()
    # TestDeadZone().test_invalid_input()
    # TestDeadZone().test_zc(show_plot=False)
    # TestMinMax().test_ops()
    # TestMinMax().test_invalid_input()
    # TestMinMax().test_zc()
