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
import jax.numpy as jnp
import matplotlib.pyplot as plt
import collimator
from collimator.library import (
    Integrator,
    Sine,
    Gain,
    DiscreteClock,
    UnitDelay,
)

from collimator.logging import logger
from collimator.simulation import SimulatorOptions, ResultsOptions, ResultsMode


pytestmark = pytest.mark.minimal


class TestContinuousTime:
    @pytest.mark.parametrize("enable_tracing", [True, False])
    def test_leaf_system(self, enable_tracing, dtype=jnp.float64):
        a = 1.5

        class ScalarLinear(collimator.LeafSystem):
            def __init__(self):
                super().__init__(name="ScalarLinear")
                self.declare_continuous_state(shape=(), ode=self.ode, dtype=dtype)

            def ode(self, time, state):
                xc = state.continuous_state
                return -a * xc

        model = ScalarLinear()
        x0 = dtype(2.0)
        t0 = dtype(0.0)
        ctx = model.create_context(time=t0)
        ctx = ctx.with_continuous_state(x0)

        tf = 1.5

        options = collimator.SimulatorOptions(
            rtol=1e-6,
            atol=1e-8,
            enable_tracing=enable_tracing,
        )
        results = collimator.simulate(model, ctx, (0.0, tf), options=options)
        ctx = results.context

        assert ctx.time == tf

        x = ctx.continuous_state
        assert jnp.allclose(x, x0 * jnp.exp(-a * tf), rtol=1e-4, atol=1e-6)

    def test_flat_diagram(self, dtype=jnp.float64):
        # Continuous-time integration with double integrator model
        # from primitives

        builder = collimator.DiagramBuilder()
        Sin_0 = builder.add(Sine(name="Sin_0"))

        v0 = dtype(-1.0)
        x0 = dtype(0.0)

        Integrator_0 = builder.add(Integrator(v0, name="Integrator_0"))  # v
        Integrator_1 = builder.add(Integrator(x0, name="Integrator_1"))  # x

        builder.connect(Sin_0.output_ports[0], Integrator_0.input_ports[0])
        builder.connect(Integrator_0.output_ports[0], Integrator_1.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        tf = 10.0
        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        results = collimator.simulate(diagram, ctx, (0.0, tf), options=options)
        ctx = results.context
        logger.debug(ctx.state)

        assert ctx.time == tf
        x = ctx[Integrator_1.system_id].continuous_state
        v = ctx[Integrator_0.system_id].continuous_state

        assert jnp.allclose(x, -jnp.sin(tf), rtol=1e-4, atol=1e-6)
        assert jnp.allclose(v, -jnp.cos(tf), rtol=1e-4, atol=1e-6)

    def test_flat_diagram_full_output(self, dtype=jnp.float64):
        builder = collimator.DiagramBuilder()
        Sin_0 = builder.add(Sine(name="Sin_0"))

        v0 = dtype(-1.0)
        x0 = dtype(0.0)

        Integrator_0 = builder.add(Integrator(v0, name="Integrator_0"))  # v
        Integrator_1 = builder.add(Integrator(x0, name="Integrator_1"))  # x

        builder.connect(Sin_0.output_ports[0], Integrator_0.input_ports[0])
        builder.connect(Integrator_0.output_ports[0], Integrator_1.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        tf = 10.0
        recorded_signals = {
            "Integrator_0": Integrator_0.output_ports[0],
            "Integrator_1": Integrator_1.output_ports[0],
        }
        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        results = collimator.simulate(
            diagram, ctx, (0.0, tf), recorded_signals=recorded_signals, options=options
        )

        ctx = results.context
        assert ctx.time == tf
        x = ctx[Integrator_1.system_id].continuous_state
        v = ctx[Integrator_0.system_id].continuous_state

        assert jnp.allclose(x, -jnp.sin(tf), rtol=1e-4, atol=1e-6)
        assert jnp.allclose(v, -jnp.cos(tf), rtol=1e-4, atol=1e-6)

        ts = results.time
        xs = results.outputs["Integrator_1"]
        vs = results.outputs["Integrator_0"]
        assert jnp.allclose(xs, -jnp.sin(ts), rtol=1e-4, atol=1e-6)
        assert jnp.allclose(vs, -jnp.cos(ts), rtol=1e-4, atol=1e-6)

    def test_nested_diagram(self, dtype=jnp.float64):
        builder = collimator.DiagramBuilder()

        v0 = dtype(-1.0)
        x0 = dtype(0.0)

        Integrator_0 = builder.add(Integrator(v0, name="Integrator_0"))  # v
        Integrator_1 = builder.add(Integrator(x0, name="Integrator_1"))  # x

        builder.connect(Integrator_0.output_ports[0], Integrator_1.input_ports[0])
        builder.export_input(Integrator_0.input_ports[0], "plant:input")

        plant = builder.build(name="plant")

        builder = collimator.DiagramBuilder()

        Sin_0 = builder.add(Sine(name="Sin_0"))
        builder.add(plant)
        builder.connect(Sin_0.output_ports[0], plant.input_ports[0])

        diagram = builder.build()
        ctx = diagram.create_context()

        tf = 10.0
        options = collimator.SimulatorOptions(rtol=1e-6, atol=1e-8)
        results = collimator.simulate(diagram, ctx, (0.0, tf), options=options)
        ctx = results.context
        logger.debug(ctx.state)

        assert ctx.time == tf
        x = ctx[Integrator_1.system_id].continuous_state
        v = ctx[Integrator_0.system_id].continuous_state

        assert jnp.allclose(x, -jnp.sin(tf), rtol=1e-4, atol=1e-6)
        assert jnp.allclose(v, -jnp.cos(tf), rtol=1e-4, atol=1e-6)


class SimpleDiscreteTimeSystem(collimator.LeafSystem):
    def __init__(self, x0=0.0, period=1.0):
        super().__init__(name="plant")

        self.declare_discrete_state(default_value=x0)

        self.declare_output_port(
            self.output,
            period=period,
            offset=0.0,
            name="y",
        )
        self.declare_periodic_update(
            self.update,
            period=period,
            offset=0.0,
        )

    # x[n+1] = x^3[n]
    def update(self, time, state, *inputs):
        x = state.discrete_state
        return x**3

    # y[n] = x[n]
    def output(self, time, state, *inputs):
        return state.discrete_state


class TestDiscreteTime:
    # See also models/test_discrete_time.py

    def test_advance(self):
        # Instantiate the System
        model = SimpleDiscreteTimeSystem()

        # Create a context for this system
        context = model.create_context()
        xd0 = 0.9
        context = context.with_discrete_state(xd0)

        # Create a simulator
        options = collimator.SimulatorOptions(max_major_steps=100)
        simulator = collimator.Simulator(model, options=options)
        sim_state = simulator.initialize(context)
        context = sim_state.context

        assert context.time == 0.0

        # Advance the simulator
        sim_state = simulator.advance_to(1.5, context)
        context = sim_state.context

        xd = context.discrete_state
        assert xd == (xd0**3) ** 3

    def test_restart(self):
        # Instantiate the System
        model = SimpleDiscreteTimeSystem()

        # Create a context for this system
        context = model.create_context()
        xd0 = 0.9
        context = context.with_discrete_state(xd0)

        # Create a simulator
        options = collimator.SimulatorOptions(max_major_steps=100)
        simulator = collimator.Simulator(model, options=options)
        sim_state = simulator.initialize(context)
        context = sim_state.context

        assert context.time == 0.0

        # Advance the simulator to just before the second update
        sim_state = simulator.advance_to(1.0, context)
        context = sim_state.context
        xd1 = context.discrete_state
        assert xd1 == xd0**3

        # Advance the simulator to just after the second update
        sim_state = simulator.advance_to(1.001, context)
        context = sim_state.context
        xd2 = context.discrete_state
        assert xd2 == xd1**3

    def test_long_simulation(self):
        # Make sure that if we run a simulation for ~10 years, we don't get overflow
        # in the integer time representation.
        tf = 3.156e8  # 10 years
        dt = 0.01 * tf
        system = DiscreteClock(dt)
        context = system.create_context()

        # First check that the default fails
        with pytest.raises(RuntimeError):
            results = collimator.simulate(
                system,
                context,
                (0.0, tf),
            )

        # Then if we increase the time scale to nanoseconds, it should work
        ns_scale = 1e-9
        options = collimator.SimulatorOptions(int_time_scale=ns_scale)
        results = collimator.simulate(
            system,
            context,
            (0.0, tf),
            options=options,
        )
        assert jnp.allclose(results.context.time, tf)


class DampedLinear(collimator.LeafSystem):
    def __init__(self, a=1.0, name="plant"):
        super().__init__(name=name)

        self.a = a

        self.declare_input_port(name=f"{name}:tau")  # One input named "tau".

        self.declare_continuous_state((), ode=self.ode)  # Two state variables.

        # One output named "x"
        self.declare_continuous_state_output(name=f"{name}:y")

    def ode(self, time, state, u):
        logger.debug(state)
        x = state.continuous_state
        return -self.a * x + u


class Actuator(collimator.LeafSystem):
    def __init__(self, name="actuator"):
        super().__init__(name=name)

        self.declare_discrete_state(default_value=0.0)

        self.declare_output_port(
            self.output,
            period=10.0,
            offset=10.0,
            name=f"{name}:y",
        )
        self.declare_periodic_update(
            self.update,
            period=10.0,
            offset=10.0,
        )

    # x[n+1] = x[n] + 1.0
    def update(self, time, state, *inputs):
        x = state.discrete_state
        return x + 1.0

    # y[n] = x[n]
    def output(self, time, state, *inputs):
        return state.discrete_state


class TestHybridTime:
    def _make_diagram(self, dt=0.1):
        builder = collimator.DiagramBuilder()

        plant = builder.add(DampedLinear())

        actuator = builder.add(Actuator())

        builder.connect(actuator.output_ports[0], plant.input_ports[0])
        return builder.build()

    @pytest.mark.parametrize("enable_tracing", [True, False])
    def test_heavy_damping(self, enable_tracing):
        # Stop just short of the final update so that the final state should
        # match the final input.
        tf = 99.9
        dt = 2.0
        diagram = self._make_diagram(dt=dt)
        logger.info("*** Built diagram ***")

        plant = diagram["plant"]
        actuator = diagram["actuator"]

        # Create a context for this system
        context = diagram.create_context()
        plant_context = context[plant.system_id].with_continuous_state(1.0)
        context = context.with_subcontext(plant.system_id, plant_context)
        act_context = context[actuator.system_id].with_discrete_state(0.0)
        context = context.with_subcontext(actuator.system_id, act_context)
        logger.info("*** Created context ***")

        options = SimulatorOptions(
            enable_tracing=enable_tracing,
            atol=1e-8,
            rtol=1e-6,
        )
        recorded_signals = {
            "plant": diagram["plant"].output_ports[0],
            "actuator": diagram["actuator"].output_ports[0],
        }
        results = collimator.simulate(
            diagram,
            context,
            (0.0, tf),
            options=options,
            recorded_signals=recorded_signals,
        )
        context = results.context

        # Check that the final state in the solution approaches the input
        xf = context[plant.system_id].continuous_state
        (uf,) = context[actuator.system_id].cache
        assert jnp.allclose(xf, uf)

        # Check that the final entry in the "solution" matches final state
        assert jnp.allclose(results.outputs["plant"][-1], xf)
        assert jnp.allclose(results.outputs["actuator"][-1], uf)


def test_ffwd(show_plot=False):
    """
    test simple feed forward model (no states, no discrete blocks)
    to verify that some results data is proiduced as expected.
    """
    builder = collimator.DiagramBuilder()
    sw = builder.add(Sine(name="sw"))
    gain = builder.add(Gain(name="gain", gain=2.0))
    # int_ = builder.add(Integrator(0.0, name="Integrator_0"))

    builder.connect(sw.output_ports[0], gain.input_ports[0])
    # builder.connect(gain.output_ports[0], int_.input_ports[0])

    tf = 10.0
    dt = 0.1
    diagram = builder.build()
    logger.info("*** Built diagram ***")

    # Create a context for this system
    context = diagram.create_context()
    logger.info("*** Created context ***")

    options = SimulatorOptions(
        max_major_step_length=dt,
    )
    recorded_signals = {"sw": sw.output_ports[0], "gain": gain.output_ports[0]}
    results = collimator.simulate(
        diagram,
        context,
        (0.0, tf),
        options=options,
        recorded_signals=recorded_signals,
    )

    if show_plot:
        time = results.time
        fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

        ax1.plot(time, results.outputs["sw"], label="sw", marker="x")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(time, results.outputs["gain"], label="gain", marker="x")
        ax2.grid(True)
        ax2.legend()

        plt.show()

    # Check that the results match the expectation
    assert len(results.outputs["sw"]) == 101
    assert jnp.allclose(results.outputs["sw"], jnp.sin(results.time))
    assert jnp.allclose(results.outputs["gain"], jnp.sin(results.time) * 2)


def test_discrete_albert(show_plot=False):
    """
    test simple discrete system for verifying the returned solution.
    """
    dt = 1.0
    builder = collimator.DiagramBuilder()
    dclk = builder.add(DiscreteClock(dt=dt))
    ud = builder.add(UnitDelay(dt=dt, initial_state=0.0))

    builder.connect(dclk.output_ports[0], ud.input_ports[0])

    tf = 10.0

    diagram = builder.build()
    context = diagram.create_context()

    # there are no events, so we should be able to predict the required major
    # steps accurately. tf/dt=10/1=10, then add one for the last sample, so 11.
    options = SimulatorOptions(
        max_major_steps=11,
        save_time_series=True,
    )
    recorded_signals = {
        "dclk": dclk.output_ports[0],
        "ud": ud.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, tf),
        options=options,
        recorded_signals=recorded_signals,
    )

    # analytical solution
    time = results.time
    print(f"time.shape={time.shape} before")
    print(f"time=\n{time}")

    # if the results samples are missing the last time sample,
    # create a new time vector which includes it, so that the
    # expected solutions here are computed including last time
    # sample.
    if time[-1] < tf:
        last_time_sample = np.array([tf])
        time_sol = np.concatenate((time, last_time_sample))
        print(f"time_sol.shape={time_sol.shape} after")
    else:
        time_sol = time

    # compute the discrete expected solution
    time_sol = np.arange(11.0)
    dclk_sol = time_sol
    ud_sol = np.roll(dclk_sol, 1)
    ud_sol[0] = 0.0

    print(f"time_sol=\n{time_sol}")
    print(f"dclk_sol=\n{dclk_sol}")
    print(f"ud_sol=\n{ud_sol}")

    if show_plot:
        fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

        ax1.plot(time, results.outputs["dclk"], label="dclk", marker="x")
        ax1.plot(time_sol, dclk_sol, label="dclk_sol", marker="o")
        ax1.grid(True)
        ax1.set_ylim(bottom=-1, top=11)
        ax1.legend()

        ax2.plot(time, results.outputs["ud"], label="ud", marker="x")
        ax2.plot(time_sol, ud_sol, label="ud_sol", marker="o")
        ax2.grid(True)
        ax2.set_ylim(bottom=-1, top=11)
        ax2.legend()

        plt.show()

    # Check that the results match the expectation
    assert jnp.allclose(time, time_sol)
    assert jnp.allclose(results.outputs["dclk"], dclk_sol)
    assert jnp.allclose(results.outputs["ud"], ud_sol)


def test_hybrid_albert(show_plot=False):
    """
    test simple hybrid system with analytical solution.
    """
    dt = 1.0
    builder = collimator.DiagramBuilder()
    int0 = builder.add(Integrator(name="int0", initial_state=0.0))
    int1 = builder.add(Integrator(name="int1", initial_state=1.0))
    gain = builder.add(Gain(name="gain", gain=-4.0))
    dclk = builder.add(DiscreteClock(dt=dt))
    ud = builder.add(UnitDelay(dt=dt, initial_state=0.0))

    builder.connect(gain.output_ports[0], int0.input_ports[0])
    builder.connect(int0.output_ports[0], int1.input_ports[0])
    builder.connect(int1.output_ports[0], gain.input_ports[0])
    builder.connect(dclk.output_ports[0], ud.input_ports[0])

    tf = 10.0

    diagram = builder.build()
    context = diagram.create_context()

    # there are no events, so we should be able to predict the required major
    # steps accurately. tf/dt=10/1=10, then add one for the last sample, so 11.
    options = SimulatorOptions(
        max_major_steps=11,
        save_time_series=True,
        atol=1e-8,
        rtol=1e-6,
    )
    recorded_signals = {
        "int0": int0.output_ports[0],
        "int1": int1.output_ports[0],
        "dclk": dclk.output_ports[0],
        "ud": ud.output_ports[0],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, tf),
        options=options,
        recorded_signals=recorded_signals,
    )

    # analytical solution
    time = results.time
    print(f"time.shape={time.shape} before")

    # if the results samples are missing the last time sample,
    # create a new time vector which includes it, so that the
    # expected solutions here are computed including last time
    # sample.
    if time[-1] < tf:
        last_time_sample = np.array([tf])
        time_sol = np.concatenate((time, last_time_sample))
        print(f"time_sol.shape={time_sol.shape} after")
    else:
        time_sol = time

    # compute extected continuous solution using sim time samples
    # so we have arrays same length and can comute error.
    int0_sol = np.sin(time * 2.0) * -2.0
    int1_sol = np.cos(time * 2.0)

    # compute the discrete expected solution using the time samples
    # thta we expected in the solution.
    dclk_sol = np.floor(time_sol)
    ud_sol = np.zeros(time_sol.shape)
    # "delay" the dclk_sol samples by a unit time 'dt'
    for idx, t in enumerate(dclk_sol):
        _idx = np.argmin(np.abs(dclk_sol - t + dt))
        ud_sol[idx] = dclk_sol[_idx]

    print(f"time_sol=\n{time_sol}")
    print(f"dclk_sol=\n{dclk_sol}")
    print(f"ud_sol=\n{ud_sol}")

    if show_plot:
        fig02, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

        ax1.plot(time, results.outputs["dclk"], label="dclk")
        ax1.plot(time_sol, dclk_sol, label="dclk_sol")
        ax1.grid(True)
        ax1.set_ylim(bottom=-1, top=11)
        ax1.legend()

        ax2.plot(time, results.outputs["ud"], label="ud")
        ax2.plot(time_sol, ud_sol, label="ud_sol")
        ax2.grid(True)
        ax2.set_ylim(bottom=-1, top=11)
        ax2.legend()

        # fig02, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(9, 12))

        # ax3.plot(time, int0_sol - results.outputs["int0"], label="err")
        # ax3.grid(True)
        # ax3.legend()

        # ax4.plot(time, int1_sol - results.outputs["int1"], label="err")
        # ax4.grid(True)
        # ax4.legend()

        # ax5.plot(time, results.outputs["int0"], label="int0")
        # ax5.plot(time, int0_sol, label="int0_sol")
        # ax5.grid(True)
        # ax5.legend()

        plt.show()

    # Check that the results match the expectation
    assert jnp.allclose(results.outputs["int0"], int0_sol, atol=1e-4)
    assert jnp.allclose(results.outputs["int1"], int1_sol, atol=1e-4)
    assert jnp.allclose(results.outputs["dclk"], dclk_sol)
    assert jnp.allclose(results.outputs["ud"], ud_sol)


def test_notimplemented_output_mode():
    sys = Sine()
    context = sys.create_context()
    results_options = ResultsOptions(mode=ResultsMode.discrete_steps_only)

    with pytest.raises(NotImplementedError) as e:
        collimator.simulate(
            sys,
            context,
            (0.0, 0.1),
            results_options=results_options,
        )

    print(e)

    assert (
        "Simulation output mode discrete_steps_only is not supported. Only 'auto' is presently supported."
        in str(e)
    )


if __name__ == "__main__":
    # test_ffwd()
    # TestHybridTime().test_heavy_damping()
    # test_ffwd(show_plot=True)
    # test_discrete_albert(show_plot=True)
    # test_hybrid_albert(show_plot=True)
    test_notimplemented_output_mode()
    pass
